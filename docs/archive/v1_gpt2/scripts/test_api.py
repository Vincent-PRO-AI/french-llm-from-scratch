#!/usr/bin/env python3
"""API FastAPI pour tester le modèle French LLM en temps réel."""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  FastAPI not installed")
    print("Installation: pip install fastapi uvicorn")

# Configuration
CHECKPOINT_DIR = Path("trained_models/runs/french_v2_phase2a_final")
METRICS_FILE = CHECKPOINT_DIR / "metrics.jsonl"


class SimpleTransformer(nn.Module):
    """Modèle Transformer simple."""
    def __init__(self, vocab_size=32000, d_model=1024, nhead=16, num_layers=18, dim_feedforward=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 5000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src):
        seq_len = src.size(1)
        x = self.embedding(src) * (self.d_model ** 0.5)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        return self.fc_out(x)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 50


class ModelInfo(BaseModel):
    status: str
    current_step: Optional[int]
    loss: Optional[float]
    checkpoint_path: Optional[str]
    model_size: str
    parameters: int
    last_update: Optional[str]


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="French LLM V2 API",
        description="API pour tester le modèle French LLM en temps réel",
        version="2.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Cache global pour le modèle
    model_cache: Dict[str, Any] = {
        "model": None,
        "device": None,
        "checkpoint_path": None,
        "step": None
    }


    def load_latest_checkpoint():
        """Charge le dernier checkpoint disponible."""
        checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"))
        
        if not checkpoints:
            return None, None
        
        latest = checkpoints[-1]
        step = int(latest.stem.split('_')[-1])
        
        # Si déjà chargé, ne pas recharger
        if model_cache["checkpoint_path"] == str(latest):
            return model_cache["model"], step
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Charger le checkpoint
            checkpoint = torch.load(latest, map_location=device, weights_only=False)
            
            # Créer le modèle
            model = SimpleTransformer(
                vocab_size=32000,
                d_model=1024,
                nhead=16,
                num_layers=18,
                dim_feedforward=4096
            )
            
            # Charger les poids
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            
            # Mettre en cache
            model_cache["model"] = model
            model_cache["device"] = device
            model_cache["checkpoint_path"] = str(latest)
            model_cache["step"] = step
            
            return model, step
            
        except Exception as e:
            print(f"❌ Erreur chargement checkpoint: {e}")
            return None, None


    def get_latest_metrics():
        """Récupère les dernières métriques."""
        if not METRICS_FILE.exists():
            return None
        
        try:
            with open(METRICS_FILE, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    return json.loads(last_line)
        except:
            return None
        return None


    def simple_tokenize(text: str) -> list:
        """Tokenisation simple basique (à améliorer avec le vrai tokenizer)."""
        # Pour l'instant, utilise des IDs aléatoires
        # TODO: Intégrer le vrai tokenizer SentencePiece
        words = text.lower().split()
        # IDs entre 100 et 31999 (éviter les tokens spéciaux)
        return [hash(w) % 31900 + 100 for w in words]


    def simple_detokenize(token_ids: list) -> str:
        """Détokenisation simple basique."""
        # Pour l'instant, renvoie les IDs
        # TODO: Intégrer le vrai tokenizer SentencePiece
        return " ".join([f"[{tid}]" for tid in token_ids])


    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Point d'entrée de l'API."""
        return {
            "name": "French LLM V2 API",
            "version": "2.0.0",
            "status": "running",
            "endpoints": [
                "/info - Informations sur le modèle",
                "/metrics - Dernières métriques de training",
                "/generate - Génération de texte",
                "/health - Santé de l'API"
            ]
        }


    @app.get("/info", response_model=ModelInfo)
    async def get_model_info():
        """Récupère les informations sur le modèle."""
        metrics = get_latest_metrics()
        checkpoints = list(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"))
        
        if checkpoints:
            latest = sorted(checkpoints)[-1]
            step = int(latest.stem.split('_')[-1])
            checkpoint_path = str(latest)
        else:
            step = None
            checkpoint_path = None
        
        return ModelInfo(
            status="training" if step and step < 110000 else "completed",
            current_step=step,
            loss=metrics.get('loss') if metrics else None,
            checkpoint_path=checkpoint_path,
            model_size="Medium (260M params)",
            parameters=260_000_000,
            last_update=metrics.get('timestamp') if metrics else None
        )


    @app.get("/metrics")
    async def get_metrics():
        """Récupère les dernières métriques de training."""
        metrics = get_latest_metrics()
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Pas de métriques disponibles")
        
        # Charger toutes les métriques pour statistiques
        all_losses = []
        try:
            with open(METRICS_FILE, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        all_losses.append(data['loss'])
                    except:
                        continue
        except:
            pass
        
        return {
            "current": metrics,
            "statistics": {
                "min_loss": min(all_losses) if all_losses else None,
                "max_loss": max(all_losses) if all_losses else None,
                "avg_loss": sum(all_losses) / len(all_losses) if all_losses else None,
                "total_steps": len(all_losses)
            }
        }


    @app.get("/bilan")
    async def get_bilan():
        """Bilan complet du training."""
        metrics = get_latest_metrics()
        checkpoints = list(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"))
        
        # Analyser toutes les métriques
        all_metrics = []
        try:
            with open(METRICS_FILE, 'r') as f:
                for line in f:
                    try:
                        all_metrics.append(json.loads(line))
                    except:
                        continue
        except:
            pass
        
        if not all_metrics:
            return {"error": "Pas de métriques disponibles"}
        
        losses = [m['loss'] for m in all_metrics]
        speeds = [m.get('speed_steps_per_sec', 0) for m in all_metrics if 'speed_steps_per_sec' in m]
        
        # Calculer les tendances
        recent_50 = losses[-50:] if len(losses) >= 50 else losses
        recent_10 = losses[-10:] if len(losses) >= 10 else losses
        
        first_step = all_metrics[0]['step']
        last_step = all_metrics[-1]['step']
        total_steps = last_step - first_step
        
        # Analyse overfit
        min_loss = min(losses)
        current_loss = losses[-1]
        overfit_score = current_loss - min_loss
        
        # Temps estimé
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        remaining_steps = 110000 - last_step
        eta_hours = remaining_steps / (avg_speed * 3600) if avg_speed > 0 else 0
        
        return {
            "training_status": {
                "phase": "Phase 2A",
                "current_step": last_step,
                "target_step": 110000,
                "progress_pct": round((total_steps / 10000) * 100, 2),
                "completed_steps": total_steps,
                "remaining_steps": remaining_steps
            },
            "performance": {
                "current_loss": round(current_loss, 2),
                "min_loss": round(min_loss, 2),
                "max_loss": round(max(losses), 2),
                "avg_loss": round(sum(losses) / len(losses), 2),
                "avg_recent_50": round(sum(recent_50) / len(recent_50), 2),
                "avg_recent_10": round(sum(recent_10) / len(recent_10), 2)
            },
            "speed": {
                "current_steps_per_sec": round(speeds[-1], 4) if speeds else 0,
                "avg_steps_per_sec": round(avg_speed, 4),
                "steps_per_hour": round(avg_speed * 3600, 0) if avg_speed > 0 else 0,
                "eta_hours": round(eta_hours, 2)
            },
            "health": {
                "overfit_score": round(overfit_score, 2),
                "overfit_status": "✅ Sain" if overfit_score < 10 else "⚠️  Surveillance" if overfit_score < 15 else "❌ Overfit",
                "loss_trend": "📉 Baisse" if sum(recent_10) < sum(recent_50) / 5 else "➡️ Stable" if abs(sum(recent_10) - sum(recent_50) / 5) < 1 else "📈 Hausse",
                "learning": "✅ Actif" if current_loss < losses[0] * 0.9 else "⚠️  Lent"
            },
            "checkpoints": {
                "total": len(checkpoints),
                "latest": str(sorted(checkpoints)[-1].name) if checkpoints else None,
                "total_size_mb": round(sum(c.stat().st_size for c in checkpoints) / (1024**2), 2) if checkpoints else 0
            },
            "dataset": {
                "name": "conversations_mega_train_mistral_tokenized.pt",
                "type": "95% Français, 5% Anglais",
                "size": "653 MB",
                "tokens": "169M tokens"
            },
            "model": {
                "architecture": "SimpleTransformer",
                "parameters": "260M",
                "layers": 18,
                "heads": 16,
                "d_model": 1024,
                "vocab_size": 32000
            },
            "next_steps": [
                f"Terminer Phase 2A ({remaining_steps} steps restants)",
                "Lancer Phase 2B sur dataset 100% FR",
                "Export HuggingFace format",
                "Conversion GGUF",
                "Publication"
            ]
        }


    @app.post("/generate")
    async def generate_text(request: GenerateRequest):
        """Génère du texte à partir d'un prompt."""
        model, step = load_latest_checkpoint()
        
        if model is None:
            raise HTTPException(status_code=503, detail="Modèle non disponible")
        
        try:
            # Tokeniser (basique pour l'instant)
            input_ids = simple_tokenize(request.prompt)
            
            if len(input_ids) == 0:
                input_ids = [100]  # Token par défaut
            
            input_tensor = torch.tensor([input_ids], device=model_cache["device"])
            
            # Générer
            with torch.no_grad():
                generated_ids = input_ids.copy()
                
                for _ in range(request.max_tokens):
                    # Forward pass
                    logits = model(input_tensor)
                    next_token_logits = logits[0, -1, :] / request.temperature
                    
                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, request.top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token = top_k_indices[torch.multinomial(probs, 1)]
                    
                    generated_ids.append(next_token.item())
                    
                    # Ajouter au tensor
                    input_tensor = torch.cat([
                        input_tensor,
                        next_token.unsqueeze(0).unsqueeze(0)
                    ], dim=1)
                    
                    # Limiter la longueur du contexte
                    if input_tensor.size(1) > 512:
                        input_tensor = input_tensor[:, -512:]
            
            # Détokeniser (basique)
            generated_text = simple_detokenize(generated_ids)
            
            return {
                "prompt": request.prompt,
                "generated": generated_text,
                "tokens_generated": len(generated_ids) - len(input_ids),
                "model_step": step,
                "note": "Génération basique - Tokenizer simplifié (IDs de hash)"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur génération: {str(e)}")


    @app.get("/health")
    async def health_check():
        """Vérifie la santé de l'API."""
        model, step = load_latest_checkpoint()
        metrics = get_latest_metrics()
        
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "current_step": step,
            "cuda_available": torch.cuda.is_available(),
            "metrics_available": metrics is not None,
            "timestamp": datetime.now().isoformat()
        }


def main():
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI non installé")
        print("Installation: pip install fastapi uvicorn")
        return
    
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 API French LLM V2")
    print("="*60)
    print("\n📊 Endpoints disponibles:")
    print("   • GET  /           - Info API")
    print("   • GET  /info       - Info modèle")
    print("   • GET  /metrics    - Métriques training")
    print("   • GET  /bilan      - Bilan complet")
    print("   • POST /generate   - Génération texte")
    print("   • GET  /health     - Santé API")
    print("\n🌐 Accès:")
    print("   http://localhost:8080")
    print("\n📖 Documentation interactive:")
    print("   http://localhost:8080/docs")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == '__main__':
    main()
