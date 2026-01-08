#!/usr/bin/env python3
"""
🚀 API DE TEST MODÈLE FRENCH LLM V2
===================================
API FastAPI pour tester le modèle avec génération de texte réelle.
Utilise le tokenizer et le modèle entraîné.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List
import json
import sys

# Add root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    print("❌ FastAPI not available - install: pip install fastapi uvicorn")
    sys.exit(1)

# Import model from simple_train
try:
    from simple_train import SimpleTransformer
    print("✅ Model imported from simple_train.py")
except:
    print("⚠️  Defining SimpleTransformer locally...")
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=32000, d_model=1024, nhead=16, num_layers=18, dim_feedforward=4096):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, vocab_size)
            self.d_model = d_model

        def forward(self, src):
            x = self.embedding(src) * (self.d_model ** 0.5)
            x = self.encoder(x)
            return self.head(x)


# Tokenizer simple (basique)
class SimpleTokenizer:
    """Tokenizer basique pour tests rapides."""
    
    def __init__(self):
        self.vocab_size = 32000
        # Tokens spéciaux
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0
        
    def encode(self, text: str) -> List[int]:
        """Encode texte en IDs (méthode basique)."""
        # Pour tests: hash des mots
        words = text.lower().split()
        ids = [self.bos_id]
        for word in words:
            # Hash simple pour avoir des IDs cohérents
            word_id = (hash(word) % (self.vocab_size - 100)) + 100
            ids.append(word_id)
        ids.append(self.eos_id)
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode IDs en texte (méthode basique)."""
        # Pour tests: afficher les IDs
        tokens = []
        for id in ids:
            if id == self.bos_id:
                tokens.append("<BOS>")
            elif id == self.eos_id:
                tokens.append("<EOS>")
            elif id == self.pad_id:
                tokens.append("<PAD>")
            else:
                tokens.append(f"[{id}]")
        return " ".join(tokens)


# Global model cache
MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "checkpoint_step": None
}


def load_model_and_tokenizer():
    """Charge le modèle et le tokenizer."""
    
    if MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["model"], MODEL_CACHE["tokenizer"], MODEL_CACHE["device"]
    
    print("\n🔧 Chargement du modèle...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Trouver le dernier checkpoint
    checkpoint_dir = Path("trained_models/runs/french_v2_phase2a_final")
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    
    if not checkpoints:
        raise Exception("❌ Aucun checkpoint trouvé!")
    
    latest = checkpoints[-1]
    step = int(latest.stem.split('_')[-1])
    print(f"   Checkpoint: {latest.name} (step {step})")
    
    # Charger checkpoint
    checkpoint = torch.load(latest, map_location=device, weights_only=False)
    
    # Créer modèle
    model = SimpleTransformer(
        vocab_size=32000,
        d_model=1024,
        nhead=16,
        num_layers=18,
        dim_feedforward=4096
    )
    
    # Charger poids
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Tokenizer
    tokenizer = SimpleTokenizer()
    
    # Cache
    MODEL_CACHE["model"] = model
    MODEL_CACHE["tokenizer"] = tokenizer
    MODEL_CACHE["device"] = device
    MODEL_CACHE["checkpoint_step"] = step
    
    print(f"✅ Modèle chargé! {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    return model, tokenizer, device


# FastAPI app
app = FastAPI(
    title="French LLM V2 - Test API",
    description="API pour tester le modèle French LLM V2 avec génération de texte",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    generated_ids: List[int]
    num_tokens: int
    model_step: int
    note: str


@app.get("/")
def root():
    return {
        "name": "French LLM V2 Test API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "POST /generate": "Générer du texte",
            "GET /info": "Info modèle",
            "GET /health": "Santé API"
        }
    }


@app.get("/info")
def get_info():
    """Info sur le modèle."""
    
    # Charger si pas déjà fait
    if MODEL_CACHE["model"] is None:
        try:
            load_model_and_tokenizer()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Erreur chargement: {e}")
    
    checkpoint_dir = Path("trained_models/runs/french_v2_phase2a_final")
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    
    return {
        "model": "SimpleTransformer",
        "architecture": "260M parameters, 18 layers, 16 heads",
        "checkpoint_step": MODEL_CACHE["checkpoint_step"],
        "total_checkpoints": len(checkpoints),
        "device": str(MODEL_CACHE["device"]),
        "tokenizer": "Simple (basique pour tests)",
        "vocab_size": 32000,
        "status": "loaded" if MODEL_CACHE["model"] else "not_loaded"
    }


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": MODEL_CACHE["model"] is not None
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Génère du texte à partir d'un prompt."""
    
    try:
        # Charger modèle
        model, tokenizer, device = load_model_and_tokenizer()
        
        print(f"\n🎯 Génération:")
        print(f"   Prompt: {req.prompt}")
        print(f"   Max tokens: {req.max_tokens}")
        print(f"   Temperature: {req.temperature}")
        
        # Encoder le prompt
        input_ids = tokenizer.encode(req.prompt)
        print(f"   Input IDs: {input_ids[:10]}... ({len(input_ids)} tokens)")
        
        # Préparer tensor
        input_tensor = torch.tensor([input_ids], device=device)
        
        # Générer
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for step in range(req.max_tokens):
                # Forward
                logits = model(input_tensor)
                next_logits = logits[0, -1, :] / req.temperature
                
                # Top-k filtering
                if req.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_logits, min(req.top_k, next_logits.size(-1)))
                    
                    # Top-p filtering
                    if req.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(top_k_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > req.top_p
                        sorted_indices_to_remove[0] = False  # Keep at least one token
                        
                        indices_to_keep = sorted_indices[~sorted_indices_to_remove]
                        logits_to_keep = sorted_logits[~sorted_indices_to_remove]
                        
                        probs = torch.softmax(logits_to_keep, dim=-1)
                        next_token_idx = torch.multinomial(probs, 1)
                        next_token = top_k_indices[indices_to_keep[next_token_idx]]
                    else:
                        probs = torch.softmax(top_k_logits, dim=-1)
                        next_token = top_k_indices[torch.multinomial(probs, 1)]
                else:
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                
                next_token_id = next_token.item()
                generated_ids.append(next_token_id)
                
                # Stop si EOS
                if next_token_id == tokenizer.eos_id:
                    print(f"   ✅ EOS atteint au step {step + 1}")
                    break
                
                # Ajouter au tensor
                new_token_tensor = next_token.view(1, 1)
                input_tensor = torch.cat([input_tensor, new_token_tensor], dim=1)
                
                # Limiter contexte
                if input_tensor.size(1) > 512:
                    input_tensor = input_tensor[:, -512:]
                
                if (step + 1) % 20 == 0:
                    print(f"   Generated {step + 1}/{req.max_tokens} tokens...")
        
        # Decoder
        generated_text = tokenizer.decode(generated_ids)
        
        print(f"✅ Génération terminée: {len(generated_ids)} tokens")
        
        return GenerateResponse(
            prompt=req.prompt,
            generated_text=generated_text,
            generated_ids=generated_ids[:50],  # Limiter pour la réponse
            num_tokens=len(generated_ids),
            model_step=MODEL_CACHE["checkpoint_step"],
            note="Tokenizer basique - Pour tests uniquement. Utilisez SentencePiece pour production."
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur génération: {str(e)}")


def main():
    print("\n" + "="*70)
    print("🚀 API DE TEST - FRENCH LLM V2")
    print("="*70)
    print("\n📋 Endpoints:")
    print("   POST /generate  - Génération de texte")
    print("   GET  /info      - Informations modèle")
    print("   GET  /health    - Health check")
    print("\n🌐 Accès:")
    print("   http://localhost:9000")
    print("\n📖 Documentation:")
    print("   http://localhost:9000/docs")
    print("\n💡 Test rapide:")
    print('   curl -X POST http://localhost:9000/generate \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"prompt": "Bonjour je suis", "max_tokens": 50}\'')
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")


if __name__ == '__main__':
    main()
