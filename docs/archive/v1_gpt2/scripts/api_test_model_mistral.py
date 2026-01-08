#!/usr/bin/env python3
"""
🚀 API DE TEST MODÈLE FRENCH LLM V2 - AVEC TOKENIZER MISTRAL
============================================================
API FastAPI pour tester le modèle avec génération de texte réelle.
Utilise le tokenizer Mistral (SentencePiece) pour des résultats corrects.
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

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False
    print("❌ transformers not available - install: pip install transformers")
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


# Global model cache
MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "checkpoint_step": None
}


def load_model_and_tokenizer():
    """Charge le modèle et le tokenizer Mistral."""
    
    if MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["model"], MODEL_CACHE["tokenizer"], MODEL_CACHE["device"]
    
    print("\n🔧 Chargement du modèle et tokenizer Mistral...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Charger Phase 2A (110k) - Meilleure qualité
    checkpoint_dirs = [
        Path("trained_models/runs/french_v2_phase2a_final")
    ]
    
    latest = None
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
            if checkpoints:
                latest = checkpoints[-1]
                break
    
    if latest is None:
        raise Exception("❌ Aucun checkpoint trouvé!")
    
    step = int(latest.stem.split('_')[-1])
    print(f"   Checkpoint: {latest.name} (step {step:,})")
    
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
    
    # Tokenizer Mistral
    print("   Loading Mistral tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    print(f"   Vocab size: {len(tokenizer):,}")
    
    # Cache
    MODEL_CACHE["model"] = model
    MODEL_CACHE["tokenizer"] = tokenizer
    MODEL_CACHE["device"] = device
    MODEL_CACHE["checkpoint_step"] = step
    
    print(f"✅ Modèle chargé! {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    return model, tokenizer, device


def clean_bpe_artifacts(text: str) -> str:
    """Nettoie les artefacts BPE du texte généré."""
    text = text.replace('Ġ', ' ')
    text = text.replace('Ċ', '\n')
    text = text.replace('ĉ', '\t')
    
    # Nettoyer espaces multiples
    import re
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    return text


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    device,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9
):
    """Génère du texte avec sampling nucleus (top-p)."""
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    generated_ids = input_ids[0].tolist()
    
    # Génération
    for _ in range(max_tokens):
        # Forward
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(input_ids)
        
        # Logits du dernier token
        logits = outputs[0, -1, :] / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # Append
        generated_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        # Limit context window (garde les derniers 512 tokens)
        if input_ids.size(1) > 512:
            input_ids = input_ids[:, -512:]
    
    # Decode
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    generated_text_clean = clean_bpe_artifacts(generated_text)
    
    return generated_text_clean, generated_ids


# FastAPI app
app = FastAPI(
    title="French LLM V2 - Test API (Mistral Tokenizer)",
    description="API pour tester le modèle French LLM V2 avec génération de texte et tokenizer Mistral",
    version="2.1.0"
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
    num_tokens: int
    model_step: int
    tokenizer: str


@app.get("/")
def root():
    return {
        "name": "French LLM V2 Test API (Mistral)",
        "version": "2.1.0",
        "status": "running",
        "endpoints": {
            "POST /generate": "Générer du texte",
            "GET /info": "Info modèle",
            "GET /health": "Santé API"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_CACHE["model"] is not None,
        "checkpoint_step": MODEL_CACHE["checkpoint_step"]
    }


@app.get("/info")
def info():
    model, tokenizer, device = load_model_and_tokenizer()
    
    return {
        "model": "SimpleTransformer",
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(tokenizer),
        "tokenizer": "Mistral SentencePiece",
        "checkpoint_step": MODEL_CACHE["checkpoint_step"],
        "device": str(device),
        "architecture": {
            "d_model": 1024,
            "num_layers": 18,
            "num_heads": 16,
            "dim_feedforward": 4096
        }
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Génère du texte à partir d'un prompt."""
    
    try:
        model, tokenizer, device = load_model_and_tokenizer()
        
        generated_text, generated_ids = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            num_tokens=len(generated_ids),
            model_step=MODEL_CACHE["checkpoint_step"],
            tokenizer="Mistral SentencePiece (32k vocab)"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération: {str(e)}")


def main():
    """Lance le serveur FastAPI."""
    
    print("\n" + "="*70)
    print("🚀 FRENCH LLM V2 - TEST API (MISTRAL TOKENIZER)")
    print("="*70)
    
    # Pre-load model
    try:
        load_model_and_tokenizer()
    except Exception as e:
        print(f"\n❌ Erreur chargement: {e}")
        return 1
    
    print("\n📡 Démarrage serveur sur http://localhost:9001")
    print("   Endpoints:")
    print("   - POST /generate : Générer du texte")
    print("   - GET /info      : Info modèle")
    print("   - GET /health    : Santé API")
    print("\n   Ctrl+C pour arrêter\n")
    
    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="info")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
