#!/usr/bin/env python3
"""
🚀 FRENCH LLM V2 - CHAT API INTERACTIVE
======================================
API FastAPI avec tokenizer SentencePiece pour chat interactif en français.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List
import json
import sys

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("❌ FastAPI required - install: pip install fastapi uvicorn")
    sys.exit(1)

try:
    from sentencepiece import SentencePieceProcessor
except ImportError:
    print("❌ sentencepiece required - install: pip install sentencepiece")
    sys.exit(1)


# ============================================================================
# MODÈLE LLAMA-COMPATIBLE
# ============================================================================

class SimpleTransformer(nn.Module):
    """Modèle Transformer simple."""
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


# ============================================================================
# TOKENIZER SENTENCEPIECE
# ============================================================================

class SPMTokenizer:
    """Wrapper SentencePiece tokenizer."""
    
    def __init__(self, model_path: str):
        self.sp = SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = self.sp.vocab_size()
        
        # Tokens spéciaux
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()
        self.pad_id = 0
        
    def encode(self, text: str) -> List[int]:
        """Encode texte en IDs."""
        ids = self.sp.EncodeAsIds(text)
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode IDs en texte."""
        text = self.sp.DecodeIds(ids)
        return text
    
    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode texte en pieces (pour debug)."""
        return self.sp.EncodeAsPieces(text)


# ============================================================================
# CACHE GLOBAL
# ============================================================================

MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "checkpoint_step": None
}


def load_model():
    """Charge le modèle Phase 2A (110k) avec tokenizer SentencePiece."""
    
    if MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["model"], MODEL_CACHE["tokenizer"], MODEL_CACHE["device"]
    
    print("\n🔧 Chargement modèle Phase 2A...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Checkpoint
    checkpoint_path = Path("trained_models/runs/french_v2_phase2a_final/checkpoint_step_110000.pt")
    
    if not checkpoint_path.exists():
        raise Exception(f"❌ Checkpoint not found: {checkpoint_path}")
    
    print(f"   Checkpoint: {checkpoint_path.name}")
    
    # Charger checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Modèle
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
    
    # Tokenizer SentencePiece
    tokenizer_path = "trained_models/tokenizers/vincent_tokenizer_FR_v2/tokenizer.model"
    if not Path(tokenizer_path).exists():
        raise Exception(f"❌ Tokenizer not found: {tokenizer_path}")
    
    print(f"   Tokenizer: vincent_tokenizer_FR_v2")
    tokenizer = SPMTokenizer(tokenizer_path)
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    # Cache
    MODEL_CACHE["model"] = model
    MODEL_CACHE["tokenizer"] = tokenizer
    MODEL_CACHE["device"] = device
    MODEL_CACHE["checkpoint_step"] = 110000
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modèle chargé! {params:,} paramètres")
    
    return model, tokenizer, device


@torch.no_grad()
def generate_chat(model, tokenizer, device, prompt: str, max_tokens=100, temperature=0.7, top_p=0.9):
    """Génère du texte avec sampling nucleus (top-p)."""
    
    # Encode
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated_ids = input_ids.copy()
    
    # Génération
    for _ in range(max_tokens):
        # Forward pass
        with torch.no_grad():
            try:
                output = model(input_tensor)
            except:
                break
        
        # Logits du dernier token
        logits = output[0, -1, :] / max(temperature, 0.01)
        probs = torch.softmax(logits, dim=-1)
        
        # Top-p filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
        
        # Sample
        try:
            next_id = torch.multinomial(probs, num_samples=1).item()
        except:
            next_id = torch.argmax(probs).item()
        
        # Check EOS
        if next_id == tokenizer.eos_id:
            break
        
        generated_ids.append(next_id)
        
        # Update input (keep last 512 tokens)
        input_tensor = torch.tensor([generated_ids[-512:]], dtype=torch.long, device=device)
    
    # Decode
    text = tokenizer.decode(generated_ids)
    return text, len(generated_ids)


# ============================================================================
# FASTAPI
# ============================================================================

app = FastAPI(
    title="French LLM V2 - Chat API",
    description="API interactive pour tester le modèle en chat français",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9


class ChatResponse(BaseModel):
    user_message: str
    model_response: str
    num_tokens: int
    checkpoint_step: int


@app.get("/")
def root():
    return {
        "name": "French LLM V2 - Chat API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "POST /chat": "Chat avec le modèle",
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
    model, tokenizer, device = load_model()
    return {
        "model": "SimpleTransformer (Phase 2A)",
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": tokenizer.vocab_size,
        "checkpoint_step": 110000,
        "device": str(device),
        "tokenizer": "SentencePiece (vincent_tokenizer_FR_v2)",
        "dataset": "169M tokens (95% FR + 5% EN)",
        "final_loss": 27.44,
        "architecture": {
            "d_model": 1024,
            "num_layers": 18,
            "num_heads": 16,
            "dim_feedforward": 4096
        }
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat interactif avec le modèle."""
    
    try:
        model, tokenizer, device = load_model()
        
        response, num_tokens = generate_chat(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return ChatResponse(
            user_message=request.message,
            model_response=response,
            num_tokens=num_tokens,
            checkpoint_step=110000
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def main():
    print("\n" + "="*70)
    print("🚀 FRENCH LLM V2 - CHAT API (INTERACTIVE)")
    print("="*70)
    
    # Pre-load
    try:
        load_model()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    print("\n📡 Server starting on http://localhost:9002")
    print("   POST /chat   - Chat avec le modèle")
    print("   GET /info    - Info modèle")
    print("   GET /health  - API health")
    print("\n🧪 Test commands:")
    print("   curl -X POST http://localhost:9002/chat \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"message\":\"Bonjour, comment allez-vous ?\"}'")
    print("\n   curl http://localhost:9002/info\n")
    
    uvicorn.run(app, host="0.0.0.0", port=9002, log_level="warning")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
