#!/usr/bin/env python3
"""FastAPI inference server for French LLM model."""
import sys
sys.path.append('.')
import torch
from pathlib import Path
from scripts.train_subtitles_transformer import Config, SubtitleTrainer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch.nn.functional as F
from typing import Optional

# Load model once
print("Loading model...")
ckpt_path = Path('trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt')
if not ckpt_path.exists():
    # Try latest checkpoint
    run_dir = Path('trained_models/runs/french_v3_finetune_resample_1')
    checkpoints = sorted(run_dir.glob('checkpoint_step_*.pt'), reverse=True)
    if checkpoints:
        ckpt_path = checkpoints[0]
        print(f"Using latest checkpoint: {ckpt_path.name}")
    else:
        raise FileNotFoundError("No checkpoints found")

try:
    import pathlib as _pl
    torch.serialization.add_safe_globals([_pl.PosixPath])
except Exception:
    pass

ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model_state = ck.get('model_state') or ck.get('model')
ck_cfg = ck.get('config') or {}

cfg = Config()
cfg.device = 'cpu'
cfg.tokenizer_path = Path('data_clean/mistral_tokenizer')
if isinstance(ck_cfg, dict):
    cfg.embed_dim = ck_cfg.get('embed_dim', cfg.embed_dim)
    cfg.num_layers = ck_cfg.get('num_layers', cfg.num_layers)
    cfg.num_heads = ck_cfg.get('num_heads', cfg.num_heads)
    cfg.ff_hidden_dim = ck_cfg.get('ff_hidden_dim', cfg.ff_hidden_dim)
    cfg.vocab_size = ck_cfg.get('vocab_size', cfg.vocab_size)
    cfg.block_size = ck_cfg.get('block_size', cfg.block_size)

trainer = SubtitleTrainer(cfg)
model = trainer.model.to('cpu')
model.load_state_dict(model_state, strict=False)
model.eval()
print(f"✅ Model loaded from {ckpt_path.name}")

app = FastAPI(title="French LLM API", version="1.0")

class GenerateRequest(BaseModel):
    prompt: str = "Bonjour, je suis "
    max_tokens: int = 80
    temperature: float = 0.95
    top_k: int = 200

class GenerateResponse(BaseModel):
    prompt: str
    generated_ids: list
    temperature: float
    top_k: int

def sample(prompt_ids, max_new_tokens=40, temperature=1.0, top_k=50):
    x = torch.tensor(prompt_ids, dtype=torch.long, device='cpu').unsqueeze(0)
    out = x
    for _ in range(max_new_tokens):
        with torch.no_grad():
            y = model(out)
            if y.dim() == 3:
                logits = y[:, -1, :]
            else:
                logits = y
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                minv = v[:, -1].unsqueeze(1)
                logits = torch.where(logits < minv, torch.full_like(logits, -1e10), logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_token], dim=1)
    return out.squeeze(0).tolist()

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        # Encode prompt
        prompt_ids = []
        try:
            from typing import Any, cast
            tok = cast(Any, getattr(trainer, "tokenizer", None))
            if tok is not None:
                enc = tok.encode(request.prompt)
                prompt_ids = list(getattr(enc, "ids", enc))
            else:
                raise RuntimeError("Tokenizer not available")
        except Exception:
            prompt_ids = [ord(c) % cfg.vocab_size for c in request.prompt]
        
        if not prompt_ids:
            raise HTTPException(status_code=400, detail="Could not encode prompt")
        
        # Generate
        out_ids = sample(
            prompt_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        tail_ids = out_ids[len(prompt_ids):]
        
        return GenerateResponse(
            prompt=request.prompt,
            generated_ids=tail_ids,
            temperature=request.temperature,
            top_k=request.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model": ckpt_path.name}

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting API on http://localhost:8000")
    print("📝 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
