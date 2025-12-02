import os
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="French LLM from scratch - Inference API")

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

_MODEL = None
_TOKENIZER = None


def _resolve_model_path() -> str:
    local = os.getenv("MODEL_LOCAL_DIR")
    if local and os.path.isdir(local):
        return local
    return os.getenv("MODEL_ID", "vincent-pro-ai/french-llm-from-scratch")


def get_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = _resolve_model_path()
        _TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        _MODEL = AutoModelForCausalLM.from_pretrained(model_path)
        _MODEL.eval()
    return _MODEL, _TOKENIZER


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40


@app.get("/")
def root():
    """Serve the chat interface"""
    static_index = os.path.join(static_dir, "index.html")
    if os.path.exists(static_index):
        return FileResponse(static_index)
    return {"message": "French LLM API", "endpoints": ["/generate", "/healthz", "/docs"]}


@app.get("/healthz")
def healthz():
    # Ne charge pas le modèle ici, juste une sonde rapide
    return {"status": "ok", "model": _resolve_model_path()}


@app.post("/generate")
def generate(req: GenerateRequest):
    import torch

    model, tokenizer = get_model()
    inputs = tokenizer(req.prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id or 0,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"text": text}
