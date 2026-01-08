#!/usr/bin/env python3
"""
API FastAPI pour tester le modèle GGUF avec llama.cpp
Permet de générer du texte français via une API REST.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import json
from pathlib import Path
from typing import Optional
import uvicorn


app = FastAPI(
    title="French LLM V3-Mistral Test API",
    description="API pour tester le modèle GGUF avec llama.cpp",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
MODEL_PATH = Path("trained_models/french-llm-v3-mistral-f16-v5.gguf")
LLAMA_CLI = Path("/home/vincent/llama.cpp/build/bin/llama-cli")


class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    threads: int = 4


class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_used: int
    model_info: dict


@app.get("/health")
def health():
    """Vérifier la santé de l'API."""
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "llama_cli_exists": LLAMA_CLI.exists(),
    }


@app.get("/model-info")
def model_info():
    """Obtenir les informations du modèle."""
    config_path = Path("trained_models/french-llm-v3-mistral-hf-compatible/config.json")
    
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    
    with open(config_path) as f:
        config = json.load(f)
    
    return {
        "model": "French-LLM-V3-Mistral",
        "vocab_size": config.get("vocab_size"),
        "hidden_size": config.get("hidden_size"),
        "num_layers": config.get("num_hidden_layers"),
        "num_heads": config.get("num_attention_heads"),
        "max_position": config.get("max_position_embeddings"),
        "gguf_path": str(MODEL_PATH),
        "gguf_size_mb": MODEL_PATH.stat().st_size / 1e6 if MODEL_PATH.exists() else 0,
    }


@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    """Générer du texte avec le modèle."""
    
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {MODEL_PATH}")
    
    if not LLAMA_CLI.exists():
        raise HTTPException(status_code=404, detail=f"llama-cli not found: {LLAMA_CLI}")
    
    try:
        # Préparer la commande
        cmd = [
            str(LLAMA_CLI),
            "-m", str(MODEL_PATH),
            "--prompt", request.prompt,
            "-n", str(request.max_tokens),
            "--temp", str(request.temperature),
            "--top-p", str(request.top_p),
            "-t", str(request.threads),
            "--no-display-prompt",  # Ne pas afficher le prompt dans la sortie
        ]
        
        # Exécuter llama.cpp
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"llama.cpp error: {result.stderr}"
            )
        
        # Parser la sortie
        generated_text = result.stdout.strip()
        
        # Compter les tokens (approximativement)
        tokens_used = len(generated_text.split())
        
        return GenerationResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            tokens_used=tokens_used,
            model_info={
                "model_name": "French-LLM-V3-Mistral",
                "temperature": request.temperature,
                "top_p": request.top_p,
                "threads": request.threads,
            }
        )
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Generation timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-generate")
def batch_generate(prompts: list[str]):
    """Générer du texte pour plusieurs prompts."""
    results = []
    
    for prompt in prompts:
        try:
            req = GenerationRequest(prompt=prompt, max_tokens=50)
            result = generate(req)
            results.append(result.dict())
        except Exception as e:
            results.append({"prompt": prompt, "error": str(e)})
    
    return {"results": results}


@app.get("/test-prompts")
def test_prompts():
    """Exécuter les prompts de test."""
    test_data = [
        "Bonjour, comment ça va",
        "La France est un pays",
        "L'intelligence artificielle",
        "Un jour, il y avait",
        "Paris est la capitale",
    ]
    
    results = []
    for prompt in test_data:
        try:
            req = GenerationRequest(prompt=prompt, max_tokens=50)
            result = generate(req)
            results.append({
                "prompt": prompt,
                "generated": result.generated_text,
            })
        except Exception as e:
            results.append({
                "prompt": prompt,
                "error": str(e)
            })
    
    return {"results": results}


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Lancement de l'API de test du modèle")
    print("=" * 80)
    print(f"\n📁 Modèle: {MODEL_PATH}")
    print(f"📁 llama-cli: {LLAMA_CLI}")
    print(f"\n✅ Modèle existe: {MODEL_PATH.exists()}")
    print(f"✅ llama-cli existe: {LLAMA_CLI.exists()}")
    print(f"\n🌐 API disponible à: http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs")
    print(f"📊 ReDoc: http://localhost:8000/redoc")
    print("\n" + "=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
