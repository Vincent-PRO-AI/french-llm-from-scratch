#!/usr/bin/env python3
"""
API simple de test du modèle GGUF avec llama.cpp
Utilise FastAPI pour expose une API REST
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
from pathlib import Path
import uvicorn


app = FastAPI(
    title="French LLM V3 Test",
    description="Test API for French LLM GGUF model",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("trained_models/french-llm-v3-mistral-f16-v5.gguf")
LLAMA_CLI = Path("/home/vincent/llama.cpp/build/bin/llama-cli")


class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9


@app.get("/health")
def health():
    """Vérifier que tout est opérationnel."""
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "llama_cli_exists": LLAMA_CLI.exists(),
    }


@app.post("/generate")
def generate(req: GenerationRequest):
    """Générer du texte avec le modèle GGUF."""
    
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    if not LLAMA_CLI.exists():
        raise HTTPException(status_code=404, detail="llama-cli not found")
    
    try:
        cmd = [
            str(LLAMA_CLI),
            "-m", str(MODEL_PATH),
            "--prompt", req.prompt,
            "-n", str(req.max_tokens),
            "--temp", str(req.temperature),
            "--top-p", str(req.top_p),
            "-t", "4",
            "--no-display-prompt",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"llama.cpp error: {result.stderr}"
            )
        
        return {
            "prompt": req.prompt,
            "generated_text": result.stdout.strip(),
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n🚀 API de test du modèle GGUF")
    print(f"   Modèle: {MODEL_PATH}")
    print(f"   llama-cli: {LLAMA_CLI}")
    print(f"\n✅ Disponible à: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
