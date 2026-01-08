#!/usr/bin/env python3
"""
🚀 FastAPI Server - Test French GPT-2 Model
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="French GPT-2 API",
    description="Test API for French GPT-2 model",
    version="1.0.0"
)

# Global model and tokenizer
model = None
tokenizer = None

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer
    
    logger.info("🔄 Loading model...")
    
    try:
        model_path = "models/french_gpt2_lm_studio"
        
        # Load tokenizer
        logger.info(f"📝 Loading tokenizer from {model_path}...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        logger.info(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
        
        # Load model
        logger.info(f"🤖 Loading model from {model_path}...")
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.eval()
        logger.info(f"✅ Model loaded ({model.config.model_type})")
        logger.info(f"   Parameters: {model.num_parameters() / 1e6:.0f}M")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "French GPT-2 API",
        "status": "running",
        "model": "GPT-2 (124M)",
        "language": "French",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "info": "/info"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.get("/info")
async def info():
    """Get model info"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model.config.model_type,
        "parameters": model.num_parameters(),
        "vocab_size": tokenizer.vocab_size,
        "context_length": model.config.n_ctx,
        "hidden_size": model.config.n_embd,
        "num_layers": model.config.n_layer,
        "num_heads": model.config.n_head
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from prompt"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"📝 Generating for prompt: {request.prompt[:50]}...")
        
        # Tokenize
        inputs = tokenizer(request.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        logger.info(f"   Input tokens: {input_ids.shape}")
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        tokens_generated = output_ids.shape[1] - input_ids.shape[1]
        
        logger.info(f"✅ Generated {tokens_generated} tokens")
        
        return GenerationResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            tokens_generated=tokens_generated
        )
    
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-simple")
async def generate_simple(prompt: str, max_length: int = 100):
    """Simple text generation (for testing)"""
    
    request = GenerationRequest(
        prompt=prompt,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9
    )
    return await generate(request)

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 STARTING FRENCH GPT-2 API")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
