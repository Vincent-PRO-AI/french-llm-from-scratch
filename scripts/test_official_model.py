#!/usr/bin/env python3
"""
Diagnostic: Test the official TinyTransformerLM from the repo.
This script uses the exact class definition from train_subtitles_transformer.py
to verify if existing checkpoints are functional.
"""
import torch
import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scripts.train_subtitles_transformer import TinyTransformerLM, Config, auto_device
from transformers import AutoTokenizer

def test_checkpoint(checkpoint_path, tokenizer_path):
    device = auto_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 2. Load checkpoint metadata to reconstruct Config
    print(f"Loading checkpoint: {checkpoint_path}")
    import pathlib
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Check if config is in checkpoint
    if "config" in checkpoint:
        cfg_dict = checkpoint["config"]
        # Handle cases where cfg_dict is a Config object instead of dict
        if hasattr(cfg_dict, "__dict__"):
            cfg_dict = cfg_dict.__dict__
        cfg = Config(**{k: v for k, v in cfg_dict.items() if hasattr(Config, k)})
    else:
        # Fallback to Medium architecture (most common in V3)
        from scripts.train_subtitles_transformer import apply_arch_preset
        cfg = Config()
        apply_arch_preset(cfg, "medium")
        cfg.vocab_size = tokenizer.vocab_size

    # Ensure vocab size matches
    cfg.vocab_size = tokenizer.vocab_size
    cfg.device = str(device)
    
    # 3. Initialize model
    model = TinyTransformerLM(cfg)
    
    # 4. Load state dict
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint) # Direct load
        
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")

    # 5. Inference test
    prompt = "L'intelligence artificielle est"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)
    
    generated = input_ids
    with torch.no_grad():
        for _ in range(30):
            # The official TinyTransformerLM forward handles mask internally
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            
            # Greedy
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            token_str = tokenizer.decode(next_token[0])
            print(token_str, end="", flush=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print("\n\nFull output:", tokenizer.decode(generated[0]))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt")
    parser.add_argument("tokenizer", nargs="?", default="data_clean/mistral_tokenizer")
    args = parser.parse_args()
    
    if Path(args.checkpoint).exists():
        test_checkpoint(args.checkpoint, args.tokenizer)
    else:
        print(f"❌ Checkpoint not found: {args.checkpoint}")
