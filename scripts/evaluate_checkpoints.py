#!/usr/bin/env python3
"""
Evaluate multiple checkpoints on specific prompts.
"""
import torch
import sys
import argparse
from pathlib import Path
from transformers import AutoTokenizer

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scripts.train_subtitles_transformer import TinyTransformerLM, Config, SubtitleTrainer

def load_checkpoint_and_generate(checkpoint_path, tokenizer_path, prompts):
    print(f"\n{'='*80}")
    print(f"EVALUATING: {checkpoint_path}")
    print(f"{'='*80}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer from {tokenizer_path}: {e}")
        return

    # 2. Load Checkpoint
    try:
        # Safe globals hack for newer torch versions if needed
        import pathlib
        try:
            torch.serialization.add_safe_globals([pathlib.PosixPath])
        except AttributeError:
            pass # Older torch versions don't have this
            
        print(f"Loading checkpoint...", end="", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(" Done.")
    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")
        return

    # 3. Model Config
    config_dict = checkpoint.get("config", {})
    if hasattr(config_dict, "__dict__"):
        config_dict = config_dict.__dict__
    
    # Force some defaults if missing
    cfg = Config()
    for k, v in config_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
            
    # Override important match-ups
    cfg.vocab_size = tokenizer.vocab_size
    cfg.device = device
    
    # 4. Initialize Model
    model = TinyTransformerLM(cfg)
    
    state_dict = checkpoint.get("model_state", checkpoint)
    if not isinstance(state_dict, dict):
        # Maybe it's directly the state dict
        state_dict = checkpoint
        
    # Filter state dict to match model if needed (handling DDP prefixes etc)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "") # Compile prefix
        if name.startswith("module."): # DDP prefix
            name = name[7:]
        new_state_dict[name] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"Warning loading state dict: {e}")

    model.to(device)
    model.eval()

    # 5. Generate
    for prompt in prompts:
        print(f"\n🔹 Prompt: {prompt}")
        print(f"🔸 Reponse: ", end="", flush=True)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Simple generation loop
        generated = input_ids
        max_new_tokens = 60
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(generated)
                next_token_logits = logits[:, -1, :]
                
                # Temperature sampling
                probs = torch.softmax(next_token_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Print token as we go
                token_str = tokenizer.decode(next_token[0])
                print(token_str, end="", flush=True)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        print() # Newline

def main():
    prompts = [
        "A quoi sert une RTX ?", 
        "Comment se rendre a Paris ?", 
        "Quelle est ta couleur préférée ?"
    ]
    
    tokenizer_path = "data_clean/mistral_tokenizer"
    
    checkpoints_to_test = [
        # 160k - The "Good" candidate
        "trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt",
        
        # 250k - The "Optimized" one (suspected Byte-level or broken)
        "trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt",
        
        # 140k - The "Grand" run
        "trained_models/runs/french_v3_finetune_grand_60k/checkpoint_step_140000.pt",

        # Old V2 (Archived) - Might fail if arch mismatch, but let's try
        "docs/archive/v1_gpt2/old_runs/trained_models/runs/french_v2_phase2b_100pct_fr/checkpoint_step_112000.pt"
    ]
    
    for ckpt in checkpoints_to_test:
        if Path(ckpt).exists():
            load_checkpoint_and_generate(ckpt, tokenizer_path, prompts)
        else:
            print(f"\n⚠️ Skipped missing: {ckpt}")

if __name__ == "__main__":
    main()
