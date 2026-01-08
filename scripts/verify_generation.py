import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

# Add scripts/ to path to import the model class
sys.path.append("scripts")
# Try to import TinyTransformerLM, Config
# However, importing might run some code if not guarded, let's assume it is safe or just copy the class structure if it fails.
# Since I cannot easily import from a script that has no .py extension if it was an executable without .py, but here it is .py.
# But inside train_subtitles_transformer.py, there might be imports that fail if environment is slightly different or if paths are relative.
# Let's try normal import.

try:
    from train_subtitles_transformer import TinyTransformerLM, Config
except ImportError:
    print("Could not import TinyTransformerLM from scripts/train_subtitles_transformer.py")
    sys.exit(1)

def load_model(checkpoint_path, device="cuda"):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg = Config()
    # Apply Medium Preset manually to match the training
    cfg.num_layers = 18
    cfg.num_heads = 16
    cfg.embed_dim = 1024
    cfg.ff_hidden_dim = 4096
    cfg.block_size = 1024
    cfg.vocab_size = 32000
    cfg.dropout = 0.0 # Eval mode
    
    # Override with checkpoint config if available
    if "config" in checkpoint:
        print("Found config in checkpoint, using it.")
        # Need to handle if checkpoint config is dict or object
        ckpt_conf = checkpoint["config"]
        # Basic copy
        for k, v in ckpt_conf.__dict__.items() if hasattr(ckpt_conf, "__dict__") else ckpt_conf.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    
    print("Model Config:")
    print(f"  Layers: {cfg.num_layers}")
    print(f"  Embed: {cfg.embed_dim}")
    print(f"  Heads: {cfg.num_heads}")
    
    model = TinyTransformerLM(cfg)
    
    state_dict = checkpoint.get("model_state", checkpoint)
    # Remove _orig_mod. prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def generate(model, tokenizer, prompt="Bonjour je suis", max_new_tokens=50, device="cuda", temperature=0.7):
    encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\nPrompt: '{prompt}'")
    print("Generating...", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= 1024 else input_ids[:, -1024:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] 
            
            # Temperature
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)
            print(".", end="", flush=True)
            
    print("\nDecoding...")
    output_ids = input_ids[0].tolist()
    decoded = tokenizer.decode(output_ids)
    print(f"\n----------------------------------------")
    print(decoded)
    print(f"----------------------------------------")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="trained_models/runs/french_v3_finetune_pure_fr_160k_long/checkpoint_step_200000.pt")
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_path = "data_clean/mistral_tokenizer/tokenizer.json"
    checkpoint_path = args.checkpoint
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
        
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = load_model(checkpoint_path, device)
    
    prompts = [
        "Bonjour, je suis très heureux de",
        "L'intelligence artificielle permet de",
        "La France est un pays situé en",
        "Est-ce que tu peux m'aider avec",
        "Il était une fois"
    ]
    
    for p in prompts:
        generate(model, tokenizer, p, device=device, temperature=args.temp)
