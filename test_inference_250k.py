#!/home/vincent/code/repo/french-llm-from-scratch/.venv/bin/python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false
"""Quick inference test and comparison checkpoint_250k vs checkpoint_145k"""

import torch
import json
from pathlib import Path
import sys
import time

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

try:
    from scripts.train_subtitles_transformer import GPTLanguageModel
except ImportError:
    # Try alternative name
    from scripts.train_subtitles_transformer import TinyTransformerLM as GPTLanguageModel

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model with config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'vocab_size': 32000,
            'd_model': 1024,
            'nhead': 16,
            'num_layers': 18,
            'dim_feedforward': 4096,
            'max_seq_length': 1024,
            'dropout': 0.1
        }
    
    model = GPTLanguageModel(**config).to(device)
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_tokens=100, device="cuda"):
    """Generate text from prompt"""
    
    # Simple byte-level fallback to avoid dependency on tokenizer helpers
    try:
        input_ids = tokenizer.encode(prompt)
    except Exception:
        input_ids = [ord(c) % 256 for c in prompt]
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    start_time = time.time()
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
                logits = model(generated)
            
            next_token = logits[0, -1, :].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    elapsed = time.time() - start_time
    generated_text = tokenizer.decode(generated[0].tolist())
    
    return generated_text, elapsed, len(generated[0])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}\n")
    
    # Create tokenizer (fallback byte-level)
    class SimpleTokenizer:
        vocab_size = 32000
        def encode(self, text: str):
            return [ord(c) % 256 for c in text]
        def decode(self, ids):
            return ''.join(chr(int(i) % 256) for i in ids)
        eos_token_id = 0
    tokenizer = SimpleTokenizer()
    
    # Test prompts
    prompts = [
        "L'intelligence artificielle",
        "La France est",
        "Le futur de",
    ]
    
    checkpoints = [
        ("145k", Path("trained_models/runs/french_v3_finetune_grand_60k/checkpoint_step_145000.pt")),
        ("250k", Path("trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt")),
    ]
    
    results = {"benchmarks": []}
    
    for checkpoint_name, checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"📊 CHECKPOINT: {checkpoint_name}")
        print(f"{'='*60}\n")
        
        model = load_model(checkpoint_path, device)
        print(f"✅ Model loaded from {checkpoint_path.name}")
        print(f"   Model size: {checkpoint_path.stat().st_size / 1024**3:.2f} GB\n")
        
        for i, prompt in enumerate(prompts, 1):
            generated, elapsed, tokens = generate_text(model, tokenizer, prompt, max_tokens=80, device=device)
            
            tokens_per_sec = (tokens - len(tokenizer.encode(prompt))) / elapsed
            
            print(f"  Sample {i}:")
            print(f"    Prompt:   {prompt}")
            print(f"    Generated: {generated[:150]}...")
            print(f"    ⏱️ Time: {elapsed:.2f}s | Speed: {tokens_per_sec:.1f} tokens/sec | Tokens: {tokens}")
            print()
            
            results["benchmarks"].append({
                "checkpoint": checkpoint_name,
                "prompt": prompt,
                "generated_text": generated,
                "generation_time_s": elapsed,
                "tokens_generated": tokens,
                "tokens_per_sec": tokens_per_sec
            })
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Save results
    output_path = Path("inference_benchmark_250k.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📈 SUMMARY")
    print(f"{'='*60}\n")
    
    for checkpoint_name in ["145k", "250k"]:
        bench = [b for b in results["benchmarks"] if b["checkpoint"] == checkpoint_name]
        if bench:
            avg_speed = sum(b["tokens_per_sec"] for b in bench) / len(bench)
            print(f"{checkpoint_name}: {avg_speed:.1f} tokens/sec avg")

if __name__ == "__main__":
    main()
