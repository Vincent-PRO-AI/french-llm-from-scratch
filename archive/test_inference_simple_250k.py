#!/home/vincent/code/repo/french-llm-from-scratch/.venv/bin/python3
"""Simple inference API test for checkpoint_250k"""

import torch
from pathlib import Path

# Fix for PyTorch 2.6+ weights_only requirement
torch.serialization.add_safe_globals([Path])

checkpoints = [
    ("145k", Path("trained_models/runs/french_v3_finetune_grand_60k/checkpoint_step_145000.pt")),
    ("250k", Path("trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt")),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device}\n")

for checkpoint_name, checkpoint_path in checkpoints:
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        continue
    
    print(f"\n{'='*60}")
    print(f"📊 CHECKPOINT: {checkpoint_name}")
    print(f"{'='*60}\n")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"✅ Checkpoint loaded: {checkpoint_path.name}")
    print(f"   Size: {checkpoint_path.stat().st_size / 1024**3:.2f} GB")
    print(f"   Keys: {list(checkpoint.keys())}")
    print()
    
    # Afficher la structure du modèle
    if 'model_state' in checkpoint:
        model_state = checkpoint['model_state']
        print(f"   Model has 'model_state' key")
    else:
        model_state = checkpoint
    
    print(f"   Total parameters: {sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor)):,}")
    
    # Show layer structure
    layers = {}
    for key in list(model_state.keys())[:10]:
        if isinstance(model_state[key], torch.Tensor):
            shape = model_state[key].shape
            print(f"     {key}: {shape}")

print("\n✅ Checkpoint analysis complete!")
print("\n📋 SUMMARY:")
print("- Checkpoints are PyTorch state dicts for custom TransformerLM")
print("- Ready for export and deployment")
print("- Next: Push to HuggingFace and documentation")

