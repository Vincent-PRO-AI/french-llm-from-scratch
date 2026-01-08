#!/home/vincent/code/repo/french-llm-from-scratch/.venv/bin/python3
"""Test model responses to questions"""

import torch
from pathlib import Path

# Fix for PyTorch 2.6+ weights_only requirement  
torch.serialization.add_safe_globals([Path])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Testing model on device: {device}\n")

checkpoint_path = Path("trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt")

print(f"Loading checkpoint: {checkpoint_path.name}")
print(f"Size: {checkpoint_path.stat().st_size / 1024**3:.2f} GB\n")

try:
    # Load with timeout
    print("⏳ This may take a moment...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"✅ Checkpoint loaded successfully!")
    print(f"Keys: {list(checkpoint.keys())}\n")
    
    if 'model_state' in checkpoint:
        print(f"✅ Found 'model_state' key")
        model_state = checkpoint['model_state']
    else:
        print(f"✅ Direct state dict")
        model_state = checkpoint
    
    # Count parameters
    total_params = 0
    for key, val in model_state.items():
        if isinstance(val, torch.Tensor):
            total_params += val.numel()
    
    print(f"Total parameters: {total_params:,}")
    
except Exception as e:
    print(f"❌ Error loading: {e}")
    print(f"\nDirect checkpoint info via torch tools:")
    import subprocess
    result = subprocess.run(["python", "-c", f"""
import torch
cp = torch.load('{checkpoint_path}', map_location='cpu')
print(f'Checkpoint keys: {{list(cp.keys())}}')
if 'model_state' in cp:
    print(f'Model state tensors: {{len(cp["model_state"])}}')
else:
    print(f'State dict tensors: {{len(cp)}}')
"""], capture_output=True, text=True, timeout=30)
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")

