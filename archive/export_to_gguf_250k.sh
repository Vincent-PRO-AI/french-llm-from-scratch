#!/bin/bash
# Export checkpoint_step_250000.pt to GGUF format

set -e

CHECKPOINT="trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt"
EXPORT_DIR="exported_models"

echo "🚀 EXPORTING MODEL TO GGUF"
echo "============================"
echo ""
echo "Checkpoint: $CHECKPOINT"
echo "Export dir: $EXPORT_DIR"
echo ""

# Utilise le venv du projet
PYTHON=".venv/bin/python"

# Créer export_checkpoint.py qui sera exécuté
cat > /tmp/export_checkpoint.py << 'EXPORT_PYTHON'
import torch
import json
from pathlib import Path
import sys
sys.path.insert(0, "/home/vincent/code/repo/french-llm-from-scratch/scripts")

from train_subtitles_transformer import TransformerLM, create_tokenizer

def export_to_hf_format():
    """Convert PyTorch checkpoint to HuggingFace compatible format"""
    
    checkpoint_path = Path("trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt")
    export_dir = Path("exported_models/french_250k_hf")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 Loading checkpoint from {checkpoint_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Load tokenizer
    tokenizer = create_tokenizer()
    tokenizer.save_pretrained(str(export_dir))
    print("✅ Tokenizer saved")
    
    # Load model
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=1024,
        nhead=16,
        num_layers=18,
        dim_feedforward=4096,
        max_seq_length=1024,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    print("✅ Model weights loaded")
    
    # Save to HF format
    torch.save(model.state_dict(), export_dir / "pytorch_model.bin")
    
    # Create config.json for HF compatibility
    config = {
        "architectures": ["TransformerLM"],
        "d_model": 1024,
        "num_layers": 18,
        "num_heads": 16,
        "dim_feedforward": 4096,
        "vocab_size": tokenizer.vocab_size,
        "max_position_embeddings": 1024,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "model_type": "french_transformer",
        "torch_dtype": "float32",
        "transformers_version": "4.40.0"
    }
    
    with open(export_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Model exported to {export_dir}")
    print("")
    print("Next step: Convert to GGUF using:")
    print(f"  python -m ggml.convert --model-dir {export_dir} --outfile {export_dir}/model.gguf")
    
    return str(export_dir)

if __name__ == "__main__":
    export_to_hf_format()
EXPORT_PYTHON

# Run export
echo "🔄 Converting to HF format..."
$PYTHON /tmp/export_checkpoint.py

echo ""
echo "✅ HF export complete!"
echo "📍 Exported to: exported_models/french_250k_hf"
