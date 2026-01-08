#!/usr/bin/env python3
"""
Export French LLM V3-mistral to GGUF format for LM Studio.
Requires llama.cpp conversion tools.
"""

import torch
import json
from pathlib import Path
import shutil
import subprocess
import sys


def export_checkpoint_to_pytorch_format(checkpoint_path: Path, output_dir: Path):
    """Convert checkpoint to HuggingFace-compatible format."""
    
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config = checkpoint.get('config', {})
    model_state = checkpoint.get('model_state', {})
    
    print(f"✅ Loaded model with {sum(p.numel() for p in model_state.values()):,} parameters")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    pytorch_model_path = output_dir / "pytorch_model.bin"
    torch.save(model_state, pytorch_model_path)
    print(f"💾 Saved weights: {pytorch_model_path}")
    
    # Create config.json for HuggingFace compatibility
    hf_config = {
        "architectures": ["GPTTransformer"],
        "model_type": "gpt",
        "vocab_size": config.get('vocab_size', 117440),  # Mistral tokenizer
        "hidden_size": config.get('embed_dim', 1024),
        "num_hidden_layers": config.get('num_layers', 18),
        "num_attention_heads": config.get('num_heads', 16),
        "intermediate_size": config.get('ff_hidden_dim', 4096),
        "hidden_act": "gelu",
        "hidden_dropout_prob": config.get('dropout', 0.1),
        "attention_probs_dropout_prob": config.get('dropout', 0.1),
        "max_position_embeddings": config.get('block_size', 1024),
        "type_vocab_size": 1,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "use_cache": True,
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(hf_config, f, indent=2)
    print(f"📝 Saved config: {config_path}")
    
    # Save training metadata
    metadata = {
        "model_name": "French-LLM-V3-Mistral",
        "training_steps": 100000,
        "final_train_loss": 5.72,
        "final_val_loss": 5.29,
        "batch_size": 12,
        "learning_rate": 1e-4,
        "architecture": "medium (260M params)",
        "dataset": "conversations_mega (French conversations)",
        "tokenizer": "mistralai/Mistral-7B-v0.1",
        "vocab_size": 117440,
    }
    
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Saved metadata: {metadata_path}")
    
    # Copy tokenizer if available
    tokenizer_src = Path("data_clean/mistral_tokenizer")
    if tokenizer_src.exists():
        tokenizer_dst = output_dir / "tokenizer"
        shutil.copytree(tokenizer_src, tokenizer_dst, dirs_exist_ok=True)
        print(f"🔤 Copied tokenizer: {tokenizer_dst}")
    
    return output_dir


def convert_to_gguf(model_dir: Path, output_gguf: Path, quantization: str = "Q4_K_M"):
    """
    Convert PyTorch model to GGUF using llama.cpp tools.
    Requires llama.cpp to be installed.
    """
    
    print(f"\n🔧 Converting to GGUF format (quantization: {quantization})...")
    
    # Check if llama.cpp is available
    llama_cpp_path = Path.home() / "llama.cpp"
    if not llama_cpp_path.exists():
        print("❌ llama.cpp not found. Please install it:")
        print("   git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        print("   cd ~/llama.cpp && make")
        return None
    
    convert_script = llama_cpp_path / "convert.py"
    quantize_bin = llama_cpp_path / "quantize"
    
    if not convert_script.exists():
        print(f"❌ Conversion script not found: {convert_script}")
        return None
    
    # Step 1: Convert to GGUF FP16
    fp16_output = output_gguf.parent / f"{output_gguf.stem}_fp16.gguf"
    
    print(f"📦 Step 1/2: Converting to FP16 GGUF...")
    try:
        subprocess.run([
            sys.executable,
            str(convert_script),
            str(model_dir),
            "--outfile", str(fp16_output),
            "--outtype", "f16"
        ], check=True)
        print(f"✅ FP16 GGUF created: {fp16_output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        return None
    
    # Step 2: Quantize
    if quantization != "f16" and quantize_bin.exists():
        print(f"📦 Step 2/2: Quantizing to {quantization}...")
        try:
            subprocess.run([
                str(quantize_bin),
                str(fp16_output),
                str(output_gguf),
                quantization
            ], check=True)
            print(f"✅ Quantized GGUF created: {output_gguf}")
            
            # Clean up FP16 file
            fp16_output.unlink()
            print(f"🧹 Cleaned up FP16 file")
            
            return output_gguf
        except subprocess.CalledProcessError as e:
            print(f"❌ Quantization failed: {e}")
            print(f"💡 FP16 model available at: {fp16_output}")
            return fp16_output
    else:
        return fp16_output


def main():
    # Paths
    checkpoint_path = Path("trained_models/runs/french-llm-from-scratch-V3-mistral/checkpoint.pt")
    output_hf_dir = Path("trained_models/french-llm-v3-mistral-hf")
    output_gguf = Path("trained_models/french-llm-v3-mistral-Q4_K_M.gguf")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print("=" * 60)
    print("🚀 French LLM V3-Mistral Export to GGUF")
    print("=" * 60)
    
    # Step 1: Export to HuggingFace format
    print("\n📦 Step 1: Converting checkpoint to HuggingFace format...")
    hf_dir = export_checkpoint_to_pytorch_format(checkpoint_path, output_hf_dir)
    
    print("\n✅ HuggingFace format ready!")
    print(f"📁 Location: {hf_dir}")
    print(f"💡 You can now use this with transformers library or continue to GGUF conversion")
    
    # Step 2: Convert to GGUF (optional, requires llama.cpp)
    print("\n" + "=" * 60)
    response = input("Continue with GGUF conversion? (requires llama.cpp) [y/N]: ")
    
    if response.lower() == 'y':
        gguf_path = convert_to_gguf(hf_dir, output_gguf, quantization="Q4_K_M")
        
        if gguf_path:
            print("\n" + "=" * 60)
            print("🎉 Export completed successfully!")
            print("=" * 60)
            print(f"📁 HuggingFace format: {hf_dir}")
            print(f"📦 GGUF model: {gguf_path}")
            print(f"💾 Size: {gguf_path.stat().st_size / 1e9:.2f} GB")
            print("\n🔥 To use in LM Studio:")
            print(f"   1. Copy {gguf_path} to LM Studio models folder")
            print(f"   2. Or use: File > Import Model > {gguf_path}")
    else:
        print("\n📝 GGUF conversion skipped.")
        print(f"💡 To convert manually later:")
        print(f"   python export_v3_to_gguf.py")


if __name__ == "__main__":
    main()
