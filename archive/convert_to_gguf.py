#!/usr/bin/env python3
"""
🔄 CONVERT TO GGUF - Using HuggingFace transformers
Convertit le modèle PyTorch en GGUF pour LM Studio
"""

import torch
import json
import struct
import os
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def convert_to_gguf_manual(model_dir, output_path):
    """Convert PyTorch model to GGUF format manually"""
    
    print(f"\n{'='*70}")
    print("🔄 CONVERTING TO GGUF")
    print(f"{'='*70}\n")
    
    print(f"📦 Loading model from: {model_dir}")
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    config = model.config
    
    # Get state dict
    state_dict = model.state_dict()
    
    print(f"✅ Model loaded")
    print(f"   Architecture: {config.model_type}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num layers: {config.num_hidden_layers}")
    print(f"   Num heads: {config.num_attention_heads}\n")
    
    # Create output dir
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # GGUF format header
    print(f"🔧 Writing GGUF format...")
    
    with open(output_path, 'wb') as f:
        # Magic number
        f.write(b'GGUF')
        
        # Version (3)
        f.write(struct.pack('<I', 3))
        
        # Tensor count
        n_tensors = len(state_dict)
        f.write(struct.pack('<Q', n_tensors))
        
        # Key-value pairs count
        ff_size = config.n_inner if hasattr(config, 'n_inner') and config.n_inner else config.hidden_size * 4
        
        kv_pairs = {
            "general.architecture": "gpt2",
            "general.name": "French GPT-2",
            "general.quantization_version": 2,
            "gpt2.context_length": config.max_position_embeddings,
            "gpt2.embedding_length": config.hidden_size,
            "gpt2.feed_forward_length": ff_size,
            "gpt2.attention.head_count": config.num_attention_heads,
            "gpt2.block_count": config.num_hidden_layers,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": list(range(config.vocab_size)),
        }
        
        f.write(struct.pack('<Q', len(kv_pairs)))
        
        # Write key-value pairs
        for key, value in kv_pairs.items():
            # Key string
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(key_bytes)))
            f.write(key_bytes)
            
            # Value type and content
            if isinstance(value, str):
                f.write(struct.pack('<I', 8))  # String type
                val_bytes = value.encode('utf-8')
                f.write(struct.pack('<Q', len(val_bytes)))
                f.write(val_bytes)
            elif isinstance(value, int):
                f.write(struct.pack('<I', 5))  # Int type
                f.write(struct.pack('<q', value))
            elif isinstance(value, list):
                f.write(struct.pack('<I', 5))  # Int array type
                f.write(struct.pack('<Q', len(value)))
                for v in value:
                    f.write(struct.pack('<I', v))
        
        # Write tensors
        print(f"   Tensors: {n_tensors}")
        for i, (name, tensor) in enumerate(state_dict.items()):
            if i % 10 == 0:
                print(f"   - {i}/{n_tensors}")
            
            # Tensor name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            
            # Shape
            shape = tensor.shape
            f.write(struct.pack('<I', len(shape)))
            for s in shape:
                f.write(struct.pack('<I', s))
            
            # Type (F32 = 0)
            f.write(struct.pack('<I', 0))
            
            # Offset (placeholder)
            f.write(struct.pack('<Q', 0))
            
            # Data
            tensor_bytes = tensor.detach().cpu().float().numpy().tobytes()
            f.write(tensor_bytes)
    
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"\n✅ GGUF created: {output_path}")
    print(f"   Size: {size_gb:.2f} GB\n")
    
    return True

def main():
    model_dir = "trained_models/outputs/french_gpt2_pytorch"
    output_file = "trained_models/outputs/french_gpt2_lm_studio.gguf"
    
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    try:
        convert_to_gguf_manual(model_dir, output_file)
        
        print(f"{'='*70}")
        print("✅ CONVERSION COMPLETE!")
        print(f"{'='*70}\n")
        print("📍 GGUF file ready for LM Studio:")
        print(f"   {output_file}\n")
        print("🎯 Next steps:")
        print("   1. Download LM Studio: https://lmstudio.ai/")
        print("   2. File → Load Model")
        print(f"   3. Select: {output_file}")
        print("   4. Chat in French!\n")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
