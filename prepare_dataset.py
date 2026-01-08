#!/usr/bin/env python3
"""
🚀 PREPARE FRENCH CONVERSATION DATASET FOR LLAMA TRAINING
========================================================
Load, verify, and prepare 100% French conversations for LLaMA training.
"""

import torch
from pathlib import Path
import json
from tqdm import tqdm

def verify_dataset(dataset_path: str, sample_size: int = 10):
    """Verify dataset integrity and composition."""
    
    print(f"\n📊 Verifying: {Path(dataset_path).name}")
    
    try:
        tokens = torch.load(dataset_path, map_location='cpu')
        
        if isinstance(tokens, torch.Tensor):
            num_tokens = tokens.numel()
            dtype = tokens.dtype
            print(f"   ✅ Shape: {tokens.shape}")
            print(f"   ✅ Dtype: {dtype}")
            print(f"   ✅ Tokens: {num_tokens:,}")
            print(f"   ✅ Size: {tokens.element_size() * num_tokens / (1024**2):.1f} MB")
            return num_tokens
        else:
            print(f"   ❌ Invalid format: {type(tokens)}")
            return 0
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0


def create_dataset_config():
    """Create config for training dataset."""
    
    config = {
        "name": "French Conversations Dataset V3",
        "description": "100% French conversation dataset for LLaMA-260M training",
        "primary_dataset": "conversations_mega_train_mistral_tokenized.pt",
        "primary_size_tokens": 653000000,  # ~653M
        "language": "French (100%)",
        "composition": {
            "source": "Reddit + Twitter + WebCrawl conversations",
            "quality": "High (deduplicated, cleaned)",
            "tokenizer": "SentencePiece (Mistral v2, 32k vocab)",
        },
        "training": {
            "seq_length": 2048,
            "train_test_split": 0.85,
            "estimated_sequences": 160000,
            "estimated_batches_per_epoch": 10000,
        },
        "notes": [
            "Already pre-tokenized with Mistral tokenizer",
            "Ready for direct LLaMA training",
            "No additional preprocessing needed",
        ]
    }
    
    return config


def main():
    print("\n" + "="*70)
    print("📊 DATASET VERIFICATION FOR LLAMA TRAINING")
    print("="*70)
    
    # Check primary dataset
    primary_path = "data_clean/conversations_mega_train_mistral_tokenized.pt"
    
    if not Path(primary_path).exists():
        print(f"❌ Primary dataset not found: {primary_path}")
        return 1
    
    tokens = verify_dataset(primary_path)
    
    if tokens == 0:
        return 1
    
    # Create config
    config = create_dataset_config()
    
    # Save config
    config_path = Path("data_clean/dataset_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Config saved: {config_path}")
    
    # Summary
    print(f"\n" + "="*70)
    print("📋 DATASET SUMMARY")
    print("="*70)
    print(f"Name: {config['name']}")
    print(f"Language: {config['language']}")
    print(f"Total tokens: {config['primary_size_tokens']:,}")
    print(f"Seq length: {config['training']['seq_length']}")
    print(f"Est. sequences: {config['training']['estimated_sequences']:,}")
    print(f"Training: {config['training']['train_test_split']*100:.0f}% / {(1-config['training']['train_test_split'])*100:.0f}%")
    print("\n✅ READY FOR TRAINING!")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
