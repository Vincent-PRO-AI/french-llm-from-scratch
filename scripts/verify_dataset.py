#!/usr/bin/env python3
"""
Verify the content of the tokenized dataset by decoding a random sample.
"""
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer

def verify_dataset():
    # Force check the other one
    # data_path = Path("data_clean/finetune_conversations_fr_mistral_tokenized.pt") 
    data_path = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")
    tokenizer_path = Path("data_clean/mistral_tokenizer")
    
    if not data_path.exists():
        # fallback
        data_path = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")
        print(f"⚠️ Primary dataset not found, checking: {data_path}")

    print(f"📂 Loading dataset: {data_path}")
    try:
        data = torch.load(data_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Error loading torch file: {e}")
        return

    # Handle various formats (list, dict, tensor)
    if isinstance(data, dict):
        if 'input_ids' in data:
            data = data['input_ids']
        else:
            data = list(data.values())[0]

    if isinstance(data, list):
        data = torch.tensor(data, dtype=torch.long)

    print(f"📊 Dataset shape: {data.shape}")
    print(f"ℹ️  Total tokens: {data.numel():,}")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print("\n🔍 Decoding sample (tokens 1000-1200):")
    # Take a slice
    sample_ids = data[1000:1200]
    decoded = tokenizer.decode(sample_ids)
    print("-" * 60)
    print(decoded)
    print("-" * 60)
    
    # Check for French words
    french_markers = ["le", "la", "et", "est", "je", "vous", "pour", "c'est"]
    count = sum(1 for w in french_markers if w in decoded.lower().split())
    if count > 2:
        print("✅ Detection: Looks like FRENCH")
    else:
        print("⚠️ Detection: Does NOT look like standard French. Please check output above.")

if __name__ == "__main__":
    verify_dataset()
