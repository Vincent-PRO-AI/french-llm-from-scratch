import torch
from transformers import AutoTokenizer
import os
from pathlib import Path
import random

def inspect_dataset():
    base_path = Path("data_clean")
    tokenizer_path = base_path / "mistral_tokenizer"
    data_file = base_path / "conversations_mega_train_mistral_tokenized.pt"

    print(f"Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading data from {data_file}...")
    if not data_file.exists():
        print(f"File not found: {data_file}")
        return

    try:
        data = torch.load(data_file, map_location="cpu")
        if isinstance(data, dict):
            if 'input_ids' in data:
                data = data['input_ids']
            else:
                data = list(data.values())[0]
        
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
            
        print(f"Data loaded. Shape: {data.shape}, Dtype: {data.dtype}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Inspect beginning
    print("\n--- Beginning of Dataset (First 500 tokens) ---")
    start_tokens = data[:500].tolist()
    decoded_start = tokenizer.decode(start_tokens)
    print(decoded_start)
    
    # Inspect random middle chunk
    if len(data) > 10000:
        mid_idx = random.randint(5000, len(data) - 1000)
        print(f"\n--- Random Chunk (Offset {mid_idx}, 500 tokens) ---")
        mid_tokens = data[mid_idx:mid_idx+500].tolist()
        decoded_mid = tokenizer.decode(mid_tokens)
        print(decoded_mid)
    
    # Check for Arabic characters specifically
    # Arabic unicode range: 0600–06FF
    print("\n--- Character Analysis ---")
    arabic_count = 0
    total_chars = 0
    
    # Sample 10,000 tokens for analysis
    sample_size = min(10000, len(data))
    sample_tokens = data[:sample_size].tolist()
    sample_text = tokenizer.decode(sample_tokens)
    
    for char in sample_text:
        if '\u0600' <= char <= '\u06FF':
            arabic_count += 1
        total_chars += 1
            
    print(f"Analyzed {total_chars} characters.")
    print(f"Arabic characters found: {arabic_count}")
    if total_chars > 0:
        print(f"Percentage: {arabic_count / total_chars * 100:.2f}%")

if __name__ == "__main__":
    inspect_dataset()
