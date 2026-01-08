#!/usr/bin/env python3
"""
Prepare a clean French dataset for fine-tuning.
1. Reads conversations_mega_train_fr.txt
2. Filters out English lines aggressively
3. Tokenizes using Mistral tokenizer
4. Saves to .pt file
"""
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

def is_french(text):
    # Simple heuristic
    common_fr = {"le", "la", "les", "des", "est", "une", "pour", "dans", "avec", "plus", "pas", "nous", "vous"}
    common_en = {"the", "and", "is", "for", "with", "that", "this", "you", "are", "have"}
    
    words = set(text.lower().split())
    score_fr = len(words.intersection(common_fr))
    score_en = len(words.intersection(common_en))
    
    # Must have significantly more French common words
    if score_en > score_fr:
        return False
    if score_fr == 0:
        return False
    return True

def main():
    source_file = Path("data_clean/conversations_mega_train_fr.txt")
    output_file = Path("data_clean/verified_french_clean.pt")
    tokenizer_path = Path("data_clean/mistral_tokenizer")
    
    print(f"📖 Reading {source_file}...")
    with open(source_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"Total lines: {len(lines)}")
    
    # Filter
    french_lines = []
    skipped = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if is_french(line):
            french_lines.append(line)
        else:
            skipped += 1
            
    print(f"✅ Kept {len(french_lines)} French lines")
    print(f"🗑️  Skipped {skipped} potentially English/Noise lines")
    
    if len(french_lines) == 0:
        print("❌ No valid lines found! Aborting.")
        return

    # Tokenize
    print("\n⚙️  Tokenizing with Mistral...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    all_tokens = []
    # Batch processing for speed
    batch_size = 1000
    for i in tqdm(range(0, len(french_lines), batch_size)):
        batch = french_lines[i : i + batch_size]
        # Adding BOS/EOS logic? Usually dataset needs EOS between samples
        # Mistral uses <s> (1) and </s> (2)
        # We'll just separate lines with EOS (2)
        encoded_batch = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for seq in encoded_batch:
            all_tokens.extend(seq)
            all_tokens.append(tokenizer.eos_token_id)
            
    # Convert to tensor
    print(f"\n📦 Converting {len(all_tokens)} tokens to tensor...")
    tensor_data = torch.tensor(all_tokens, dtype=torch.long)
    
    print(f"💾 Saving to {output_file}")
    torch.save(tensor_data, output_file)
    print("✅ Done!")

if __name__ == "__main__":
    main()
