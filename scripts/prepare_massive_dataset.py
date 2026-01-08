#!/usr/bin/env python3
"""
Prepare a MASSIVE clean French dataset for long training (8-12h).
1. Merges all raw French text sources
2. Aggressively filters out English/Code/Garbage
3. Tokenizes cleanly with Mistral
4. Saves as 'massive_french_clean.pt'
"""
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import re

def is_good_french(text):
    if len(text) < 10: 
        return False
        
    words = text.lower().split()
    if len(words) < 3:
        return False

    # 1. Check for common French stopwords (high density required)
    french_stopwords = {"le", "la", "les", "des", "est", "une", "pour", "dans", "avec", "plus", "pas", "nous", "vous", "c'est", "il", "elle", "ce", "qui", "que"}
    french_count = sum(1 for w in words if w in french_stopwords)
    
    # 2. Check for English stopwords
    english_stopwords = {"the", "and", "is", "for", "with", "that", "this", "you", "are", "have", "from", "be"}
    english_count = sum(1 for w in words if w in english_stopwords)
    
    # Heuristic: Must have more French than English, and at least some French structure
    if english_count > french_count:
        return False
    
    # Must have at least 10% French stopwords density or very high absolute count
    if french_count < 2 and len(words) > 20: 
        return False # Too long without a stopword -> likely code or list

    # 3. Code detection
    if "{" in text and "}" in text and ";" in text:
        return False # Looks like code
        
    return True

def main():
    # Input files
    source_dir = Path("data_clean/raw_sources")
    output_file = Path("data_clean/massive_french_clean.pt")
    tokenizer_path = Path("data_clean/mistral_tokenizer")
    
    all_files = sorted(source_dir.glob("*.txt"))
    print(f"📦 Found {len(all_files)} source files:")
    for f in all_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    # Read and Filter
    clean_lines = []
    total_lines = 0
    
    print("\n📖 Reading and Filtering lines...")
    for f in all_files:
        print(f"  Processing {f.name}...")
        with open(f, "r", encoding="utf-8", errors="replace") as file:
            for line in file:
                total_lines += 1
                line = line.strip()
                if is_good_french(line):
                    clean_lines.append(line)
                    
        print(f"  -> Total clean so far: {len(clean_lines)}")

    print(f"\n📊 Stats:")
    print(f"  Total Raw Lines: {total_lines}")
    print(f"  Kept Clean Lines: {len(clean_lines)} ({len(clean_lines)/total_lines*100:.1f}%)")
    
    # Tokenize
    print("\n⚙️  Tokenizing ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    all_tokens = []
    batch_size = 5000
    
    for i in tqdm(range(0, len(clean_lines), batch_size)):
        batch = clean_lines[i : i + batch_size]
        # Mistral uses <s> (1) and </s> (2). We add EOS (2) after each line.
        encodings = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for seq in encodings:
            all_tokens.extend(seq)
            all_tokens.append(tokenizer.eos_token_id)
            
    print(f"\n📦 Final Token Count: {len(all_tokens):,}")
    
    # Save
    print(f"💾 Saving to {output_file}...")
    torch.save(torch.tensor(all_tokens, dtype=torch.long), output_file)
    print("✅ Done! Dataset ready for long training.")

if __name__ == "__main__":
    main()
