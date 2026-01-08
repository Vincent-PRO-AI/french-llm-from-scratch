#!/usr/bin/env python3
"""
Prepare Phase 2 dataset (10GB French Corpus) for training.
Tokenizes the raw text binary files into numpy memory-mapped files (uint32).
"""
import os
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import mmap

def tokenize_file_to_bin(input_path, output_path, tokenizer, chunk_size=1024*1024):
    print(f"Processing {input_path} -> {output_path}")
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    # Calculate size for progress bar
    file_size = input_path.stat().st_size
    
    # Temporary list to hold tokens before writing to disk
    # We write in chunks to keep memory usage low
    
    # Initialize output file
    # We don't know exact token count yet, so we append
    if output_path.exists():
        os.remove(output_path)
        
    total_tokens = 0
    
    with open(input_path, 'rb') as f:
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Tokenizing")
        
        while True:
            # Read a chunk of text
            # We read bytes and decode, handling potential split characters at boundaries
            # This is a bit tricky with utf-8. 
            # Simpler approach: read lines or use a robust reader.
            # Given the file format seems to be text, let's try reading as text with errors='ignore'
            # But 'read(size)' on text file might split chars.
            # Let's read bytes, decode, and keep the last partial char for next chunk if needed.
            # Actually, for 10GB, reading line by line is safest if lines are reasonable.
            # If it's one huge line, we have a problem.
            # Let's assume standard text file structure.
            
            lines = f.readlines(chunk_size)
            if not lines:
                break
                
            text_chunk = b"".join(lines).decode('utf-8', errors='ignore')
            pbar.update(len(text_chunk.encode('utf-8')))
            
            # Tokenize
            # add_special_tokens=False because we stream
            tokens = tokenizer.encode(text_chunk, add_special_tokens=False)
            
            if not tokens:
                continue
                
            # Convert to uint32
            tokens_np = np.array(tokens, dtype=np.uint32)
            
            # Append to file
            with open(output_path, "ab") as f_out:
                f_out.write(tokens_np.tobytes())
                
            total_tokens += len(tokens)
            
    print(f"\n✅ Finished. Total tokens: {total_tokens:,}")
    print(f"Saved to {output_path}")

def main():
    base_path = Path("/mnt/g/backupllmfromscratch/data_clean")
    output_dir = Path("data_clean/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer_path = Path("data_clean/mistral_tokenizer")
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 1. Test Set (100MB)
    input_test = base_path / "corpus_test_100m.bin"
    output_test = output_dir / "val_tokens.bin"
    tokenize_file_to_bin(input_test, output_test, tokenizer)
    
    # 2. Train Set (10GB)
    input_train = base_path / "corpus_10g.bin"
    output_train = output_dir / "train_tokens.bin"
    tokenize_file_to_bin(input_train, output_train, tokenizer)

if __name__ == "__main__":
    main()
