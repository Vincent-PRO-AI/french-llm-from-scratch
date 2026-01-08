import os
from pathlib import Path
from langdetect import detect, LangDetectException
import random

def check_corpus_language():
    file_path = Path("/mnt/g/backupllmfromscratch/data_clean/corpus_test_100m.bin")
    file_size = file_path.stat().st_size
    
    print(f"File: {file_path}")
    print(f"Size: {file_size / (1024**3):.2f} GB")
    
    with open(file_path, 'rb') as f:
        # Check 10 random chunks
        for i in range(10):
            offset = random.randint(0, file_size - 10000)
            f.seek(offset)
            # Read enough to skip partial line at start
            chunk = f.read(2000)
            try:
                text = chunk.decode('utf-8', errors='ignore')
                # Skip first partial line
                if '\n' in text:
                    text = text.split('\n', 1)[1]
                
                # Detect
                try:
                    lang = detect(text[:1000])
                    sample = text[:100].replace('\n', ' ')
                    print(f"Chunk {i}: {lang} | Sample: {sample}...")
                except LangDetectException:
                    print(f"Chunk {i}: Detection failed")
            except Exception as e:
                print(f"Chunk {i}: Error {e}")

if __name__ == "__main__":
    check_corpus_language()
