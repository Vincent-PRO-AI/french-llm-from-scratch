import os
from pathlib import Path
from langdetect import detect, LangDetectException
from tqdm import tqdm
import concurrent.futures

def is_french(text):
    try:
        # Quick heuristic to avoid running detection on very short text or code
        if len(text) < 50:
            return False
        return detect(text) == 'fr'
    except LangDetectException:
        return False

def filter_file(input_path, output_path):
    print(f"Processing {input_path}...")
    text = Path(input_path).read_text(encoding='utf-8', errors='ignore')
    
    # Split by double newline which seems to be the separator used in combine script
    chunks = text.split('\n\n')
    print(f"Found {len(chunks)} chunks. Filtering for French content...")
    
    french_chunks = []
    english_count = 0
    other_count = 0
    
    # Use parallel processing for speed
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda c: (c, is_french(c)), chunks), total=len(chunks)))
        
    for chunk, is_fr in results:
        if is_fr:
            french_chunks.append(chunk)
        else:
            # Optional: Check what we are discarding
            # if "Assistant" in chunk:
            #     print(f"Discarded: {chunk[:100]}...")
            english_count += 1 # Simplifying assumption for stats
            
    print(f"Kept {len(french_chunks)} French chunks.")
    print(f"Discarded {len(chunks) - len(french_chunks)} chunks.")
    
    output_text = "\n\n".join(french_chunks)
    Path(output_path).write_text(output_text, encoding='utf-8')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    base_dir = Path("data_clean")
    
    # Filter Train
    filter_file(base_dir / "conversations_mega_train.txt", base_dir / "conversations_mega_train_fr.txt")
    
    # Filter Test
    filter_file(base_dir / "conversations_mega_test.txt", base_dir / "conversations_mega_test_fr.txt")
