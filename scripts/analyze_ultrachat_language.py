import os
from pathlib import Path
from langdetect import detect, LangDetectException
from tqdm import tqdm
import concurrent.futures

def detect_language(text):
    try:
        if len(text) < 50:
            return None
        return detect(text)
    except LangDetectException:
        return None

def analyze_file(filepath):
    print(f"Analyzing {filepath}...")
    try:
        text = Path(filepath).read_text(encoding='utf-8', errors='ignore')
        chunks = text.split('\n\n')
        
        langs = []
        # Analyze a sample of chunks to be faster
        sample_size = min(1000, len(chunks))
        sample_chunks = chunks[:sample_size]
        
        for chunk in sample_chunks:
            lang = detect_language(chunk)
            if lang:
                langs.append(lang)
                
        fr_count = langs.count('fr')
        en_count = langs.count('en')
        total = len(langs)
        
        if total == 0:
            return 0, 0, 0
            
        return fr_count, en_count, total
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, 0, 0

def main():
    base_path = Path("/mnt/g/backupllmfromscratch/data_clean/conversations_massive/")
    files = sorted(list(base_path.glob("ultrachat_extended_part_*.txt")))
    
    # Analyze first 10 files
    files_to_check = files[:10]
    
    print(f"Checking {len(files_to_check)} files from {base_path}...")
    
    total_fr = 0
    total_en = 0
    total_samples = 0
    
    for f in files_to_check:
        fr, en, tot = analyze_file(f)
        print(f"  {f.name}: FR={fr} ({fr/tot*100:.1f}%), EN={en} ({en/tot*100:.1f}%)")
        total_fr += fr
        total_en += en
        total_samples += tot
        
    print("\n--- Summary ---")
    if total_samples > 0:
        print(f"Total Samples: {total_samples}")
        print(f"French: {total_fr} ({total_fr/total_samples*100:.1f}%)")
        print(f"English: {total_en} ({total_en/total_samples*100:.1f}%)")
    else:
        print("No samples found.")

if __name__ == "__main__":
    main()
