#!/usr/bin/env python3
"""
Combine all text datasets into train/test splits for Mistral tokenization.
Keeps 15% of data for testing to avoid data leakage.
"""
from pathlib import Path
import random

def combine_all_text():
    print("📚 COMBINAISON DE TOUS LES CORPUS TEXTE (TRAIN/TEST)")
    print("=" * 60)
    
    output_train = Path("data_clean/conversations_mega_train.txt")
    output_test = Path("data_clean/conversations_mega_test.txt")
    sources = []
    
    # Sources principales
    text_files = [
        ("data_clean/conversations_all/conversations_all_fr.txt", "Conversations base"),
        ("data_clean/conversations_extra/conversations_extra_fr.txt", "Conversations extra"),
        ("data_clean/conversations_massive/ultrachat_extended_fr.txt", "UltraChat extended"),
        ("data_clean/conversations_massive/oasst2_fr.txt", "OASST2"),
        ("data_clean/conversations_massive/dolly_fr.txt", "Dolly"),
    ]
    
    total_chars = 0
    all_texts = []
    
    for filepath, name in text_files:
        path = Path(filepath)
        if path.exists():
            print(f"\n✅ {name}")
            text = path.read_text(encoding='utf-8', errors='ignore')
            size_mb = len(text.encode('utf-8')) / (1024 * 1024)
            print(f"   Fichier: {path}")
            print(f"   Taille: {size_mb:.1f} MB")
            print(f"   Caractères: {len(text):,}")
            
            # Découper en chunks de ~10k caractères pour mélanger
            chunk_size = 10000
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            all_texts.extend(chunks)
            total_chars += len(text)
            sources.append(name)
        else:
            print(f"\n⚠️  {name} - ABSENT")
            print(f"   Chemin: {path}")
    
    if not all_texts:
        print("\n❌ Aucun fichier texte trouvé!")
        return
    
    print(f"\n📊 TOTAL COLLECTÉ:")
    print(f"   Chunks: {len(all_texts):,}")
    print(f"   Caractères: {total_chars:,}")
    print(f"   Sources: {', '.join(sources)}")
    
    # Mélanger les chunks
    print(f"\n🔀 Mélange des {len(all_texts)} chunks...")
    random.seed(42)  # Pour reproductibilité
    random.shuffle(all_texts)
    
    # Séparer train/test (85%/15%)
    test_ratio = 0.15
    test_size = int(len(all_texts) * test_ratio)
    
    test_chunks = all_texts[:test_size]
    train_chunks = all_texts[test_size:]
    
    print(f"\n📋 SPLIT TRAIN/TEST:")
    print(f"   Train: {len(train_chunks):,} chunks ({100-test_ratio*100:.0f}%)")
    print(f"   Test: {len(test_chunks):,} chunks ({test_ratio*100:.0f}%)")
    
    # Combiner
    print(f"\n⚙️  Combinaison des chunks...")
    train_text = "\n\n".join(train_chunks)
    test_text = "\n\n".join(test_chunks)
    
    # Sauvegarder
    output_train.write_text(train_text, encoding='utf-8')
    output_test.write_text(test_text, encoding='utf-8')
    
    train_size_mb = output_train.stat().st_size / (1024 * 1024)
    test_size_mb = output_test.stat().st_size / (1024 * 1024)
    
    print(f"\n💾 CORPUS SAUVEGARDÉS")
    print(f"   Train: {output_train} ({train_size_mb:.1f} MB)")
    print(f"   Test: {output_test} ({test_size_mb:.1f} MB)")
    print(f"   Ratio: {len(train_text)/(len(train_text)+len(test_text))*100:.1f}% / {len(test_text)/(len(train_text)+len(test_text))*100:.1f}%")
    print(f"\n✅ Prêt pour tokenisation Mistral (avec split train/test)!")

if __name__ == "__main__":
    combine_all_text()
