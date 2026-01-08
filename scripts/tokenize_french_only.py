#!/usr/bin/env python3
"""
Tokenize ONLY the filtered French datasets with Mistral-7B tokenizer.
Overwrites the standard tokenized files so the training script picks them up.
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer
import json

def tokenize_french_only():
    print("🔄 TOKENIZATION DU DATASET 100% FRANÇAIS")
    print("=" * 60)
    
    # Charger le tokenizer Mistral sauvegardé
    tokenizer_path = Path("data_clean/mistral_tokenizer")
    if not tokenizer_path.exists():
        print("❌ Tokenizer Mistral non trouvé. Lancement du téléchargement...")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenizer.save_pretrained(tokenizer_path)
    else:
        print(f"✅ Chargement tokenizer depuis {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Traiter train et test
    # INPUT: _fr.txt (le fichier filtré)
    # OUTPUT: _mistral_tokenized.pt (le nom attendu par le script d'entraînement)
    splits = [
        ("conversations_mega_train_fr.txt", "conversations_mega_train_mistral_tokenized.pt", "TRAIN"),
        ("conversations_mega_test_fr.txt", "conversations_mega_test_mistral_tokenized.pt", "TEST")
    ]
    
    for input_name, output_name, split_name in splits:
        input_file = Path(f"data_clean/{input_name}")
        output_file = Path(f"data_clean/{output_name}")
        
        if not input_file.exists():
            print(f"\n⚠️  {split_name}: {input_file} absent - skip")
            continue
        
        print(f"\n📂 {split_name}: Lecture corpus filtré ({input_name})")
        text = input_file.read_text(encoding='utf-8', errors='ignore')
        
        print(f"   Tokenisation en cours...")
        # Tokenize without special tokens first, we might add EOS later if needed by the trainer
        # But usually for causal LM we just tokenize the stream.
        # Mistral tokenizer adds BOS by default? Let's check.
        # We'll just use standard encode.
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        print(f"   Tokens générés: {len(tokens):,}")
        
        print(f"   Sauvegarde dans {output_name}...")
        torch.save(tokens, output_file)
        
    print("\n✅ Terminé! Vous pouvez relancer l'entraînement.")

if __name__ == "__main__":
    tokenize_french_only()
