#!/usr/bin/env python3
"""
Re-tokenize conversations_mega train/test datasets with Mistral-7B tokenizer.
This will create separate tokenized datasets compatible with vocab_size=117k
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer
import json

def tokenize_with_mistral():
    print("🔄 RE-TOKENIZATION AVEC MISTRAL-7B TOKENIZER")
    print("=" * 60)
    
    from transformers import AutoTokenizer
    
    # Charger le tokenizer Mistral sauvegardé
    tokenizer_path = Path("data_clean/mistral_tokenizer")
    if not tokenizer_path.exists():
        print("❌ Tokenizer Mistral non trouvé. Lancement du téléchargement...")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenizer.save_pretrained(tokenizer_path)
        print(f"✅ Tokenizer sauvegardé dans {tokenizer_path}")
    else:
        print(f"✅ Chargement tokenizer depuis {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print(f"\n📊 Vocabulaire Mistral: {len(tokenizer)} tokens")
    
    # Traiter train et test séparément
    splits = [
        ("conversations_mega_train.txt", "conversations_mega_train_mistral_tokenized.pt", "TRAIN"),
        ("conversations_mega_test.txt", "conversations_mega_test_mistral_tokenized.pt", "TEST")
    ]
    
    for input_name, output_name, split_name in splits:
        input_file = Path(f"data_clean/{input_name}")
        output_file = Path(f"data_clean/{output_name}")
        
        if not input_file.exists():
            print(f"\n⚠️  {split_name}: {input_file} absent - skip")
            continue
        
        print(f"\n📂 {split_name}: Lecture corpus")
        text = input_file.read_text(encoding='utf-8', errors='ignore')
        
        file_size_mb = len(text.encode('utf-8')) / (1024 * 1024)
        print(f"   Taille: {file_size_mb:.1f} MB")
        print(f"   Caractères: {len(text):,}")
        
        # Tokenisation
        print(f"\n⚙️  {split_name}: Tokenisation en cours...")
        encoded = tokenizer.encode(text, add_special_tokens=False)
        
        print(f"✅ {split_name}: Tokenisation terminée!")
        print(f"   Tokens: {len(encoded):,}")
        print(f"   Compression: {len(text) / len(encoded):.2f} chars/token")
        
        # Sauvegarder en format PyTorch
        tokens_tensor = torch.tensor(encoded, dtype=torch.long)
        torch.save(tokens_tensor, output_file)
        
        output_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n💾 {split_name}: Sauvegarde")
        print(f"   Fichier: {output_file}")
        print(f"   Taille: {output_size_mb:.1f} MB")
        
        # Stats complémentaires
        print(f"\n📈 {split_name}: STATISTIQUES")
        print(f"   Tokens: {len(encoded):,}")
        print(f"   Vocab utilisé: {len(set(encoded)):,} / {len(tokenizer):,}")
        print(f"   Token moyen: {sum(encoded) / len(encoded):.1f}")
        print(f"   Token max: {max(encoded):,}")
        
        # Exemple de décodage
        print(f"\n🔍 {split_name}: Exemple (premiers 100 tokens)")
        sample = tokenizer.decode(encoded[:100])
        print(f"   {sample[:200]}...")
        print()
    
    print("✅ RE-TOKENIZATION TERMINÉE!")
    print("   Fichiers créés:")
    for _, output_name, split_name in splits:
        path = Path(f"data_clean/{output_name}")
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   - {split_name}: {path} ({size_mb:.1f} MB)")
    print("   Prêt pour training avec vocab_size=117k")

if __name__ == "__main__":
    tokenize_with_mistral()
