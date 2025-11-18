#!/usr/bin/env python3
"""
Télécharge et sauvegarde le tokenizer Mistral-7B-v0.1 pour usage futur
À utiliser quand on entraînera un nouveau modèle from scratch
"""
from transformers import AutoTokenizer
from pathlib import Path

SAVE_DIR = Path("trained_models/tokenizers/mistral-7b-v0.1")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("📥 TÉLÉCHARGEMENT DU TOKENIZER MISTRAL-7B-V0.1")
print("=" * 80)
print(f"Destination: {SAVE_DIR}")
print()

try:
    print("🔄 Téléchargement depuis HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    print("💾 Sauvegarde locale...")
    tokenizer.save_pretrained(SAVE_DIR)
    
    print("\n✅ TOKENIZER MISTRAL SAUVEGARDÉ")
    print("=" * 80)
    print(f"📂 Localisation: {SAVE_DIR}")
    print(f"📊 Vocabulaire: {tokenizer.vocab_size:,} tokens")
    print(f"🔤 Type: {tokenizer.__class__.__name__}")
    print()
    print("💡 Usage futur:")
    print("   Pour entraîner un nouveau modèle from scratch avec ce tokenizer:")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{SAVE_DIR}')")
    print()
    print("⚠️  NOTE: Ne PAS utiliser pour modèle actuel (115k steps)")
    print("   Utiliser uniquement pour nouveau modèle from scratch")
    print("=" * 80)
    
    # Test rapide
    test_text = "Bonjour, comment vas-tu ? Voici un test du tokenizer."
    tokens = tokenizer.encode(test_text)
    print(f"\n🧪 Test: '{test_text}'")
    print(f"   Tokens: {len(tokens)} tokens")
    print(f"   Ratio: {len(test_text.split()) / len(tokens):.2f} mots/token")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    exit(1)
