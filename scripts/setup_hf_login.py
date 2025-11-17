#!/usr/bin/env python3
"""Script pour se connecter à HuggingFace et télécharger LMSYS"""
from huggingface_hub import login
from datasets import load_dataset
import sys
import os

print("🔑 Connexion à HuggingFace...")
print("📝 Entre ton token (depuis https://huggingface.co/settings/tokens)")
print()

# Récupérer le token
token = os.environ.get('HF_TOKEN')
if token:
    print("✅ Token trouvé dans HF_TOKEN")
else:
    token = input("Token: ").strip()

try:
    # Login avec le token
    login(token=token)
    print("\n✅ Connexion réussie!")
    
    # Tester l'accès à LMSYS
    print("\n🔄 Test d'accès à lmsys/lmsys-chat-1m...")
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    
    # Prendre un exemple
    first_item = next(iter(ds))
    print("✅ Accès OK!")
    print(f"   Premier exemple: {list(first_item.keys())}")
    
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    sys.exit(1)

print("\n🎉 Configuration terminée!")
print("💡 Tu peux maintenant relancer le téléchargement:")
print("   .venv/bin/python scripts/download_best_conversations.py --max-total 80000")
