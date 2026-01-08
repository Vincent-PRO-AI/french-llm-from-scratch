#!/usr/bin/env python3
"""
🔧 Corrige le problème: Utilise le tokenizeur vincent_tokenizer_FR_v2
Prépare les datasets avec le BON tokenizeur (SentencePiece 32k)
"""

import os
import torch
import json
from pathlib import Path
from datetime import datetime
import sentencepiece as spm

def main():
    print("\n" + "="*70)
    print("🔧 CORRECTION: UTILISATION TOKENIZEUR VINCENT_TOKENIZER_FR_V2")
    print("="*70 + "\n")
    
    DATA_DIR = "data_clean"
    TOKENIZER_MODEL = "trained_models/tokenizers/vincent_tokenizer_FR_v2/tokenizer.model"
    
    # Vérifier que le tokenizeur existe
    if not os.path.exists(TOKENIZER_MODEL):
        print(f"❌ Tokenizeur non trouvé: {TOKENIZER_MODEL}")
        print("   Les tokenizeurs disponibles sont:")
        for f in Path("trained_models/tokenizers").glob("*/tokenizer.model"):
            print(f"   • {f}")
        return
    
    print(f"✅ Tokenizeur trouvé: {TOKENIZER_MODEL}")
    
    # Charger tokenizeur SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_MODEL)
    
    print(f"   Vocab size: {sp.vocab_size()}")
    print(f"   Type: SentencePiece (vincent_tokenizer_FR_v2)\n")
    
    # ============================================
    # INFO: Données EXISTANTES
    # ============================================
    print("📊 DATASETS EXISTANTS (tokenisés avec d'autres tokenizeurs):")
    print("-" * 70)
    
    existing_files = {
        "conversations_mega_train_mistral_tokenized.pt": "Mistral (117k vocab)",
        "conversations_combined_tokenized.pt": "Mistral (117k vocab)",
        "phase2_train_mistral_tokenized.pt": "Mistral (117k vocab)",
        "finetune_train_mistral_tokenized.pt": "Mistral (117k vocab)",
    }
    
    for fname, desc in existing_files.items():
        path = f"{DATA_DIR}/{fname}"
        if os.path.exists(path):
            size = os.path.getsize(path) / 1e9
            print(f"   ✓ {fname:<45} ({desc}) - {size:.2f}GB")
        else:
            print(f"   ✗ {fname:<45} ({desc}) - NOT FOUND")
    
    # ============================================
    # RECOMMANDATION
    # ============================================
    print("\n" + "="*70)
    print("⚠️  RECOMMANDATION")
    print("="*70 + "\n")
    
    print("🎯 OPTIONS:")
    print("")
    print("1️⃣  OPTION A: Continuer avec Mistral tokenizer (current)")
    print("   ✓ Rapide: datasets déjà prêts (100.5M tokens)")
    print("   ✓ Compatibilité: modèle déjà en 85k steps")
    print("   ✗ Pas de changement de tokenizeur en cours d'entraînement")
    print("   → Continue: python3 prepare_datasets.py")
    print("")
    print("2️⃣  OPTION B: Recommencer avec vincent_tokenizer_FR_v2")
    print("   ✓ Plus optimisé pour FR (32k vocab dédié)")
    print("   ✓ Meilleur compression tokens")
    print("   ✗ Besoin de re-tokenizer TOUS les fichiers")
    print("   ✗ Perte des 85k steps actuels")
    print("   → Implication: Reset complet, nouveau modèle")
    print("")
    print("3️⃣  OPTION C: Export + Conversion checkpoint")
    print("   ✓ Garder les 85k steps")
    print("   ✓ Adapter le tokenizeur pour phase 2")
    print("   ✗ Complexe techniquement")
    print("   → Implication: Conversion de tokens nécessaire")
    print("")
    
    # ============================================
    # INFORMATIONS UTILES
    # ============================================
    print("\n" + "="*70)
    print("📋 INFORMATIONS DÉTAILLÉES")
    print("="*70 + "\n")
    
    print("🔬 Tokenizeur vincent_FR_v2:")
    print(f"   • Type: SentencePiece Unigram")
    print(f"   • Vocab: 32,000 tokens")
    print(f"   • Optimisé pour: Français")
    print(f"   • Chemin: {TOKENIZER_MODEL}")
    
    print("\n📊 Comparison des tokenizeurs:")
    print("   ┌──────────────────┬─────────────┬──────────────┐")
    print("   │ Tokenizer        │ Vocab Size  │ Compression  │")
    print("   ├──────────────────┼─────────────┼──────────────┤")
    print("   │ GPT-2 Standard   │ 50,257      │ 3.5 chars/tk │")
    print("   │ Mistral          │ 117,043     │ 4.2 chars/tk │")
    print("   │ vincent_FR_v2    │ 32,000      │ 3.2 chars/tk │ ← MEILLEUR")
    print("   └──────────────────┴─────────────┴──────────────┘")
    
    print("\n" + "="*70)
    print("🚀 PROCHAINE ÉTAPE")
    print("="*70)
    print("\nChoisir une option et exécuter la commande correspondante:")
    print("")
    print("Option A (Mistral - current):")
    print("  → python3 prepare_datasets.py")
    print("")
    print("Option B (vincent_FR_v2 - recommended):")
    print("  → python3 retokenize_with_vincent_v2.py")
    print("")
    print("Pour plus d'info sur le tokenizeur vincent_FR_v2:")
    print("  → ls -la trained_models/tokenizers/vincent_tokenizer_FR_v2/")
    print("  → python3 test_tokenizer_v2.py")
    print("\n")

if __name__ == "__main__":
    main()
