#!/usr/bin/env python3
"""
📊 Comparaison complète de tous les tokenizers disponibles
"""

import json
import os
from pathlib import Path
import sentencepiece as spm

def analyze_tokenizers():
    print("\n" + "="*80)
    print("📊 ANALYSE COMPLÈTE DES TOKENIZERS DISPONIBLES")
    print("="*80 + "\n")
    
    tokenizers = {}
    
    # ============================================
    # 1. VINCENT_TOKENIZER_FR_V2
    # ============================================
    print("1️⃣  VINCENT_TOKENIZER_FR_V2")
    print("-" * 80)
    
    vincent_path = "trained_models/tokenizers/vincent_tokenizer_FR_v2/tokenizer.model"
    if os.path.exists(vincent_path):
        sp = spm.SentencePieceProcessor()
        sp.Load(vincent_path)
        
        info = {
            "name": "vincent_tokenizer_FR_v2",
            "type": "SentencePiece",
            "model_type": "Unigram",
            "vocab_size": sp.vocab_size(),
            "file_size": os.path.getsize(vincent_path),
            "path": vincent_path,
            "description": "Tokenizer optimisé pour le français (Unigram, 32k vocab)"
        }
        
        print(f"   Type:           {info['type']} ({info['model_type']})")
        print(f"   Vocab Size:     {info['vocab_size']:,} tokens")
        print(f"   File Size:      {info['file_size']/1e6:.2f}MB")
        print(f"   Optimisé pour:  Français 🇫🇷")
        print(f"   Path:           {vincent_path}")
        
        # Test rapide
        test_text = "Bonjour, comment allez-vous?"
        tokens = sp.encode(test_text)
        print(f"   Test:           '{test_text}'")
        print(f"                   {len(tokens)} tokens: {sp.encode(test_text, out_type=str)}")
        
        tokenizers['vincent_FR_v2'] = info
        print("   ✅ DISPONIBLE\n")
    else:
        print("   ❌ NOT FOUND\n")
    
    # ============================================
    # 2. MISTRAL TOKENIZER
    # ============================================
    print("2️⃣  MISTRAL TOKENIZER")
    print("-" * 80)
    
    mistral_config = "data_clean/mistral_tokenizer/tokenizer_config.json"
    mistral_json = "data_clean/mistral_tokenizer/tokenizer.json"
    
    if os.path.exists(mistral_config):
        with open(mistral_config) as f:
            config = json.load(f)
        
        with open(mistral_json) as f:
            tokenizer_data = json.load(f)
        
        vocab_size = len(tokenizer_data.get('model', {}).get('vocab', {}))
        
        info = {
            "name": "Mistral Tokenizer",
            "type": "BPE (Byte Pair Encoding)",
            "model_type": "v2",
            "vocab_size": 117043,
            "file_size": os.path.getsize(mistral_json),
            "path": mistral_json,
            "description": "Tokenizer officiel Mistral (BPE, 117k vocab)"
        }
        
        print(f"   Type:           BPE (Byte Pair Encoding)")
        print(f"   Model:          Mistral v2")
        print(f"   Vocab Size:     {info['vocab_size']:,} tokens")
        print(f"   File Size:      {info['file_size']/1e6:.2f}MB")
        print(f"   Tokenizer Class: {config.get('tokenizer_class', 'Unknown')}")
        print(f"   Path:           {mistral_json}")
        print(f"   Format:         JSON (.json)")
        
        tokenizers['mistral'] = info
        print("   ✅ DISPONIBLE\n")
    else:
        print("   ❌ NOT FOUND\n")
    
    # ============================================
    # 3. GPT-2 TOKENIZER (Standard)
    # ============================================
    print("3️⃣  GPT-2 TOKENIZER (Standard)")
    print("-" * 80)
    
    gpt2_checkpoint = "trained_models/runs/french_llm_hf/checkpoint-80000/tokenizer_config.json"
    
    if os.path.exists(gpt2_checkpoint):
        with open(gpt2_checkpoint) as f:
            config = json.load(f)
        
        info = {
            "name": "GPT-2 Tokenizer",
            "type": "BPE (Byte Pair Encoding)",
            "model_type": "GPT-2 Standard",
            "vocab_size": 50257,
            "path": gpt2_checkpoint,
            "description": "Tokenizer standard GPT-2 (50257 vocab)"
        }
        
        print(f"   Type:           BPE (Byte Pair Encoding)")
        print(f"   Model:          GPT-2 Standard")
        print(f"   Vocab Size:     {info['vocab_size']:,} tokens")
        print(f"   Tokenizer Class: {config.get('tokenizer_class', 'Unknown')}")
        print(f"   Path:           {gpt2_checkpoint}")
        print(f"   Note:           Utilisé dans Phase 1 training")
        print(f"   Status:         ⚠️  INCOMPATIBLE avec données Mistral pre-tokenized")
        
        tokenizers['gpt2'] = info
        print("   ✅ DISPONIBLE\n")
    else:
        print("   ❌ NOT FOUND\n")
    
    # ============================================
    # RÉSUMÉ COMPARATIF
    # ============================================
    print("\n" + "="*80)
    print("📊 TABLEAU COMPARATIF")
    print("="*80 + "\n")
    
    print("┌─────────────────────┬──────────┬──────────┬──────────┬────────────────────┐")
    print("│ Tokenizer           │ Type     │ Vocab    │ Optimisé │ Status             │")
    print("├─────────────────────┼──────────┼──────────┼──────────┼────────────────────┤")
    print("│ Vincent_FR_v2       │ Unigram  │ 32,000   │ Français │ ✅ Ready           │")
    print("│ Mistral             │ BPE      │ 117,043  │ Multilang│ ✅ Ready           │")
    print("│ GPT-2 Standard      │ BPE      │ 50,257   │ Anglais  │ ⚠️  Incompatible   │")
    print("└─────────────────────┴──────────┴──────────┴──────────┴────────────────────┘")
    
    # ============================================
    # DONNÉES PRE-TOKENISÉES
    # ============================================
    print("\n" + "="*80)
    print("💾 DONNÉES PRE-TOKENISÉES DISPONIBLES")
    print("="*80 + "\n")
    
    print("🟢 MISTRAL TOKENIZED (117k vocab):")
    print("-" * 80)
    
    mistral_data = [
        ("conversations_mega_train_mistral_tokenized.pt", "85.5M tokens"),
        ("conversations_combined_tokenized.pt", "15M tokens"),
        ("fineweb_fr_tokenized.pt", "variable"),
        ("fineweb2_fr_tokenized.pt", "variable"),
        ("dolly_fr_tokenized.pt", "variable"),
    ]
    
    for fname, size in mistral_data:
        path = f"data_clean/{fname}"
        if os.path.exists(path):
            actual_size = os.path.getsize(path) / 1e9
            print(f"   ✅ {fname:<45} {actual_size:.2f}GB ({size})")
        else:
            print(f"   ⏭️  {fname:<45} (~{size})")
    
    print("\n🟡 VINCENT_FR_V2 TOKENIZED:")
    print("-" * 80)
    print("   ❌ Aucune donnée pré-tokenisée avec ce tokenizer")
    print("   ⚠️  Nécessite une re-tokenization complète (~4h)")
    
    print("\n🔵 GPT-2 TOKENIZED:")
    print("-" * 80)
    print("   ⚠️  Utilisé pendant Phase 1 training")
    print("   ❌ Pas de données additionnelles disponibles")
    print("   ⚠️  INCOMPATIBLE avec données Mistral")
    
    # ============================================
    # RECOMMANDATIONS
    # ============================================
    print("\n" + "="*80)
    print("🎯 RECOMMANDATIONS PAR CAS D'USAGE")
    print("="*80 + "\n")
    
    print("1️⃣  Pour LM Studio (Quick & Working):")
    print("   → Utiliser: MISTRAL (117k)")
    print("   → Raison: Données pre-tokenized, compatible GGUF")
    print("   → Temps: Immédiat (données prêtes)")
    print()
    
    print("2️⃣  Pour Qualité Optimale FR:")
    print("   → Utiliser: VINCENT_FR_V2 (32k)")
    print("   → Raison: Optimisé français, meilleure compression")
    print("   → Coût: ~4h tokenization + 12h training")
    print()
    
    print("3️⃣  ❌ Ne PAS utiliser:")
    print("   → GPT-2 Standard: Tokenizer mismatch fondamental")
    print()
    
    # ============================================
    # ACTIONS RECOMMANDÉES
    # ============================================
    print("\n" + "="*80)
    print("🚀 ACTIONS RECOMMANDÉES")
    print("="*80 + "\n")
    
    print("Option A (FASTEST - 2h):")
    print("  1. Créer modèle Mistral Small (125M)")
    print("  2. Utiliser données Mistral (100.5M tokens)")
    print("  3. Entraîner 80k steps")
    print("  4. Export GGUF → LM Studio ✅")
    print()
    
    print("Option B (OPTIMAL - 16h):")
    print("  1. Re-tokeniser toutes données avec vincent_FR_v2 (~4h)")
    print("  2. Créer modèle GPT-2 ou Mistral")
    print("  3. Entraîner 80k steps (~12h)")
    print("  4. Export GGUF → LM Studio ✅")
    print()
    
    print("Option C (CURRENT - 0h):")
    print("  ❌ Continuer avec Phase 1 GPT-2")
    print("  ❌ Résultat inutilisable LM Studio")
    print()

if __name__ == "__main__":
    analyze_tokenizers()
