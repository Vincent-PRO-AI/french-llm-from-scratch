#!/usr/bin/env python3
"""
🔄 Tokenization complète des données françaises pour entraînement ultérieur
Prépare 3 datasets:
1. Phase 2 (80k steps): Wikipedia + FineWeb + Conversations
2. Fine-tuning (60k steps): Conversations FR pures
3. Benchmark: Test set pour validation

Configuration: Mistral tokenizer (117k vocab), séquences 512 tokens
"""

import os
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
from datetime import datetime
import numpy as np

def find_files(directory, extensions):
    """Trouve tous les fichiers avec les extensions données"""
    found = []
    if os.path.exists(directory):
        for ext in extensions:
            found.extend(Path(directory).glob(f"**/*{ext}"))
    return found

def tokenize_file(filepath, tokenizer, max_length=512):
    """Tokenize un fichier et retourne les tokens"""
    print(f"   📖 Tokenizing: {filepath.name}", end=" ... ", flush=True)
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Tokenization
        tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"✅ {len(tokens):,} tokens")
        return torch.tensor(tokens, dtype=torch.long)
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return torch.tensor([], dtype=torch.long)

def create_sequences(tokens, seq_length=512, overlap=0):
    """Crée des séquences de longueur fixe"""
    sequences = []
    stride = seq_length - overlap
    
    for i in range(0, len(tokens) - seq_length + 1, stride):
        seq = tokens[i:i + seq_length]
        if len(seq) == seq_length:
            sequences.append(seq)
    
    return sequences

def main():
    print("\n" + "="*70)
    print("🔄 TOKENIZATION DONNÉES FRANÇAISES - PRÉ-ENTRAÎNEMENT")
    print("="*70 + "\n")
    
    # Configuration
    TOKENIZER_NAME = "gpt2"  # Utiliser GPT-2 tokenizer (plus rapide)
    DATA_DIR = "data_clean"
    OUTPUT_DIR = "data_clean"
    SEQ_LENGTH = 512
    
    print(f"⚙️  Configuration:")
    print(f"   Tokenizer: {TOKENIZER_NAME}")
    print(f"   Data dir: {DATA_DIR}")
    print(f"   Seq length: {SEQ_LENGTH} tokens")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Charger tokenizer (GPT-2 standard, plus rapide)
    print("📥 Chargement tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print(f"   ✅ Vocab size: {len(tokenizer):,}\n")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return
    
    # ============================================
    # DATASET 1: PHASE 2 (80k steps)
    # Wikipedia + FineWeb + Conversations mix
    # ============================================
    print("📊 DATASET 1: PHASE 2 (80k steps) - Mixed FR Data")
    print("-" * 70)
    
    phase2_tokens = torch.tensor([], dtype=torch.long)
    phase2_sources = []
    
    # Wikipedia FR
    print("\n1️⃣  Wikipedia Français:")
    wiki_files = find_files(f"{DATA_DIR}/wikipedia_fr", [".txt"])
    wiki_tokens = torch.tensor([], dtype=torch.long)
    for f in wiki_files[:5]:  # Top 5 fichiers
        tokens = tokenize_file(f, tokenizer)
        wiki_tokens = torch.cat([wiki_tokens, tokens])
        phase2_sources.append({"source": "wikipedia", "file": f.name, "tokens": len(tokens)})
    
    print(f"   📈 Total Wikipedia: {len(wiki_tokens):,} tokens")
    phase2_tokens = torch.cat([phase2_tokens, wiki_tokens])
    
    # FineWeb FR
    print("\n2️⃣  FineWeb Français:")
    fineweb_files = find_files(f"{DATA_DIR}/fineweb_fr", [".txt", ".pt"])
    fineweb_tokens = torch.tensor([], dtype=torch.long)
    for f in fineweb_files[:3]:
        if f.suffix == '.pt':
            tokens = torch.load(f, map_location='cpu')
        else:
            tokens = tokenize_file(f, tokenizer)
        fineweb_tokens = torch.cat([fineweb_tokens, tokens])
        phase2_sources.append({"source": "fineweb", "file": f.name, "tokens": len(tokens)})
    
    print(f"   📈 Total FineWeb: {len(fineweb_tokens):,} tokens")
    phase2_tokens = torch.cat([phase2_tokens, fineweb_tokens])
    
    # Conversations mix
    print("\n3️⃣  Conversations Françaises:")
    conv_files = find_files(f"{DATA_DIR}/conversations", [".txt"])
    conv_tokens = torch.tensor([], dtype=torch.long)
    for f in conv_files[:5]:
        tokens = tokenize_file(f, tokenizer)
        conv_tokens = torch.cat([conv_tokens, tokens])
        phase2_sources.append({"source": "conversations", "file": f.name, "tokens": len(tokens)})
    
    print(f"   📈 Total Conversations: {len(conv_tokens):,} tokens")
    phase2_tokens = torch.cat([phase2_tokens, conv_tokens])
    
    # Créer séquences Phase 2
    print("\n📝 Création séquences (512 tokens)...")
    phase2_sequences = create_sequences(phase2_tokens, SEQ_LENGTH)
    phase2_output = f"{OUTPUT_DIR}/phase2_mixed_fr_mistral_tokenized.pt"
    torch.save(phase2_tokens, phase2_output)
    
    print(f"   ✅ {len(phase2_sequences):,} séquences")
    print(f"   ✅ {len(phase2_tokens):,} tokens total ({len(phase2_tokens)/1e6:.1f}M)")
    print(f"   📁 Sauvegardé: {phase2_output}")
    
    # ============================================
    # DATASET 2: FINE-TUNING (60k steps)
    # Conversations FR uniquement
    # ============================================
    print("\n\n📊 DATASET 2: FINE-TUNING (60k steps) - Pure Conversations FR")
    print("-" * 70)
    
    print("\n🗣️  Conversations Françaises (fine-tuning):")
    
    # Charger toutes les conversations tokenisées
    conv_tokenized_files = find_files(f"{DATA_DIR}", ["conversations_*_tokenized.pt"])
    finetune_tokens = torch.tensor([], dtype=torch.long)
    finetune_sources = []
    
    for f in conv_tokenized_files[:10]:
        print(f"   📦 Loading: {f.name}", end=" ... ", flush=True)
        try:
            tokens = torch.load(f, map_location='cpu')
            finetune_tokens = torch.cat([finetune_tokens, tokens])
            finetune_sources.append({"file": f.name, "tokens": len(tokens)})
            print(f"✅ {len(tokens):,} tokens")
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    print(f"\n   📈 Total Fine-tuning: {len(finetune_tokens):,} tokens ({len(finetune_tokens)/1e6:.1f}M)")
    
    # Créer séquences Fine-tuning
    print("\n📝 Création séquences (512 tokens)...")
    finetune_sequences = create_sequences(finetune_tokens, SEQ_LENGTH)
    finetune_output = f"{OUTPUT_DIR}/finetune_conversations_fr_mistral_tokenized.pt"
    torch.save(finetune_tokens, finetune_output)
    
    print(f"   ✅ {len(finetune_sequences):,} séquences")
    print(f"   ✅ {len(finetune_tokens):,} tokens total ({len(finetune_tokens)/1e6:.1f}M)")
    print(f"   📁 Sauvegardé: {finetune_output}")
    
    # ============================================
    # SPLITS TRAIN/TEST
    # ============================================
    print("\n\n📊 SPLITS TRAIN/VALIDATION")
    print("-" * 70)
    
    # Phase 2 split
    phase2_total = len(phase2_tokens)
    phase2_train_size = int(phase2_total * 0.85)
    phase2_train = phase2_tokens[:phase2_train_size]
    phase2_test = phase2_tokens[phase2_train_size:]
    
    torch.save(phase2_train, f"{OUTPUT_DIR}/phase2_train_mistral_tokenized.pt")
    torch.save(phase2_test, f"{OUTPUT_DIR}/phase2_test_mistral_tokenized.pt")
    
    print(f"\n🔴 Phase 2 (80k steps):")
    print(f"   Train: {len(phase2_train):,} tokens ({len(phase2_train)/1e6:.1f}M)")
    print(f"   Test:  {len(phase2_test):,} tokens ({len(phase2_test)/1e6:.1f}M)")
    print(f"   Ratio: 85% / 15%")
    
    # Fine-tuning split
    finetune_total = len(finetune_tokens)
    finetune_train_size = int(finetune_total * 0.85)
    finetune_train = finetune_tokens[:finetune_train_size]
    finetune_test = finetune_tokens[finetune_train_size:]
    
    torch.save(finetune_train, f"{OUTPUT_DIR}/finetune_train_mistral_tokenized.pt")
    torch.save(finetune_test, f"{OUTPUT_DIR}/finetune_test_mistral_tokenized.pt")
    
    print(f"\n🟢 Fine-tuning Conversations (60k steps):")
    print(f"   Train: {len(finetune_train):,} tokens ({len(finetune_train)/1e6:.1f}M)")
    print(f"   Test:  {len(finetune_test):,} tokens ({len(finetune_test)/1e6:.1f}M)")
    print(f"   Ratio: 85% / 15%")
    
    # ============================================
    # MÉTADONNÉES & VÉRIFICATION
    # ============================================
    print("\n\n📋 MÉTADONNÉES & VÉRIFICATION")
    print("-" * 70)
    
    # Estimer steps nécessaires (batch_size=4, ~500 tokens par step)
    phase2_steps = (len(phase2_train) // 512) // 4
    finetune_steps = (len(finetune_train) // 512) // 4
    
    print(f"\n🎯 Capacité d'entraînement (batch_size=4):")
    print(f"   Phase 2:     {phase2_steps:,} steps ({phase2_steps / 1000:.1f}k) - Cible: 80k ✓" if phase2_steps >= 80000 else f"   Phase 2: {phase2_steps:,} steps - ❌ Besoin: 80k")
    print(f"   Fine-tuning: {finetune_steps:,} steps ({finetune_steps / 1000:.1f}k) - Cible: 60k ✓" if finetune_steps >= 60000 else f"   Fine-tuning: {finetune_steps:,} steps - ❌ Besoin: 60k")
    
    # Créer fichier de métadonnées
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "vocab_size": len(tokenizer)
        },
        "phase2": {
            "total_tokens": len(phase2_tokens),
            "train_tokens": len(phase2_train),
            "test_tokens": len(phase2_test),
            "train_sequences": len(create_sequences(phase2_train, SEQ_LENGTH)),
            "estimated_steps": phase2_steps,
            "target_steps": 80000,
            "sources": phase2_sources
        },
        "finetune": {
            "total_tokens": len(finetune_tokens),
            "train_tokens": len(finetune_train),
            "test_tokens": len(finetune_test),
            "train_sequences": len(create_sequences(finetune_train, SEQ_LENGTH)),
            "estimated_steps": finetune_steps,
            "target_steps": 60000,
            "sources": finetune_sources
        },
        "files": {
            "phase2_total": "phase2_mixed_fr_mistral_tokenized.pt",
            "phase2_train": "phase2_train_mistral_tokenized.pt",
            "phase2_test": "phase2_test_mistral_tokenized.pt",
            "finetune_total": "finetune_conversations_fr_mistral_tokenized.pt",
            "finetune_train": "finetune_train_mistral_tokenized.pt",
            "finetune_test": "finetune_test_mistral_tokenized.pt"
        }
    }
    
    metadata_file = f"{OUTPUT_DIR}/tokenization_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Métadonnées sauvegardées: {metadata_file}")
    
    # Résumé final
    print("\n\n" + "="*70)
    print("✅ TOKENIZATION COMPLÈTE")
    print("="*70)
    print(f"\n📁 Tous les fichiers sauvegardés dans: {OUTPUT_DIR}/")
    print("\n🎯 Prêt pour les prochaines phases:")
    print(f"   1. Phase 2 (80k steps): {len(phase2_train)/1e6:.1f}M tokens")
    print(f"   2. Fine-tuning (60k steps): {len(finetune_train)/1e6:.1f}M tokens")
    print("\n📊 Fichiers générés:")
    for key, value in metadata["files"].items():
        size = os.path.getsize(f"{OUTPUT_DIR}/{value}") / 1e9 if os.path.exists(f"{OUTPUT_DIR}/{value}") else 0
        print(f"   • {value:<45} {size:.2f}GB")

if __name__ == "__main__":
    main()
