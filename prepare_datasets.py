#!/usr/bin/env python3
"""
🔄 Tokenization rapide des données françaises
Crée les datasets pré-tokenisés pour Phase 2 et Fine-tuning
"""

import os
import torch
import json
from pathlib import Path
from datetime import datetime

def load_pretokenized_data():
    """Charge les données pré-tokenisées existantes"""
    print("📦 Chargement des datasets pré-tokenisés existants...\n")
    
    DATA_DIR = "data_clean"
    
    # Lister tous les fichiers tokenisés
    tokenized_files = {}
    
    # Phase 2 data - charger seulement les fichiers importants
    phase2_tokens = torch.tensor([], dtype=torch.long)
    
    # Charger conversations_mega_train (déjà tokenisé avec Mistral)
    conv_mega_path = f"{DATA_DIR}/conversations_mega_train_mistral_tokenized.pt"
    if os.path.exists(conv_mega_path):
        print(f"📥 Loading: conversations_mega_train_mistral_tokenized.pt", end=" ... ", flush=True)
        try:
            conv_tokens = torch.load(conv_mega_path, map_location='cpu', weights_only=True)
            phase2_tokens = torch.cat([phase2_tokens, conv_tokens])
            tokenized_files['conversations_mega_train'] = len(conv_tokens)
            print(f"✅ {len(conv_tokens):,} tokens")
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    # Charger conversations_combined (déjà tokenisé)
    conv_combined_path = f"{DATA_DIR}/conversations_combined_tokenized.pt"
    if os.path.exists(conv_combined_path):
        print(f"📥 Loading: conversations_combined_tokenized.pt", end=" ... ", flush=True)
        try:
            conv_tokens = torch.load(conv_combined_path, map_location='cpu', weights_only=True)
            phase2_tokens = torch.cat([phase2_tokens, conv_tokens])
            tokenized_files['conversations_combined'] = len(conv_tokens)
            print(f"✅ {len(conv_tokens):,} tokens")
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    print(f"\n📈 Total Phase 2 tokens: {len(phase2_tokens):,} ({len(phase2_tokens)/1e6:.1f}M)\n")
    
    # Fine-tuning data - utiliser uniquement les conversations_mega_train
    finetune_tokens = torch.tensor([], dtype=torch.long)
    
    print(f"🗣️  Conversations pour fine-tuning:")
    
    # Charger conversations_mega_train pour fine-tuning
    if os.path.exists(conv_mega_path):
        print(f"   📥 conversations_mega_train_mistral_tokenized.pt", end=" ... ", flush=True)
        try:
            tokens = torch.load(conv_mega_path, map_location='cpu', weights_only=True)
            finetune_tokens = torch.cat([finetune_tokens, tokens])
            print(f"✅ {len(tokens):,}")
        except Exception as e:
            print(f"❌ {e}")
    
    print(f"\n📈 Total Fine-tuning tokens: {len(finetune_tokens):,} ({len(finetune_tokens)/1e6:.1f}M)\n")
    
    return phase2_tokens, finetune_tokens, tokenized_files

def create_sequences(tokens, seq_length=512):
    """Crée des séquences de longueur fixe"""
    sequences = []
    for i in range(0, len(tokens) - seq_length + 1, seq_length):
        seq = tokens[i:i + seq_length]
        if len(seq) == seq_length:
            sequences.append(seq)
    return sequences

def main():
    print("\n" + "="*70)
    print("🔄 PRÉPARATION DATASETS POUR PHASE 2 & FINE-TUNING")
    print("="*70 + "\n")
    
    # Charger données
    phase2_tokens, finetune_tokens, sources = load_pretokenized_data()
    
    DATA_DIR = "data_clean"
    SEQ_LENGTH = 512
    
    # ============================================
    # SPLITS PHASE 2
    # ============================================
    print("📊 SPLITS PHASE 2")
    print("-" * 70)
    
    phase2_total = len(phase2_tokens)
    phase2_train_size = int(phase2_total * 0.85)
    phase2_train = phase2_tokens[:phase2_train_size]
    phase2_test = phase2_tokens[phase2_train_size:]
    
    torch.save(phase2_train, f"{DATA_DIR}/phase2_train_mistral_tokenized.pt")
    torch.save(phase2_test, f"{DATA_DIR}/phase2_test_mistral_tokenized.pt")
    torch.save(phase2_tokens, f"{DATA_DIR}/phase2_all_mistral_tokenized.pt")
    
    print(f"Total tokens: {len(phase2_tokens):,}")
    print(f"Train:  {len(phase2_train):,} ({len(phase2_train)/1e6:.1f}M) - 85%")
    print(f"Test:   {len(phase2_test):,} ({len(phase2_test)/1e6:.1f}M) - 15%")
    
    # Créer séquences Phase 2
    phase2_train_seqs = create_sequences(phase2_train, SEQ_LENGTH)
    phase2_test_seqs = create_sequences(phase2_test, SEQ_LENGTH)
    
    print(f"Train sequences: {len(phase2_train_seqs):,}")
    print(f"Test sequences:  {len(phase2_test_seqs):,}")
    
    # Estimer steps
    phase2_steps = (len(phase2_train) // SEQ_LENGTH) // 4  # batch size 4
    print(f"Estimated steps: {phase2_steps:,} (Target: 80000) {'✅' if phase2_steps >= 80000 else '❌'}")
    
    # ============================================
    # SPLITS FINE-TUNING
    # ============================================
    print("\n📊 SPLITS FINE-TUNING")
    print("-" * 70)
    
    finetune_total = len(finetune_tokens)
    finetune_train_size = int(finetune_total * 0.85)
    finetune_train = finetune_tokens[:finetune_train_size]
    finetune_test = finetune_tokens[finetune_train_size:]
    
    torch.save(finetune_train, f"{DATA_DIR}/finetune_train_mistral_tokenized.pt")
    torch.save(finetune_test, f"{DATA_DIR}/finetune_test_mistral_tokenized.pt")
    torch.save(finetune_tokens, f"{DATA_DIR}/finetune_all_mistral_tokenized.pt")
    
    print(f"Total tokens: {len(finetune_tokens):,}")
    print(f"Train:  {len(finetune_train):,} ({len(finetune_train)/1e6:.1f}M) - 85%")
    print(f"Test:   {len(finetune_test):,} ({len(finetune_test)/1e6:.1f}M) - 15%")
    
    # Créer séquences Fine-tuning
    finetune_train_seqs = create_sequences(finetune_train, SEQ_LENGTH)
    finetune_test_seqs = create_sequences(finetune_test, SEQ_LENGTH)
    
    print(f"Train sequences: {len(finetune_train_seqs):,}")
    print(f"Test sequences:  {len(finetune_test_seqs):,}")
    
    # Estimer steps
    finetune_steps = (len(finetune_train) // SEQ_LENGTH) // 4  # batch size 4
    print(f"Estimated steps: {finetune_steps:,} (Target: 60000) {'✅' if finetune_steps >= 60000 else '❌'}")
    
    # ============================================
    # MÉTADONNÉES
    # ============================================
    print("\n📋 MÉTADONNÉES")
    print("-" * 70)
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "source_files": sources,
        "phase2": {
            "total_tokens": len(phase2_tokens),
            "train_tokens": len(phase2_train),
            "test_tokens": len(phase2_test),
            "train_sequences": len(phase2_train_seqs),
            "test_sequences": len(phase2_test_seqs),
            "estimated_steps": phase2_steps,
            "target_steps": 80000,
            "status": "✅ READY" if phase2_steps >= 80000 else "❌ INSUFFICIENT"
        },
        "finetune": {
            "total_tokens": len(finetune_tokens),
            "train_tokens": len(finetune_train),
            "test_tokens": len(finetune_test),
            "train_sequences": len(finetune_train_seqs),
            "test_sequences": len(finetune_test_seqs),
            "estimated_steps": finetune_steps,
            "target_steps": 60000,
            "status": "✅ READY" if finetune_steps >= 60000 else "❌ INSUFFICIENT"
        }
    }
    
    metadata_file = f"{DATA_DIR}/tokenization_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Métadonnées sauvegardées: {metadata_file}")
    
    # ============================================
    # RÉSUMÉ FINAL
    # ============================================
    print("\n" + "="*70)
    print("✅ PRÉPARATION COMPLÈTE")
    print("="*70)
    print("\n📁 Fichiers générés:")
    print(f"   • phase2_train_mistral_tokenized.pt ({len(phase2_train)/1e9:.2f}GB)")
    print(f"   • phase2_test_mistral_tokenized.pt ({len(phase2_test)/1e9:.2f}GB)")
    print(f"   • finetune_train_mistral_tokenized.pt ({len(finetune_train)/1e9:.2f}GB)")
    print(f"   • finetune_test_mistral_tokenized.pt ({len(finetune_test)/1e9:.2f}GB)")
    print(f"   • tokenization_metadata.json")
    
    print("\n🎯 Prêt pour entraînement:")
    print(f"   ✅ Phase 2:     {phase2_steps:,} steps ({phase2_steps/1000:.1f}k)")
    print(f"   ✅ Fine-tuning: {finetune_steps:,} steps ({finetune_steps/1000:.1f}k)")
    print("\n🚀 Prochaines étapes:")
    print("   1. python3 train_phase2_80k.py")
    print("   2. python3 train_finetune_conversations_60k.py")
    print("\n")

if __name__ == "__main__":
    main()
