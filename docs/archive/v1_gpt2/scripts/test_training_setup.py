#!/usr/bin/env python3
"""
TRAINING SCRIPT SIMPLIFIÉ - FRENCH LLM V2 PHASE 2A
Sans complications, juste faire tourner les steps.
"""
import torch
import sys
from pathlib import Path
import json
import time

# Charger le checkpoint
print("📦 Chargement checkpoint...")
checkpoint_path = Path("trained_models/runs/checkpoint_step_100000.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

print(f"✅ Checkpoint chargé")
print(f"   Keys: {list(checkpoint.keys())}")

# Charger le dataset
print("\n📚 Chargement dataset...")
dataset_path = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")
tokens = torch.load(dataset_path, map_location='cpu')
print(f"✅ Dataset chargé: {tokens.shape}")

print(f"\n🚀 TRAINING CONFIG:")
print(f"   Steps: 100000 → 110000")
print(f"   Tokens: {tokens.shape[0]:,}")
print(f"   GPU: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

print(f"\n✅ Tous les fichiers sont prêts!")
print(f"   Pour lancer le training complet, utiliser:")
print(f"   .venv/bin/python3 scripts/train_subtitles_transformer.py \\")
print(f"     --arch-preset medium \\")
print(f"     --resume-from {checkpoint_path} \\")
print(f"     --pretokenized-path {dataset_path} \\")
print(f"     --run-name french_v2_phase2a_fix \\")
print(f"     --max-steps 110000 \\")
print(f"     --batch-size 10 \\")
print(f"     --device cuda")
