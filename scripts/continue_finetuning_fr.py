#!/usr/bin/env python3
"""
Launch training continuing from the best V3 checkpoint (160k).
Specialized for fine-tuning on French conversations to fix "English drift".
"""
import subprocess
import sys
from pathlib import Path

def launch_training():
    print("🚀 LANCEMENT FINETUNING FRANÇAIS (V3 160k -> 200k)")
    print("=" * 60)

    # 1. Configuration
    script_path = Path("scripts/train_subtitles_transformer.py")
    
    # Checkpoint ID 160k (Best performing model)
    checkpoint_path = Path("trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt")
    
    # 100% French conversational data (Mistral tokenized)
    data_path = Path("data_clean/massive_french_clean.pt")
    
    if data_path.exists():
        print(f"✅ Using verified CLEAN French dataset: {data_path}")
    else:
        print(f"❌ Clean dataset not found. Please run scripts/prepare_massive_dataset.py")
        return
    
    tokenizer_path = Path("data_clean/mistral_tokenizer")
    run_name = "french_v3_finetune_pure_fr_160k_long"

    # Verify existing files
    if not script_path.exists():
        print(f"❌ Script introuvable: {script_path}")
        return
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint introuvable: {checkpoint_path}")
        return
    if not data_path.exists():
        print(f"❌ Dataset introuvable: {data_path}")
        return

    # 2. Build Command
    # Using simple SubtitleTrainer (metrics via jsonl) 
    # train_subtitles_transformer.py logs to visual.jsonl / metrics.jsonl
    # Tensorboard is handled by analyzing these or via DDP wrapper, 
    # but the single-gpu script is safer to start if DDP caused issues before.
    
    cmd = [
        sys.executable, str(script_path),
        "--resume-from", str(checkpoint_path),
        "--pretokenized-path", str(data_path),
        "--tokenizer-path", str(tokenizer_path),
        "--run-name", run_name,
        
        # Training Params
        "--max-steps", "200000",        # Target 200k steps
        "--batch-size", "4",            # 4 batch size
        "--gradient-accumulation-steps", "4", # = 16 effective
        "--lr", "3e-5",                 # Secure, low LR
        
        # Logging
        "--eval-interval", "500",
        "--sample-interval", "500",
        "--checkpoint-interval", "2500",
        "--metrics-log-fraction", "0.0005", # Log every ~100 steps (200k * 0.0005)
        
        # Architecture (Must match checkpoint)
        "--arch-preset", "medium",      # 18 layers, 1024 dim
        # "--use-amp",                  # Default is True, arg not exposed
        "--device", "cuda"
    ]

    print("\n📜 Commande générée:")
    print(" ".join(cmd))
    
    # 3. Execution
    print("\n⚡ Démarrage de l'entraînement...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt par utilisateur")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur: {e}")

if __name__ == "__main__":
    launch_training()
