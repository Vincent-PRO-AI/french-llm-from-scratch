#!/usr/bin/env python3
"""
Training optimisé single-GPU pour RTX 4090 (24GB VRAM)
Reprend depuis checkpoint 250k et continue avec le plus gros dataset tokenisé
"""
import sys
sys.path.insert(0, '/home/vincent/code/repo/french-llm-from-scratch')

import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scripts.train_subtitles_transformer import SubtitleTrainer, Config

# =========== CONFIG RTX 4090 OPTIMISÉE ===========
cfg = Config(
    # Paths
    data_dir='data_clean',
    resume_run_dir=None,  # Pas de run dir, on charge direct le checkpoint
    resume_checkpoint='trained_models/checkpoint_250k_fixed.pt',  # Direct checkpoint
    run_name='french_llm_4090_optimized_250k_to_500k',
    log_dir='trained_models/logs',
    
    # Model (18 layers, 260M params)
    model_arch='tiny',
    embed_dim=1024,
    num_heads=16,
    num_layers=18,
    ff_hidden_dim=4096,
    vocab_size=32000,
    
    # Training params - OPTIMISÉ 4090
    batch_size=6,  # 4090 peut tenir 6 (vs 1-2 sur 5080)
    gradient_accumulation_steps=16,  # Effective batch = 96
    max_steps=500000,  # Continuer jusqu'à 500k
    lr=0.0001,
    weight_decay=0.0,
    
    # Checkpointing & Logging
    checkpoint_interval=5000,
    eval_interval=500,
    sample_interval=500,
    metrics_log_fraction=0.05,
    
    # Data
    block_size=1024,
    min_seq_len=512,
    train_split=0.95,
    shuffle=True,
    pretokenized_path='data_clean/french_large_filtered_mistral_tokenized.pt', # Forcer le gros dataset
    
    # Optimization
    use_amp=True,
    num_workers=1,  # Très bas pour éviter OOM RAM (80GB total)
    pin_memory=False, 
    prefetch_factor=2,
    persistent_workers=False,
    auto_ram_tune=False,  # Désactiver l'auto-adapt pour garder nos params
    auto_resource_adapt=False,  # Désactiver aussi celui-ci
    gradient_checkpointing=True,
    
    # Device
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

print(f"""
╔═══════════════════════════════════════════════════════╗
║   🚀 Training RTX 4090 OPTIMISÉ                       ║
║   📊 French LLM: 250k → 500k steps                    ║
║   🔧 Config: batch=6, grad_accum=16 (eff. 96)        ║
║   💾 Dataset: french_large_filtered_mistral (4GB)    ║
║   ⏱️  ETA: 48-72h continu                             ║
╚═══════════════════════════════════════════════════════╝
""")

try:
    print("[INFO] Initializing trainer...")
    trainer = SubtitleTrainer(cfg)
    
    print(f"[INFO] Starting from step {trainer.start_step}")
    print(f"[INFO] Device: {cfg.device}")
    print(f"[INFO] Model params: {sum(p.numel() for p in trainer.model.parameters())/1e6:.0f}M")
    
    print("\n" + "="*60)
    print("🔥 DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("="*60 + "\n")
    
    trainer.run()
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT COMPLET!")
    print("="*60)
    
except KeyboardInterrupt:
    print("\n⏸️  Interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
