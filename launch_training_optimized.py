#!/usr/bin/env python3
"""
LAUNCHER: Training optimisé pour batch_size 32-64
À lancer APRÈS augmentation RAM à 96 GB
"""

import subprocess
import sys
from pathlib import Path

def launch_optimized_training():
    """Lancer le training avec configuration optimisée."""
    
    print("\n" + "="*80)
    print("🚀 LANCEMENT TRAINING OPTIMISÉ - BATCH 64")
    print("="*80)
    
    # Vérifier la RAM disponible
    import psutil
    ram_info = psutil.virtual_memory()
    ram_total = ram_info.total / 1e9
    ram_available = ram_info.available / 1e9
    
    print(f"\n📊 État système:")
    print(f"   RAM totale: {ram_total:.1f} GB")
    print(f"   RAM disponible: {ram_available:.1f} GB")
    
    if ram_total < 90:
        print(f"\n⚠️  ATTENTION: RAM insuffisante detectée ({ram_total:.1f} GB < 96 GB)")
        print(f"   Options:")
        print(f"     1. Attendre la croissance de l'allocation Hyper-V")
        print(f"     2. Redémarrer la VM complètement")
        print(f"     3. Augmenter manuellement dans Hyper-V Manager")
        return
    
    print(f"   ✅ RAM suffisante pour batch_64!")
    
    # Chemins
    script_dir = Path(__file__).parent
    train_script = script_dir / "scripts" / "train_subtitles_transformer.py"
    checkpoint = Path("trained_models/runs/french_v3_finetune_grand_60k/checkpoint_step_145000.pt")
    
    # Vérifier les fichiers
    print(f"\n📁 Vérification des fichiers:")
    if not train_script.exists():
        print(f"   ❌ Script non trouvé: {train_script}")
        return
    print(f"   ✅ Script d'entraînement")
    
    if not checkpoint.exists():
        print(f"   ❌ Checkpoint non trouvé: {checkpoint}")
        return
    print(f"   ✅ Checkpoint 145000")
    
    # Configuration
    print(f"\n⚙️  Configuration:")
    config = {
        'batch_size': 64,
        'gradient_accumulation': 1,
        'mixed_precision': True,
        'gradient_checkpointing': True,
        'data_caching': True,
        'arch_preset': 'medium',
        'max_steps': 250000,
        'lr': 2e-4,
        'eval_interval': 100,
        'sample_interval': 500,
        'checkpoint_interval': 2500,
    }
    
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Commande
    cmd = [
        sys.executable, str(train_script),
        "--resume-from", str(checkpoint),
        "--arch-preset", "medium",
        "--batch-size", str(config['batch_size']),
        "--max-steps", str(config['max_steps']),
        "--lr", str(config['lr']),
        "--eval-interval", str(config['eval_interval']),
        "--sample-interval", str(config['sample_interval']),
        "--checkpoint-interval", str(config['checkpoint_interval']),
        "--run-name", "french_medium_optimized_batch64",
        "--device", "cuda",
    ]
    
    print(f"\n🎬 Commande:")
    print(f"   {' '.join(cmd[:5])} ...")
    
    print(f"\n" + "="*80)
    print(f"📈 STATS ATTENDUES:")
    print(f"="*80)
    print(f"   Speedup: 10-12x")
    print(f"   Vitesse: ~20 steps/sec")
    print(f"   100k steps: ~1.4 heures")
    print(f"   250k steps: ~3.5 heures")
    
    print(f"\n🚀 DÉMARRAGE ENTRAÎNEMENT...")
    print(f"="*80 + "\n")
    
    # Lancer le training
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Entraînement terminé!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️  Entraînement interrompu")
        sys.exit(0)


if __name__ == '__main__':
    launch_optimized_training()
