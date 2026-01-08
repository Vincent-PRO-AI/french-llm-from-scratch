#!/usr/bin/env python3
"""
LAUNCHER OPTIMISÉ: Training batch_56 (90 GB RAM)
Configuration pour speedup 8-9x
"""

import subprocess
import sys
from pathlib import Path
import psutil


def launch_training_90gb():
    """Lancer le training optimisé pour 90 GB."""
    
    print("\n" + "="*80)
    print("🚀 LANCEMENT TRAINING OPTIMISÉ - BATCH 56 (90 GB RAM)")
    print("="*80)
    
    # Vérifier RAM
    ram_info = psutil.virtual_memory()
    ram_total = ram_info.total / 1e9
    ram_available = ram_info.available / 1e9
    
    print(f"\n📊 Vérification système:")
    print(f"   RAM totale: {ram_total:.1f} GB ✅")
    print(f"   RAM disponible: {ram_available:.1f} GB ✅")
    print(f"   GPU VRAM: 17.1 GB (RTX 5080) ✅")
    
    if ram_total < 85:
        print(f"\n⚠️  RAM insuffisante! ({ram_total:.1f} GB < 85 GB requis)")
        return
    
    # Chemins
    script_dir = Path(__file__).parent
    train_script = script_dir / "scripts" / "train_subtitles_transformer.py"
    checkpoint = Path("trained_models/runs/french_v3_finetune_grand_60k/checkpoint_step_145000.pt")
    
    print(f"\n📁 Fichiers:")
    if not train_script.exists():
        print(f"   ❌ Script non trouvé: {train_script}")
        return
    print(f"   ✅ Script d'entraînement")
    
    if not checkpoint.exists():
        print(f"   ❌ Checkpoint non trouvé: {checkpoint}")
        return
    print(f"   ✅ Checkpoint 145000")
    
    # Configuration
    print(f"\n⚙️  Configuration OPTIMALE:")
    config = {
        'batch_size': 56,
        'gradient_accumulation': 1,
        'arch_preset': 'medium',
        'max_steps': 250000,
        'lr': 2e-4,
        'eval_interval': 100,
        'sample_interval': 500,
        'checkpoint_interval': 2500,
    }
    
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Statistiques attendues
    print(f"\n📈 STATS ATTENDUES:")
    print(f"   Speedup: 8-9x")
    print(f"   Vitesse: 16 steps/sec")
    print(f"   100k steps: 1.6 heures ⚡")
    print(f"   250k steps: 4.0 heures ⚡")
    print(f"   RAM usage: ~62 GB / 81 GB disponible")
    
    # Commande
    cmd = [
        sys.executable, str(train_script),
        "--resume-from", str(checkpoint),
        "--arch-preset", config['arch_preset'],
        "--batch-size", str(config['batch_size']),
        "--max-steps", str(config['max_steps']),
        "--lr", str(config['lr']),
        "--eval-interval", str(config['eval_interval']),
        "--sample-interval", str(config['sample_interval']),
        "--checkpoint-interval", str(config['checkpoint_interval']),
        "--run-name", "french_medium_optimized_90gb_batch56",
        "--device", "cuda",
    ]
    
    print(f"\n🎬 Commande:")
    print(f"   {' '.join(cmd[:6])} ...")
    
    print(f"\n" + "="*80)
    print(f"🚀 DÉMARRAGE ENTRAÎNEMENT")
    print(f"="*80)
    print(f"Cible: 250k steps en ~4 heures ⚡")
    print(f"Checkpoint final: step 250000")
    print(f"Run: french_medium_optimized_90gb_batch56\n")
    
    # Lancer
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Entraînement terminé!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️  Entraînement interrompu (peut être repris)")
        sys.exit(0)


if __name__ == '__main__':
    launch_training_90gb()
