#!/usr/bin/env python3
"""Lance Phase 2B: 110k → 130k steps sur dataset 100% français."""

import subprocess
import sys
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("🚀 LANCEMENT PHASE 2B - TRAINING 100% FRANÇAIS")
    print("="*70)
    
    # Configuration
    checkpoint = "trained_models/runs/french_v2_phase2a_final/checkpoint_step_110000.pt"
    dataset = "data_clean/conversations_train_20M_tokenized.pt"
    run_name = "french_v2_phase2b_100pct_fr"
    
    # Vérifications
    if not Path(checkpoint).exists():
        print(f"❌ Checkpoint introuvable: {checkpoint}")
        return 1
    
    if not Path(dataset).exists():
        print(f"❌ Dataset introuvable: {dataset}")
        return 1
    
    print(f"\n✅ Checkpoint: {Path(checkpoint).name}")
    print(f"✅ Dataset: {Path(dataset).name} (100% FR, 653MB)")
    print(f"✅ Steps: 110,000 → 130,000 (+20k)")
    print(f"✅ Run: {run_name}")
    
    # Commande
    cmd = [
        ".venv/bin/python3",
        "simple_train.py",
        "--max-steps", "130000",
        "--lr", "1e-6",
        "--batch-size", "10",
        "--seq-len", "512",
        "--checkpoint-interval", "1000",
        "--run-name", run_name
    ]
    
    print(f"\n📋 Lancement training...")
    print(f"   {' '.join(cmd)}\n")
    
    # Lancer
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Afficher output en temps réel
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        return process.returncode
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrompu par l'utilisateur")
        process.terminate()
        return 1
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
