#!/usr/bin/env python3
"""
Préparation pour redémarrage de l'entraînement
Teste le checkpoint et prépare le relancement
"""

import torch
import json
from pathlib import Path
import subprocess
import sys

def prepare_restart():
    print("\n" + "="*80)
    print("  📋 PRÉPARATION POUR REDÉMARRAGE")
    print("="*80 + "\n")
    
    base_path = Path("/home/vincent/code/repo/french-llm-from-scratch")
    
    # Trouver le dernier checkpoint
    checkpoint_dir = base_path / "trained_models/runs/french_medium_rtx5080_batch8"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        step = int(latest_ckpt.stem.split('_')[-1])
        
        print(f"✅ CHECKPOINT TROUVÉ")
        print(f"   Chemin: {latest_ckpt.name}")
        print(f"   Taille: {latest_ckpt.stat().st_size / (1024**3):.2f} GB")
        print(f"   Step: {step}")
        
        # Vérifier l'intégrité
        print(f"\n📊 VÉRIFICATION DE L'INTÉGRITÉ")
        print("-" * 80)
        
        try:
            ckpt = torch.load(latest_ckpt, map_location='cpu', weights_only=True)
            print(f"✅ Checkpoint chargeable")
            print(f"   Keys: {list(ckpt.keys())}")
            print(f"   Model state dict size: {len(ckpt.get('model_state', {}))}")
            
            if 'optimizer_state' in ckpt:
                print(f"   Optimizer state: ✅ Présent")
            if 'rng_state' in ckpt:
                print(f"   RNG state: ✅ Présent")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
        
        # Afficher les dernières métriques
        metrics_file = checkpoint_dir / "metrics.jsonl"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
            
            print(f"\n📈 MÉTRIQUES D'ENTRAÎNEMENT")
            print("-" * 80)
            print(f"Nombre de steps enregistrés: {len(lines)}")
            
            if lines:
                last_metric = json.loads(lines[-1])
                print(f"Step: {last_metric.get('step')}")
                print(f"Loss: {last_metric.get('loss'):.6f}")
                print(f"Timestamp: {last_metric.get('timestamp')}")
        
        # Préparer la commande de relance
        print(f"\n🚀 COMMANDE DE RELANCE PRÉPARÉE")
        print("-" * 80)
        
        cmd = f"""
cd /home/vincent/code/repo/french-llm-from-scratch && \\
source .venv/bin/activate && \\
nohup python3 scripts/train_subtitles_transformer.py \\
    --resume-from {latest_ckpt} \\
    --arch-preset medium \\
    --run-name french_medium_rtx5080_extended \\
    --batch-size 12 \\
    --lr 1.5e-4 \\
    --max-steps 500000 \\
    --eval-interval 250 \\
    --sample-interval 500 \\
    --checkpoint-interval 5000 \\
    --metrics-log-fraction 0.02 \\
    --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \\
    --tokenizer-path data_clean/mistral_tokenizer \\
    --gradient-accumulation-steps 1 \\
    --num-workers 8 \\
    --prefetch-factor 4 \\
    --auto-resource-adapt \\
    > training_resumed.log 2>&1 &
"""
        
        print("Commande sauvegardée dans: restart_training.sh")
        print("\nPour relancer:")
        print("  $ bash restart_training.sh")
        
        # Sauvegarder la commande
        with open(base_path / "restart_training.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(cmd.strip())
        
        import os
        os.chmod(base_path / "restart_training.sh", 0o755)
        
        print(f"\n✅ PRÉPARATION COMPLÈTE")
        print(f"   Checkpoint: {step}")
        print(f"   Cible: 500000 steps")
        print(f"   Steps restants: {500000 - step}")
        print(f"   Durée estimée: {(500000 - step) / 3.3 / 3600:.1f} heures")
        
        return True
    else:
        print("❌ Aucun checkpoint trouvé")
        return False

if __name__ == "__main__":
    success = prepare_restart()
    sys.exit(0 if success else 1)
