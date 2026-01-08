#!/usr/bin/env python3
"""
Script de correction automatique - résout le problème CUDA OOM
Réduit batch_size et relance l'entraînement proprement
"""

import os
import subprocess
import signal
import time
import sys
from pathlib import Path

def run_cmd(cmd, check=True):
    """Exécuter une commande shell"""
    print(f"▶️  {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Erreur: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout.strip())
    return True

def main():
    print("\n" + "="*80)
    print("  🔧 CORRECTION AUTOMATIQUE - CUDA OOM")
    print("="*80 + "\n")
    
    base_path = "/home/vincent/code/repo/french-llm-from-scratch"
    os.chdir(base_path)
    
    # ÉTAPE 1: Tuer les processus zombie
    print("📌 ÉTAPE 1: Arrêter les processus zombie")
    print("-" * 80)
    
    result = subprocess.run(
        ["pgrep", "-f", "train_subtitles_transformer"],
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        pids = result.stdout.strip().split('\n')
        print(f"Processus trouvés: {len(pids)}")
        for pid in pids:
            print(f"  Tuer PID {pid}...")
            os.system(f"kill -9 {pid}")
        time.sleep(2)
    
    # Vérifier que tout est mort
    result = subprocess.run(
        ["pgrep", "-f", "train_subtitles_transformer"],
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print("⚠️  Certains processus restent actifs, attendre...")
        time.sleep(5)
        os.system("pkill -9 -f train_subtitles")
        time.sleep(2)
    else:
        print("✅ Tous les processus ont été arrêtés")
    
    # ÉTAPE 2: Libérer la VRAM
    print("\n📌 ÉTAPE 2: Libérer la mémoire GPU")
    print("-" * 80)
    
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)
    
    # ÉTAPE 3: Relancer avec batch_size 8
    print("\n📌 ÉTAPE 3: Relancer l'entraînement (batch_size=8)")
    print("-" * 80)
    
    cmd = """
    source .venv/bin/activate && \\
    nohup python3 scripts/train_subtitles_transformer.py \\
        --resume-from trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt \\
        --arch-preset medium \\
        --run-name french_medium_rtx5080_batch8 \\
        --batch-size 8 \\
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
        > training_corrected.log 2>&1 &
    """
    
    result = os.system(cmd)
    time.sleep(3)
    
    # ÉTAPE 4: Vérifier que ça a démarré
    print("\n📌 ÉTAPE 4: Vérification du lancement")
    print("-" * 80)
    
    result = subprocess.run(
        ["pgrep", "-f", "train_subtitles_transformer"],
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        pids = result.stdout.strip().split('\n')
        print(f"✅ Processus lancé avec succès!")
        print(f"  PID: {pids[0]}")
        print(f"  Logs: tail -f training_corrected.log")
    else:
        print("❌ Échec du lancement")
        return 1
    
    print("\n" + "="*80)
    print("  ✅ CORRECTION LANCÉE")
    print("="*80)
    print("""
🎯 PROCHAINES ÉTAPES:

1. Vérifier les logs:
   $ tail -f training_corrected.log

2. Suivre la progression GPU:
   $ watch nvidia-smi

3. Si le problème persiste avec batch_size=8:
   → Réduire à batch_size=4
   → Ou vérifier l'intégrité du checkpoint 160k

4. Si ça fonctionne:
   → Attendre le prochain checkpoint (~165k)
   → Puis passer le reste du plan

    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
