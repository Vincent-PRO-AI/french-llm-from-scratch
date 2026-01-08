#!/usr/bin/env python3
"""
Diagnostic complet de l'état d'entraînement et résolution des problèmes
"""
import os
import json
import subprocess
import torch
import sys
from pathlib import Path
from datetime import datetime

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def print_status(item, status, detail=""):
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {item:<50} {detail}")

def diagnose():
    print("\n" + "🔍 DIAGNOSTIC COMPLET DE L'ENTRAÎNEMENT".center(80))
    print("="*80)
    
    # 1. État des processus
    print_section("1. ÉTAT DES PROCESSUS")
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True,
        text=True
    )
    training_processes = [line for line in result.stdout.split('\n') 
                         if 'train_subtitles_transformer' in line and 'grep' not in line]
    
    print(f"Processus d'entraînement actifs: {len(training_processes)}")
    for proc in training_processes[:3]:  # Afficher les 3 premiers
        parts = proc.split()
        if len(parts) >= 9:
            pid = parts[1]
            cpu = parts[2]
            mem = parts[5]
            time_elapsed = parts[7]
            print(f"  PID {pid}: CPU {cpu}%, MEM {mem} ({time_elapsed})")
    
    # 2. Vérification CUDA
    print_section("2. VÉRIFICATION CUDA")
    print_status("CUDA disponible", torch.cuda.is_available())
    if torch.cuda.is_available():
        print_status("GPU", True, f"{torch.cuda.get_device_name(0)}")
        print_status("Compute capability", True, f"{torch.cuda.get_device_capability(0)}")
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / (1024**3)
        print_status("Mémoire GPU", True, f"{total_mem:.1f} GB")
    
    # 3. Vérification des fichiers critiques
    print_section("3. VÉRIFICATION DES FICHIERS")
    base_path = "/home/vincent/code/repo/french-llm-from-scratch"
    
    checkpoint_160k = Path(base_path) / "trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt"
    print_status("Checkpoint 160k", checkpoint_160k.exists(), f"{checkpoint_160k.stat().st_size/(1024**3):.2f} GB" if checkpoint_160k.exists() else "")
    
    train_data = Path(base_path) / "data_clean/conversations_mega_train_mistral_tokenized.pt"
    print_status("Données d'entraînement", train_data.exists(), f"{train_data.stat().st_size/(1024**3):.2f} GB" if train_data.exists() else "")
    
    # 4. Analyse des logs
    print_section("4. ANALYSE DES LOGS")
    log_file = Path(base_path) / "training_rtx5080.log"
    if log_file.exists():
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Chercher les erreurs
        if "CUDA error: out of memory" in content:
            print_status("Erreur CUDA OOM détectée", False, "❌ out of memory")
        elif "Traceback" in content:
            # Extraire la dernière ligne d'erreur
            lines = content.split('\n')
            error_lines = [line for line in lines if 'Error' in line or 'Exception' in line]
            if error_lines:
                print_status("Erreur détectée", False, error_lines[-1][:50])
        else:
            print_status("Pas d'erreur détectée", True)
        
        # Chercher les steps
        lines = content.split('\n')
        step_lines = [line for line in lines if 'step=' in line and 'train_loss' in line]
        if step_lines:
            latest_step = step_lines[-1]
            print(f"  Dernier step enregistré: {latest_step[:80]}")
        else:
            print("  Aucun step enregistré (entraînement n'a pas démarré)")
    else:
        print_status("Fichier de log", False, "non trouvé")
    
    # 5. Vérification des métriques
    print_section("5. FICHIERS DE SORTIE D'ENTRAÎNEMENT")
    metrics_file = Path(base_path) / "trained_models/runs/french_medium_rtx5080_extended/metrics.jsonl"
    if metrics_file.exists():
        size = metrics_file.stat().st_size
        with open(metrics_file, 'r') as f:
            lines = len(f.readlines())
        print_status("metrics.jsonl", size > 0, f"{lines} lignes, {size} bytes")
    else:
        print_status("metrics.jsonl", False, "non trouvé")
    
    checkpoints = list(Path(base_path).glob("trained_models/runs/*/checkpoint_*.pt"))
    latest_ckpts = sorted(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))[-3:]
    print(f"\nDerniers checkpoints:")
    for ckpt in latest_ckpts:
        size = ckpt.stat().st_size / (1024**3)
        step = ckpt.stem.split('_')[-1]
        print(f"  Step {step}: {size:.2f} GB")
    
    # 6. Vérification de la configuration
    print_section("6. CONFIGURATION DE L'ENTRAÎNEMENT")
    
    # Lire le script d'entraînement pour vérifier la configuration
    train_script = Path(base_path) / "scripts/train_subtitles_transformer.py"
    if train_script.exists():
        with open(train_script, 'r') as f:
            content = f.read()
        
        # Chercher les presets
        if 'MODEL_ARCH_PRESETS' in content:
            print_status("Architecture presets", True, "définis dans le script")
        
        # Chercher la configuration medium
        if '"medium"' in content:
            print_status("Architecture 'medium' (18 layers)", True)
    
    # 7. Recommandations
    print_section("7. RECOMMANDATIONS")
    
    if "CUDA error: out of memory" in content:
        print("""
❌ PROBLÈME: L'entraînement crash avec une erreur CUDA out of memory
   
🔧 SOLUTIONS À ESSAYER:

1. RÉDUIRE LA TAILLE DU BATCH (de 16 à 8 ou 4):
   $ python3 scripts/train_subtitles_transformer.py \\
       --resume-from trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt \\
       --arch-preset medium \\
       --batch-size 8 \\
       --max-steps 500000 \\
       ... [autres paramètres]

2. FORCER LA MISE EN CACHE EN VRAM:
   $ export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
   $ python3 scripts/train_subtitles_transformer.py ...

3. VÉRIFIER L'INTÉGRITÉ DU CHECKPOINT:
   $ python3 -c "
   import torch
   ckpt = torch.load('trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt', 
                       weights_only=True)
   print('Checkpoint keys:', list(ckpt.keys()))
   print('Model state shape:', len(ckpt.get('model_state', {})))
   "

4. OPTION: Relancer depuis un checkpoint plus ancien (155k ou 150k):
   $ python3 scripts/train_subtitles_transformer.py \\
       --resume-from trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_155000.pt \\
       ...

5. TUER LES PROCESSUS ZOMBIE ET REDÉMARRER:
   $ pkill -9 -f train_subtitles
   $ nvidia-smi  # Vérifier que la VRAM est libérée
   $ bash start_training.sh  # Relancer proprement
        """)
    
    print("\n" + "="*80)
    print(f"Diagnostic complété: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

if __name__ == "__main__":
    diagnose()
