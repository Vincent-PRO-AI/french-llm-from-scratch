#!/usr/bin/env python3
"""Script qui surveille le training et lance automatiquement 20k steps sur un autre dataset."""

import json
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Configuration
METRICS_FILE = Path("trained_models/runs/french_v2_phase2a_final/metrics.jsonl")
TARGET_STEP = 110000
NEXT_DATASET = "data_clean/conversations_train_20M_tokenized.pt"  # 100% FR, 20M tokens
NEXT_RUN_NAME = "french_v2_phase2b_continuation"
NEXT_STEPS = 130000  # +20k steps
CHECKPOINT_DIR = Path("trained_models/runs/french_v2_phase2a_final")

# Paramètres overfit
OVERFIT_THRESHOLD = 15.0  # Si loss remonte de +15 depuis le min
WINDOW_SIZE = 50  # Fenêtre pour calculer la tendance


def get_current_step():
    """Récupère le step actuel depuis le fichier de métriques."""
    if not METRICS_FILE.exists():
        return None
    
    try:
        with open(METRICS_FILE, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                data = json.loads(last_line)
                return data.get('step')
    except:
        return None
    return None


def analyze_overfit():
    """Analyse si le modèle overfit."""
    if not METRICS_FILE.exists():
        return False, "Pas de métriques disponibles"
    
    try:
        losses = []
        with open(METRICS_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    losses.append(data['loss'])
                except:
                    continue
        
        if len(losses) < WINDOW_SIZE:
            return False, "Pas assez de données"
        
        recent_losses = losses[-WINDOW_SIZE:]
        min_loss = min(losses)
        current_loss = losses[-1]
        avg_recent = sum(recent_losses) / len(recent_losses)
        
        overfit_score = current_loss - min_loss
        
        status = {
            'min_loss': min_loss,
            'current_loss': current_loss,
            'avg_recent': avg_recent,
            'overfit_score': overfit_score,
            'is_overfit': overfit_score > OVERFIT_THRESHOLD
        }
        
        return status['is_overfit'], status
    
    except Exception as e:
        return False, f"Erreur: {e}"


def launch_next_phase():
    """Lance automatiquement la Phase 2B sur un nouveau dataset."""
    
    print("\n" + "="*70)
    print("🚀 LANCEMENT AUTOMATIQUE PHASE 2B")
    print("="*70)
    
    # Trouver le dernier checkpoint
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        print("❌ Aucun checkpoint trouvé!")
        return False
    
    latest_checkpoint = checkpoints[-1]
    print(f"\n✅ Checkpoint: {latest_checkpoint.name}")
    
    # Vérifier le dataset
    dataset_path = Path(NEXT_DATASET)
    if not dataset_path.exists():
        print(f"❌ Dataset introuvable: {dataset_path}")
        return False
    
    print(f"✅ Dataset: {dataset_path.name} (100% Français)")
    
    # Analyser l'overfit
    is_overfit, status = analyze_overfit()
    
    if is_overfit:
        print(f"\n⚠️  OVERFIT DÉTECTÉ!")
        print(f"   Loss min: {status['min_loss']:.2f}")
        print(f"   Loss actuelle: {status['current_loss']:.2f}")
        print(f"   Écart: +{status['overfit_score']:.2f}")
        print(f"   → Changement de dataset NÉCESSAIRE")
    else:
        print(f"\n✅ Pas d'overfit majeur")
        print(f"   Loss min: {status['min_loss']:.2f}")
        print(f"   Loss actuelle: {status['current_loss']:.2f}")
        print(f"   → Diversification dataset recommandée")
    
    # Construire la commande
    cmd = [
        sys.executable,
        "simple_train.py",
        "--resume-from", str(latest_checkpoint),
        "--dataset", str(dataset_path),
        "--max-steps", str(NEXT_STEPS),
        "--lr", "1e-6",
        "--batch-size", "10",
        "--seq-len", "512",
        "--checkpoint-interval", "1000",
        "--run-name", NEXT_RUN_NAME
    ]
    
    print(f"\n📋 Commande:")
    print(f"   {' '.join(cmd)}")
    
    # Lancer en arrière-plan
    log_file = f"training_{NEXT_RUN_NAME}.log"
    print(f"\n📝 Logs: {log_file}")
    
    print(f"\n⏱️  Configuration:")
    print(f"   Steps: {TARGET_STEP} → {NEXT_STEPS} (+{NEXT_STEPS - TARGET_STEP} steps)")
    print(f"   Dataset: 100% Français (conversations_train_20M)")
    print(f"   Learning Rate: 1e-6 (conservateur)")
    print(f"   Batch Size: 10")
    print(f"   Checkpoints: Tous les 1000 steps")
    
    print(f"\n🎯 Objectif:")
    print(f"   • Diversifier l'exposition linguistique")
    print(f"   • Éviter l'overfit sur conversations_mega")
    print(f"   • Stabiliser les performances")
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path.cwd()
            )
        
        print(f"\n✅ Phase 2B lancée! PID: {process.pid}")
        print(f"\n📊 Monitoring:")
        print(f"   Dashboard: http://localhost:5000")
        print(f"   Logs: tail -f {log_file}")
        print(f"   Métriques: trained_models/runs/{NEXT_RUN_NAME}/metrics.jsonl")
        
        print("\n" + "="*70)
        print("🎉 TRANSITION AUTOMATIQUE RÉUSSIE!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lancement: {e}")
        return False


def main():
    print("🔍 Surveillance du training Phase 2A...")
    print(f"   Target step: {TARGET_STEP}")
    print(f"   Prochain dataset: {NEXT_DATASET}")
    print(f"   Prochain objectif: {NEXT_STEPS} steps (+20k)\n")
    
    last_reported_step = None
    
    while True:
        current_step = get_current_step()
        
        if current_step is None:
            print("⏳ En attente du démarrage du training...")
            time.sleep(30)
            continue
        
        # Rapport toutes les 1000 steps
        if last_reported_step is None or current_step >= last_reported_step + 1000:
            progress = ((current_step - 100000) / 10000) * 100
            remaining = TARGET_STEP - current_step
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Step {current_step:,} / {TARGET_STEP:,} "
                  f"({progress:.1f}% | {remaining:,} restants)")
            last_reported_step = current_step
        
        # Vérifier si on a atteint l'objectif
        if current_step >= TARGET_STEP:
            print(f"\n✅ Phase 2A terminée! Step {current_step} atteint.")
            
            # Attendre quelques secondes pour s'assurer que le checkpoint est sauvegardé
            print("⏳ Attente checkpoint final (10s)...")
            time.sleep(10)
            
            # Lancer la phase suivante
            success = launch_next_phase()
            
            if success:
                print("✅ Continuez à surveiller via le dashboard!")
                sys.exit(0)
            else:
                print("❌ Échec du lancement automatique")
                sys.exit(1)
        
        # Vérifier toutes les 30 secondes
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Surveillance interrompue par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)
