#!/usr/bin/env python3
"""
Dashboard simple pour suivre la progression d'entraînement
Affiche les métriques en temps réel avec live-tail
"""
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from collections import deque

def format_time(seconds):
    """Formate les secondes en HH:MM:SS"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

def get_eta(steps, steps_per_sec):
    """Calcule ETA basé sur les steps restants"""
    remaining = 500000 - steps
    if steps_per_sec <= 0:
        return "∞"
    eta_seconds = remaining / steps_per_sec
    return format_time(eta_seconds)

def main():
    metrics_file = Path("trained_models/runs/french_medium_rtx5080_extended/metrics.jsonl")
    
    print("=" * 80)
    print("🚀 DASHBOARD ENTRAÎNEMENT - French LLM RTX 5080")
    print("=" * 80)
    
    last_metrics = None
    step_times = deque(maxlen=100)
    
    while True:
        if not metrics_file.exists():
            print("⏳ Attente du fichier metrics...")
            time.sleep(5)
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                time.sleep(5)
                continue
            
            metrics = json.loads(lines[-1])
            
            if metrics == last_metrics:
                time.sleep(5)
                continue
            
            last_metrics = metrics
            
            step = metrics.get('step', 0)
            loss = metrics.get('loss', 0)
            val_loss = metrics.get('val_loss', None)
            learning_rate = metrics.get('lr', 0)
            elapsed = metrics.get('elapsed_seconds', 0)
            
            # Calculer la vitesse
            if elapsed > 0:
                steps_per_sec = step / elapsed
            else:
                steps_per_sec = 0
            
            # Barre de progression
            progress = (step / 500000) * 100
            filled = int(progress / 2)
            bar = "█" * filled + "░" * (50 - filled)
            
            # Affichage
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Étape {step:,}/500,000")
            print(f"├─ Progression: [{bar}] {progress:5.1f}%")
            print(f"├─ Loss Train: {loss:7.4f}", end="")
            if val_loss:
                print(f" | Loss Val: {val_loss:7.4f}")
            else:
                print()
            print(f"├─ LR: {learning_rate:.2e} | Vitesse: {steps_per_sec:.2f} steps/s")
            print(f"├─ Temps écoulé: {format_time(elapsed)}")
            print(f"└─ ETA: {get_eta(step, steps_per_sec)}")
            
            if step >= 500000:
                print("\n" + "=" * 80)
                print("✅ ENTRAÎNEMENT TERMINÉ!")
                print("=" * 80)
                return
            
            time.sleep(10)
            
        except Exception as e:
            print(f"⚠️  Erreur: {e}")
            time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard arrêté")
