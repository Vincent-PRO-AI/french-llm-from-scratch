#!/usr/bin/env python3
"""
Analyse détaillée des métriques de training.
Génère des graphiques et rapports statistiques.
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict

def analyze_metrics(metrics_file):
    """Analyser les métriques d'entraînement."""
    
    print(f"\n📊 ANALYSE DES MÉTRIQUES")
    print(f"{'='*70}")
    print(f"Fichier: {metrics_file}")
    
    # Charger les métriques
    metrics = defaultdict(list)
    steps_train = []
    steps_val = []
    
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print("❌ Fichier vide")
        return None
    
    for line in lines:
        data = json.loads(line)
        split = data['split']
        step = data['step']
        loss = data['loss']
        
        metrics[split].append(loss)
        if split == 'train':
            steps_train.append(step)
        else:
            steps_val.append(step)
    
    # Statistiques
    print(f"\n1️⃣ COUVERTURE")
    print(f"   Train steps: {len(metrics['train'])}")
    print(f"   Val steps: {len(metrics['val'])}")
    
    if metrics['train']:
        print(f"\n2️⃣ LOSS - TRAIN")
        train_loss = np.array(metrics['train'])
        print(f"   Minimum: {train_loss.min():.4f}")
        print(f"   Maximum: {train_loss.max():.4f}")
        print(f"   Moyenne: {train_loss.mean():.4f}")
        print(f"   Std: {train_loss.std():.4f}")
        print(f"   Derniers 100 steps (moyenne): {train_loss[-100:].mean():.4f}")
        print(f"   Tendance: {'↓ baisse' if train_loss[-100:].mean() < train_loss[-200:-100].mean() else '↑ hausse'}")
    
    if metrics['val']:
        print(f"\n3️⃣ LOSS - VALIDATION")
        val_loss = np.array(metrics['val'])
        print(f"   Minimum: {val_loss.min():.4f}")
        print(f"   Maximum: {val_loss.max():.4f}")
        print(f"   Moyenne: {val_loss.mean():.4f}")
        print(f"   Std: {val_loss.std():.4f}")
        print(f"   Derniers 100 steps (moyenne): {val_loss[-100:].mean():.4f}")
        print(f"   Tendance: {'↓ baisse' if val_loss[-100:].mean() < val_loss[-200:-100].mean() else '↑ hausse'}")
    
    if metrics['train'] and metrics['val']:
        # Overfitting check
        train_recent = np.array(metrics['train'][-100:]).mean()
        val_recent = np.array(metrics['val'][-100:]).mean()
        gap = val_recent - train_recent
        
        print(f"\n4️⃣ OVERFITTING CHECK")
        print(f"   Train loss (100 derniers): {train_recent:.4f}")
        print(f"   Val loss (100 derniers): {val_recent:.4f}")
        print(f"   Gap: {gap:.4f}")
        
        if gap < 0.2:
            print(f"   ✅ Gap faible - bonne généralisation")
        elif gap < 0.5:
            print(f"   ⚠️  Gap modéré - overfitting léger")
        else:
            print(f"   ❌ Gap élevé - overfitting sévère")
    
    print(f"\n{'='*70}")
    return {
        'train_steps': len(metrics['train']),
        'val_steps': len(metrics['val']),
        'train_loss': train_loss.tolist() if metrics['train'] else [],
        'val_loss': val_loss.tolist() if metrics['val'] else []
    }


def main():
    # Trouver tous les fichiers de métriques
    metrics_files = list(Path('trained_models/runs').glob('*/metrics.jsonl'))
    
    if not metrics_files:
        print("❌ Aucun fichier de métriques trouvé")
        return
    
    # Analyser les plus importants
    important_runs = [
        'french_v3_finetune_grand_60k',
        'french_medium_rtx5080_batch8',
        'french_medium_rtx5080_extended',
        'french_large_rtx5080_500k'
    ]
    
    results = {}
    for run_name in important_runs:
        metrics_file = Path(f'trained_models/runs/{run_name}/metrics.jsonl')
        if metrics_file.exists():
            result = analyze_metrics(str(metrics_file))
            if result:
                results[run_name] = result
    
    # Comparaison
    print(f"\n📈 COMPARAISON DES RUNS")
    print(f"{'='*70}")
    for run_name, data in results.items():
        if data and data['train_steps'] > 0:
            final_train = data['train_loss'][-1] if data['train_loss'] else 'N/A'
            final_val = data['val_loss'][-1] if data['val_loss'] else 'N/A'
            print(f"\n{run_name}")
            print(f"   Train steps: {data['train_steps']}")
            print(f"   Final train loss: {final_train}")
            print(f"   Final val loss: {final_val}")


if __name__ == '__main__':
    main()
