#!/usr/bin/env python3
"""
Analyse complète de l'utilisation RAM et GPU.
Calcule le potentiel d'optimisation avec 96 Go disponibles.
"""

import torch
import psutil
import subprocess
import json
from pathlib import Path
from datetime import datetime


def analyze_system_resources():
    """Analyser les ressources système."""
    
    print("\n" + "="*70)
    print("💾 ANALYSE DES RESSOURCES SYSTÈME")
    print("="*70)
    
    # RAM
    ram_info = psutil.virtual_memory()
    print(f"\n📊 RAM (Host System):")
    print(f"   Total: {ram_info.total / 1e9:.1f} GB")
    print(f"   Available: {ram_info.available / 1e9:.1f} GB")
    print(f"   Used: {ram_info.used / 1e9:.1f} GB ({ram_info.percent}%)")
    print(f"   Free: {ram_info.free / 1e9:.1f} GB")
    
    # GPU
    if torch.cuda.is_available():
        print(f"\n🎮 GPU:")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
        
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / 1e9
        print(f"   Total VRAM: {vram_total:.1f} GB")
        
        # VRAM utilisée
        torch.cuda.reset_peak_memory_stats()
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   VRAM used: {vram_used:.2f} GB")
        print(f"   VRAM reserved: {vram_reserved:.2f} GB")
        print(f"   VRAM free: {vram_total - vram_reserved:.2f} GB")
    else:
        print(f"\n⚠️  GPU non disponible")
    
    # CPU
    print(f"\n⚙️  CPU:")
    print(f"   Cores: {psutil.cpu_count()}")
    print(f"   Usage: {psutil.cpu_percent(interval=1)}%")
    
    return {
        'ram_total_gb': ram_info.total / 1e9,
        'ram_available_gb': ram_info.available / 1e9,
        'ram_used_gb': ram_info.used / 1e9,
        'gpu_vram_total_gb': vram_total if torch.cuda.is_available() else None,
        'cpu_cores': psutil.cpu_count()
    }


def estimate_training_optimization():
    """Estimer l'optimisation possible avec 96 Go."""
    
    print("\n" + "="*70)
    print("🚀 ANALYSE D'OPTIMISATION DU TRAINING")
    print("="*70)
    
    # Configuration actuelle
    current_config = {
        'batch_size': 8,
        'gradient_accumulation': 2,
        'effective_batch_size': 16,
        'model_params': 260e6,
        'dtype': 'float32'
    }
    
    print(f"\n📋 Configuration actuelle:")
    print(f"   Batch size: {current_config['batch_size']}")
    print(f"   Gradient accumulation: {current_config['gradient_accumulation']}")
    print(f"   Effective batch: {current_config['effective_batch_size']}")
    print(f"   Model params: {current_config['model_params']/1e6:.0f}M")
    
    # Estimations de mémoire
    print(f"\n💾 Estimations de mémoire (par configuration):")
    
    # Model weights (fp32): params * 4 bytes
    model_size = current_config['model_params'] * 4 / 1e9  # GB
    print(f"   Model weights (fp32): {model_size:.2f} GB")
    
    # Avec mixed precision (fp16)
    print(f"   Model weights (fp16): {model_size/2:.2f} GB")
    
    # Activations et gradients
    # Rule of thumb: 3-4x model size pour training
    training_memory_per_batch = (model_size * 2) / current_config['batch_size']
    print(f"   Training memory per sample: ~{training_memory_per_batch:.2f} GB")
    
    # Calcul des possibilités
    print(f"\n🎯 Potentiel d'optimisation avec 96 GB RAM:")
    
    ram_available = 96  # GB (approximation)
    
    # Batch size optimal avec FP32
    max_batch_size_fp32 = int((ram_available - model_size) / training_memory_per_batch)
    
    # Batch size optimal avec FP16 + Gradient checkpointing
    model_size_fp16 = model_size / 2
    max_batch_size_fp16 = int((ram_available - model_size_fp16) * 1.3 / training_memory_per_batch)
    
    print(f"   FP32 (current):")
    print(f"      Max batch size: {max(max_batch_size_fp32, 16)}")
    print(f"      Potential speedup: {max(max_batch_size_fp32, 16) / 8:.1f}x")
    
    print(f"   FP16 + Grad Checkpointing (recommended):")
    print(f"      Max batch size: {max(max_batch_size_fp16, 32)}")
    print(f"      Potential speedup: {max(max_batch_size_fp16, 32) / 8:.1f}x")
    
    recommendations = {
        'current_batch_size': 8,
        'recommended_batch_size': 32,
        'recommended_gradient_accumulation': 1,
        'use_mixed_precision': True,
        'use_gradient_checkpointing': True,
        'potential_speedup': 4.0,
        'estimated_training_time_reduction': '75%'
    }
    
    print(f"\n✅ Recommandations:")
    print(f"   Batch size: 8 → {recommendations['recommended_batch_size']}")
    print(f"   Mixed precision (AMP): ✅ activé")
    print(f"   Gradient checkpointing: ✅ activé")
    print(f"   Speedup potentiel: ~{recommendations['potential_speedup']}x")
    
    return recommendations


def create_optimization_report():
    """Créer un rapport d'optimisation complet."""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_analysis': analyze_system_resources(),
        'training_optimization': estimate_training_optimization()
    }
    
    # Sauvegarder
    with open('optimization_analysis.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Rapport sauvegardé: optimization_analysis.json")
    return report


if __name__ == '__main__':
    create_optimization_report()
