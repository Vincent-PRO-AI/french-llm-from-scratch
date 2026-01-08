#!/usr/bin/env python3
"""
Lancement training optimisé avec support multi-GPU et FP8 (optionnel)

OPTIMISATIONS IMPLÉMENTÉES:
1. Multi-GPU avec DataParallel ou DistributedDataParallel
2. Batch size augmenté (utilise plus de VRAM)
3. Gradient checkpointing (économise VRAM)
4. DataLoader workers augmentés (utilise plus de RAM)
5. Pin memory pour transferts CPU→GPU rapides
6. Support FP8 avec transformer_engine (si disponible)
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

# Configuration
CHECKPOINT_PATH = "trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_160000.pt"
DATASET_PATH = "data_clean/french_large_filtered_mistral_tokenized.pt"
OUTPUT_DIR = "trained_models/runs/french_multi_gpu_optimized"
MAX_STEPS = 220000

# OPTIMISATIONS PERFORMANCE
NUM_GPUS = torch.cuda.device_count()
BATCH_SIZE_PER_GPU = 8  # Augmenté de 4 à 8 (utilise plus de VRAM)
GRAD_ACCUMULATION = 2   # Batch effectif = NUM_GPUS * 8 * 2 = 16/32 selon GPUs
NUM_WORKERS = 8         # Plus de workers pour chargement données (utilise RAM)
PIN_MEMORY = True       # Accélère transferts CPU→GPU
PREFETCH_FACTOR = 4     # Précharge les batches en RAM

# FP8 (nécessite transformer_engine de NVIDIA)
USE_FP8 = False  # Passer à True si transformer_engine installé


def setup_distributed():
    """Configure l'environnement distribué pour multi-GPU"""
    if "RANK" in os.environ:
        # Lancement via torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Single node, multi-GPU
        rank = 0
        local_rank = 0
        world_size = NUM_GPUS
        
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    return rank, local_rank, world_size


def get_training_command(use_ddp=True):
    """Génère la commande d'entraînement optimisée"""
    
    if NUM_GPUS > 1 and use_ddp:
        # Multi-GPU avec DistributedDataParallel (recommandé)
        cmd = f"""
torchrun \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node={NUM_GPUS} \\
    scripts/train_subtitles_transformer.py \\
    --resume-from {CHECKPOINT_PATH} \\
    --max-steps {MAX_STEPS} \\
    --output-dir {OUTPUT_DIR} \\
    --batch-size {BATCH_SIZE_PER_GPU} \\
    --gradient-accumulation-steps {GRAD_ACCUMULATION} \\
    --learning-rate 3e-5 \\
    --num-workers {NUM_WORKERS} \\
    --pin-memory \\
    --prefetch-factor {PREFETCH_FACTOR} \\
    --gradient-checkpointing \\
    --use-amp \\
    {"--use-fp8" if USE_FP8 else ""} \\
    --dataset-path {DATASET_PATH}
"""
    elif NUM_GPUS > 1:
        # Multi-GPU avec DataParallel (plus simple mais moins efficace)
        cmd = f"""
python scripts/train_subtitles_transformer.py \\
    --resume-from {CHECKPOINT_PATH} \\
    --max-steps {MAX_STEPS} \\
    --output-dir {OUTPUT_DIR} \\
    --batch-size {BATCH_SIZE_PER_GPU * NUM_GPUS} \\
    --gradient-accumulation-steps {GRAD_ACCUMULATION} \\
    --learning-rate 3e-5 \\
    --num-workers {NUM_WORKERS} \\
    --pin-memory \\
    --prefetch-factor {PREFETCH_FACTOR} \\
    --gradient-checkpointing \\
    --use-amp \\
    --data-parallel \\
    --dataset-path {DATASET_PATH}
"""
    else:
        # Single GPU optimisé
        cmd = f"""
python scripts/train_subtitles_transformer.py \\
    --resume-from {CHECKPOINT_PATH} \\
    --max-steps {MAX_STEPS} \\
    --output-dir {OUTPUT_DIR} \\
    --batch-size {BATCH_SIZE_PER_GPU} \\
    --gradient-accumulation-steps {GRAD_ACCUMULATION} \\
    --learning-rate 3e-5 \\
    --num-workers {NUM_WORKERS} \\
    --pin-memory \\
    --prefetch-factor {PREFETCH_FACTOR} \\
    --gradient-checkpointing \\
    --use-amp \\
    --dataset-path {DATASET_PATH}
"""
    
    return cmd.strip()


def check_system_resources():
    """Affiche les ressources système disponibles"""
    import psutil
    
    print("=" * 80)
    print("🔧 CONFIGURATION SYSTÈME")
    print("=" * 80)
    
    # GPU
    print(f"\n🎮 GPU: {NUM_GPUS} x {torch.cuda.get_device_name(0)}")
    for i in range(NUM_GPUS):
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {mem_gb:.1f} GB VRAM")
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"\n💾 RAM: {ram.total/1e9:.1f} GB total, {ram.available/1e9:.1f} GB disponible")
    
    # CPU
    cpu_count = os.cpu_count()
    print(f"⚡ CPU: {cpu_count} cores")
    
    # Configuration training
    print(f"\n📊 CONFIGURATION TRAINING")
    print(f"   Batch size par GPU: {BATCH_SIZE_PER_GPU}")
    print(f"   Gradient accumulation: {GRAD_ACCUMULATION}")
    total_batch = NUM_GPUS * BATCH_SIZE_PER_GPU * GRAD_ACCUMULATION
    print(f"   Batch effectif total: {total_batch}")
    print(f"   DataLoader workers: {NUM_WORKERS}")
    print(f"   Précision: {'FP8' if USE_FP8 else 'FP16 (AMP)'}")
    
    # Estimations
    vram_per_gpu = BATCH_SIZE_PER_GPU * 1.5  # Approximation
    print(f"\n📈 ESTIMATIONS")
    print(f"   VRAM utilisée par GPU: ~{vram_per_gpu:.1f} GB")
    print(f"   RAM pour DataLoader: ~{NUM_WORKERS * 2} GB")
    print(f"   Speedup multi-GPU théorique: {NUM_GPUS * 0.85:.1f}x")
    
    print("=" * 80)


def main():
    """Point d'entrée principal"""
    
    # Affiche la configuration
    check_system_resources()
    
    # Génère et affiche la commande
    cmd = get_training_command(use_ddp=True)
    
    print("\n🚀 COMMANDE D'ENTRAÎNEMENT")
    print("=" * 80)
    print(cmd)
    print("=" * 80)
    
    # Demande confirmation
    response = input("\n▶️  Lancer l'entraînement ? (y/n): ")
    if response.lower() == 'y':
        print("\n🔥 Démarrage de l'entraînement...\n")
        os.system(cmd)
    else:
        print("\n❌ Annulé. Vous pouvez copier la commande ci-dessus.")


if __name__ == "__main__":
    main()
