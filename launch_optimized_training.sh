#!/bin/bash
# 🚀 Lance l'entraînement avec optimisations complètes
# OPTIMISATIONS:
# - Batch size augmenté: 8 (au lieu de 4)
# - Gradient accumulation: 2 (batch effectif = 16)
# - DataLoader workers: 8 (utilise RAM pour précalculer batches)
# - Pin memory: activé (accélère CPU→GPU)
# - Prefetch factor: 4 (précharge 4 batches)
# - Gradient checkpointing: activé (économise VRAM)
# - AMP: activé (mixed precision)

set -e

cd "$(dirname "$0")"

echo "🚀 Optimisation Performance Complète"
echo "======================================"
echo ""
echo "✅ Configuration:"
echo "   - Batch size: 8 (au lieu de 4)"
echo "   - Gradient accumulation: 2"
echo "   - Batch effectif: 16 (au lieu de 12)"
echo "   - DataLoader workers: 8"
echo "   - Pin memory: yes"
echo "   - Prefetch factor: 4"
echo "   - Gradient checkpointing: yes"
echo "   - AMP (FP16): yes"
echo ""
echo "🎯 Cible:"
echo "   - Checkpoint: 160000 → 220000 (+60k steps)"
echo "   - Dataset: french_large_filtered (530M tokens)"
echo ""

# Activer le virtualenv
source /home/vincent/.venv/bin/activate || conda activate french-llm

# Variables d'optimisation GPU/CPU
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0

# Launch training
python scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --resume-from trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_160000.pt \
    --max-steps 220000 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --gradient-checkpointing \
    --lr 3e-5 \
    --tokenizer-path data_clean/mistral_tokenizer \
    --pretokenized-path data_clean/french_large_filtered_mistral_tokenized.pt \
    --run-name french_medium_optimized_intensive \
    --checkpoint-interval 2500 \
    --eval-interval 1000 \
    --eval-batches 16 \
    --sample-interval 500 \
    --sample-max-new-tokens 120 \
    --nvme-cache-path /dev/shm/nvme_cache \
    "$@"
