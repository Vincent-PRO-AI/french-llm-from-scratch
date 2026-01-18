#!/bin/bash
# Script pour lancer l'entraînement DDP avec gestion mémoire flexible
# Utilise VRAM → RAM → NVMe fallback automatique

set -e

# Configuration
PRETOKENIZED_DATA="data_clean/conversations_mega_train_mistral_tokenized.pt"
CHECKPOINT="trained_models/checkpoint_250k_fixed.pt"
MAX_STEPS=350000
TARGET_STEPS=100000  # 100k steps = ~8-10h
LOG_FILE="logs/train_ddp_memory_flex.log"
BATCH_SIZE=4  # Réduit de 8 à 4 pour éviter OOM

# Créer le répertoire logs
mkdir -p logs

echo "=================================================="
echo "🚀 Démarrage entraînement DDP avec gestion mémoire flexible"
echo "=================================================="
echo "Configuration:"
echo "  Dataset: $PRETOKENIZED_DATA"
echo "  Checkpoint: $CHECKPOINT"
echo "  Max steps: $MAX_STEPS (incremental from 250k)"
echo "  Batch size: $BATCH_SIZE (per GPU)"
echo "  Target duration: 8-10 hours"
echo "  Log: $LOG_FILE"
echo ""
echo "Optimisations mémoire:"
echo "  ✓ Batch size réduit: 8 → 4 (moins de pression VRAM)"
echo "  ✓ PYTORCH_CUDA_ALLOC_CONF: expandable_segments"
echo "  ✓ CPU offload: Auto (si VRAM saturée)"
echo "  ✓ NVMe cache: Auto (/nvme_cache if available)"
echo "  ✓ Monitoring: En temps réel avec rapports mémoire"
echo "=================================================="
echo ""

# Activer les optimisations PyTorch
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1

# Conda activation (if needed)
source ~/.bashrc 2>/dev/null || true

# Activer l'environnement .venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Lancer le training avec DDP
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --pretokenized-path "$PRETOKENIZED_DATA" \
    --resume-from "$CHECKPOINT" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --log-dir "trained_models/runs" \
    --gradient-accumulation-steps 2 \
    --cpu-offload \
    --gradient-checkpointing \
    --checkpoint-interval 2500 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ Entraînement terminé!"
echo "   Logs: $LOG_FILE"
echo "   Checkpoint: trained_models/checkpoint_*.pt"
