#!/bin/bash
# Lancement DDP optimisé pour RTX 5080 (16GB) + RTX 4090 (24GB)

echo "🚀 Démarrage training DDP sur 2 GPUs"
echo "📊 Configuration:"
echo "  - GPU 0 (RTX 5080 16GB): batch_size=8"
echo "  - GPU 1 (RTX 4090 24GB): batch_size=10"
echo "  - Batch effectif total: 18 × 2 (accumulation) = 36"
echo ""

# Activer environnement
source .venv/bin/activate

# Variables DDP
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2

# Lancement
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train_subtitles_transformer_ddp.py \
    --max-steps 300000 \
    --batch-size 9 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-4 \
    --warmup-steps 1000 \
    --resume-from trained_models/checkpoint_250k_fixed.pt

echo ""
echo "✅ Training terminé ou interrompu"
