#!/bin/bash
set -e

echo "🚀 TRAINING OVERNIGHT - 100K STEPS DDP"
echo "========================================"

source .venv/bin/activate
mkdir -p logs

LOG="logs/train_ddp_100k.log"

echo "📊 Configuration:"
echo "  • GPU 0: RTX 5080 (16GB)"
echo "  • GPU 1: RTX 4090 (24GB)"
echo "  • Checkpoint: 250,000 → 350,000 steps"
echo "  • Steps: +100,000"
echo "  • Mode: DDP (1.7× speedup)"
echo "  • Durée estimée: 8-10 heures"
echo ""

echo "🏁 Démarrage..."
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
    --resume-from trained_models/checkpoint_250k_fixed.pt \
    --max-steps 350000 \
    --batch-size 8 \
    --log-dir trained_models/runs \
    2>&1 | tee "$LOG"

echo ""
echo "✅ Training 100k steps terminé!"
