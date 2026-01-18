#!/bin/bash
set -e

echo "🧪 TEST SINGLE GPU - 200 STEPS (validating code)"
echo "=================================================="

source .venv/bin/activate
mkdir -p logs

LOG="logs/test_single_gpu_final.log"

echo "📊 Configuration:"
echo "  • GPU: RTX 5080"
echo "  • From: 250,000 steps"
echo "  • To: 250,200 steps"
echo "  • Mode: SINGLE GPU (no DDP)"
echo ""

python scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
    --resume-from trained_models/checkpoint_250k_fixed.pt \
    --max-steps 250200 \
    --batch-size 8 \
    --log-dir trained_models/runs \
    2>&1 | tee "$LOG"

echo ""
echo "✅ Single GPU test terminé!"
tail -10 "$LOG" | grep -E "step|loss" || echo "Check: $LOG"
