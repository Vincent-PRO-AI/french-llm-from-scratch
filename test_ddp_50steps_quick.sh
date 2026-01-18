#!/bin/bash
set -e

echo "🚀 TEST RAPIDE DDP - 50 STEPS sur 2 GPUs"
echo "=========================================="

source .venv/bin/activate

# Logs
mkdir -p logs
LOG="logs/test_ddp_50steps_quick.log"

echo "📊 Configuration:"
echo "  • Checkpoint: 250,000 steps"
echo "  • Target: 50 steps (test rapide)"
echo "  • Durée estimée: ~1-2 minutes"
echo ""

echo "🏁 Démarrage DDP..."
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
    --resume-from trained_models/checkpoint_250k_fixed.pt \
    --max-steps 250050 \
    --batch-size 8 \
    --log-dir trained_models/runs \
    2>&1 | tee "$LOG"

echo ""
echo "✅ Test DDP 50 steps terminé!"
tail -20 "$LOG" | grep -E "step|loss|⚡" || echo "Check log: $LOG"
