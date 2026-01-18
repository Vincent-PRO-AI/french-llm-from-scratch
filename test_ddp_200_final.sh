#!/bin/bash
set -e

echo "🚀 TEST DDP DUAL GPU - 200 STEPS"
echo "======================================"

source .venv/bin/activate
mkdir -p logs

LOG="logs/test_ddp_200_final.log"

echo "📊 Configuration DDP:"
echo "  • GPU 0: RTX 5080 (16GB)"
echo "  • GPU 1: RTX 4090 (24GB)"  
echo "  • Checkpoint: 250,000 steps"
echo "  • Target: 250,200 steps"
echo "  • Mode: DISTRIBUTED (DDP)"
echo "  • Durée estimée: ~2 minutes (1.7× speedup)"
echo ""

echo "🏁 Démarrage..."
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
    --resume-from trained_models/checkpoint_250k_fixed.pt \
    --max-steps 250200 \
    --batch-size 8 \
    --log-dir trained_models/runs \
    2>&1 | tee "$LOG"

echo ""
echo "✅ Test DDP terminé!"
tail -15 "$LOG" | grep -E "step|loss|DDP|rank" || echo "Check: $LOG"
