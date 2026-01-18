#!/bin/bash
# Test rapide single GPU - 50 steps (~5 min)

echo "🧪 TEST SINGLE GPU - 50 steps rapides"
echo "======================================"
echo ""

source .venv/bin/activate

export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=1  # RTX 4090 (24GB)

echo "📊 Configuration:"
echo "  • GPU 1: RTX 4090 (24GB)"
echo "  • Steps: 250,000 → 250,050"
echo "  • Batch: 8 × 2 = 16 effectif"
echo "  • Durée: ~5-7 minutes"
echo ""
echo "🚀 Démarrage..."
echo ""

python scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --resume-from trained_models/checkpoint_250k_fixed.pt \
    --max-steps 250050 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --lr 1e-4 \
    2>&1 | tee logs/test_single_50steps.log

EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test réussi"
    echo "📈 Prochaine étape: DDP sur 2 GPUs"
else
    echo "⚠️  Erreur (exit code: $EXIT_CODE)"
    echo "📝 Logs: logs/test_single_50steps.log"
fi

echo ""
echo "🎮 État GPU:"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

