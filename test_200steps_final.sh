#!/bin/bash
# Test 200 steps avec checkpoint 250k - Single GPU

echo "🚀 TEST 200 STEPS - RTX 4090"
echo "================================"
echo ""

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1  # RTX 4090 (24GB)
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

echo "📊 Configuration:"
echo "  • GPU: RTX 4090 (24GB)"
echo "  • Checkpoint: 250,000 steps"
echo "  • Target: 250,200 steps (200 nouveaux)"
echo "  • Batch: 8 × 2 = 16 effectif"
echo "  • Données: Pré-tokenisées (85M tokens)"
echo "  • Durée estimée: ~15-20 minutes"
echo ""
echo "🏁 Démarrage..."
echo ""

python scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
    --resume-from trained_models/checkpoint_250k_fixed.pt \
    --max-steps 250200 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --lr 1e-4 \
    --eval-interval 50 \
    --checkpoint-interval 200 \
    2>&1 | tee logs/test_200steps_final.log

EXIT_CODE=$?

echo ""
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST RÉUSSI - 200 steps complétés"
    echo "📈 Checkpoint final: trained_models/runs/.../checkpoint_step_250200.pt"
else
    echo "⚠️  Erreur (exit code: $EXIT_CODE)"
fi

echo ""
echo "📊 État final GPU:"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu --format=csv

echo ""
echo "📝 Logs: logs/test_200steps_final.log"

