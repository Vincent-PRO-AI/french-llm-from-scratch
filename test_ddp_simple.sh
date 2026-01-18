#!/bin/bash
# Test DDP simple - 200 steps avec le script natif

echo "🧪 TEST DDP - 200 steps (script natif train_subtitles_transformer.py)"
echo "========================================================================"
echo ""

# Activer environnement
source .venv/bin/activate

# Configuration
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

echo "📊 Configuration:"
echo "  • 2 GPUs (RTX 5080 + RTX 4090)"
echo "  • 250,000 → 250,200 steps (200 nouveaux)"
echo "  • Batch effectif: ~32"
echo ""
echo "🚀 Démarrage training DDP..."
echo ""

# Lancement avec torchrun (remplace torch.distributed.launch)
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train_subtitles_transformer.py \
    --resume-from trained_models/checkpoint_250k_fixed.pt \
    --max-steps 250200 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-4 \
    2>&1 | tee logs/test_ddp_simple.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test terminé avec succès"
else
    echo "⚠️  Test interrompu (exit code: $EXIT_CODE)"
fi

echo "🎮 GPUs après training:"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

