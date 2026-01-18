#!/bin/bash
# Test DDP rapide - 200 steps (~20 minutes)

echo "🧪 TEST DDP - 200 steps sur 2 GPUs"
echo "======================================"
echo ""
echo "📊 Configuration test:"
echo "  • RTX 5080 (GPU 0): batch_size=8"
echo "  • RTX 4090 (GPU 1): batch_size=8" 
echo "  • Steps: 200 (depuis step 250k)"
echo "  • Durée estimée: ~15-20 min"
echo ""

# Activer environnement
source .venv/bin/activate

# Variables DDP
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Calculer step final (250000 + 200)
RESUME_STEP=250000
TEST_STEPS=200
MAX_STEPS=$((RESUME_STEP + TEST_STEPS))

echo "🚀 Démarrage training..."
echo "  Resume from: step $RESUME_STEP"
echo "  Target: step $MAX_STEPS"
echo ""

# Lancement DDP
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train_subtitles_transformer_ddp.py \
    --max-steps $MAX_STEPS \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-4 \
    --warmup-steps 0 \
    --resume-from trained_models/checkpoint_250k_fixed.pt \
    2>&1 | tee logs/test_ddp_200steps.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test terminé avec succès"
else
    echo "⚠️  Test interrompu (exit code: $EXIT_CODE)"
fi
echo "📝 Logs: logs/test_ddp_200steps.log"
echo ""

# Afficher utilisation GPU finale
echo "🎮 État final des GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

