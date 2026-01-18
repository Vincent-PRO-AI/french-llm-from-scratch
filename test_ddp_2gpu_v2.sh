#!/bin/bash
# Test DDP 200 steps avec 2 GPUs - Version 2 avec wrapper

echo "🚀 TEST DDP v2 - 200 STEPS sur 2 GPUs"
echo "========================================"
echo ""

source .venv/bin/activate

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

echo "📊 Configuration DDP:"
echo "  • GPU 0: RTX 5080 (16GB)"
echo "  • GPU 1: RTX 4090 (24GB)"
echo "  • Checkpoint: 250,000 steps"
echo "  • Target: 250,200 steps"
echo "  • Batch par GPU: 8"
echo "  • Batch effectif: 16 × 2 = 32"
echo "  • Durée estimée: ~6-8 minutes (1.7× speedup)"
echo ""
echo "🏁 Démarrage DDP avec wrapper..."
echo ""

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train_ddp_wrapper.py \
    --arch-preset medium \
    --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
    --resume-from trained_models/checkpoint_250k_fixed.pt \
    --max-steps 250200 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --lr 1e-4 \
    --eval-interval 50 \
    --checkpoint-interval 200 \
    2>&1 | tee logs/test_ddp_2gpu_v2.log

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST DDP RÉUSSI"
else
    echo "⚠️  Erreur (exit code: $EXIT_CODE)"
    echo "📝 Vérifier: logs/test_ddp_2gpu_v2.log"
fi

echo ""
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu --format=csv

