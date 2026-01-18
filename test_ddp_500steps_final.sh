#!/bin/bash
# Test 500 steps DDP (RTX 5080 + RTX 4090)

echo "🧪 TEST DDP - 500 STEPS"
echo "----------------------"

# Activer l'environnement Conda french-llm
source /home/vincent/miniconda3/etc/profile.d/conda.sh
conda activate french-llm

export MASTER_ADDR=localhost
export MASTER_PORT=29505
export WORLD_SIZE=2

# Lancement avec torchrun
torchrun \
    --nproc_per_node=2 \
    --master_port=29505 \
    scripts/train_subtitles_transformer_ddp.py \
    --run-name "test_ddp_500steps" \
    --max-steps 250500 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --resume-from trained_models/checkpoint_250k_fixed.pt

echo "----------------------"
echo "✅ Test terminé"
