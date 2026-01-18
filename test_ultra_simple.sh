#!/bin/bash
# Test ultra simple - 10 steps, sans resume

echo "🧪 TEST ULTRA SIMPLE - 10 steps from scratch"
echo "=============================================="

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

python scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --max-steps 10 \
    --batch-size 4 \
    --eval-interval 5

