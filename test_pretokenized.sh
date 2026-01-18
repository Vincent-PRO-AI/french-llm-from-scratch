#!/bin/bash
# Test rapide avec données pré-tokenisées

echo "🧪 TEST RAPIDE - Données pré-tokenisées"
echo "========================================"

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1  # RTX 4090
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

python scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
    --max-steps 10 \
    --batch-size 8 \
    --eval-interval 5 \
    --checkpoint-interval 10 \
    2>&1 | tee logs/test_pretokenized.log

echo ""
echo "✅ Test terminé"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

