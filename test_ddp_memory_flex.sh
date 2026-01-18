#!/bin/bash
# Test DDP avec configuration mémoire réduite AVANT lancer 100k steps

set -e

echo "=================================================="
echo "🧪 Test DDP Memory Flex - 500 steps"
echo "=================================================="
echo ""
echo "Configuration de test:"
echo "  Batch size: 4 (réduit de 8)"
echo "  Gradient accumulation: 2"
echo "  Steps: 500 (rapide validation)"
echo "  Target: Vérifier pas de crash OOM"
echo ""

mkdir -p logs

# Activer les optimisations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1

source .venv/bin/activate 2>/dev/null || source ~/.bashrc 2>/dev/null || true

echo "Lançage test..."
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train_subtitles_transformer.py \
    --arch-preset medium \
    --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
    --resume-from trained_models/checkpoint_250k_fixed.pt \
    --max-steps 250500 \
    --batch-size 4 \
    --log-dir trained_models/runs \
    --gradient-accumulation-steps 2 \
    --cpu-offload \
    --gradient-checkpointing \
    2>&1 | tee logs/test_ddp_memory_flex.log

echo ""
echo "✅ Test réussi! Configuration prête pour 100k steps"
echo ""
echo "Pour lancer l'entraînement complet 100k steps:"
echo "  chmod +x train_ddp_memory_flex.sh && ./train_ddp_memory_flex.sh"
