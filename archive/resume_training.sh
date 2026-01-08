#!/bin/bash
# Script pour relancer l'entraînement après validation pause

cd /home/vincent/code/repo/french-llm-from-scratch

echo "🚀 RELANCE ENTRAÎNEMENT APRÈS VALIDATION"
echo "========================================"

# Récupérer le checkpoint le plus récent
LATEST_CHECKPOINT=$(ls -t trained_models/runs/french_medium_rtx5080_extended/checkpoint_step_*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ Aucun checkpoint trouvé!"
    exit 1
fi

STEP=$(basename "$LATEST_CHECKPOINT" | sed 's/checkpoint_step_//;s/.pt//')
echo "✅ Checkpoint trouvé: $LATEST_CHECKPOINT"
echo "📊 Steps: $STEP"

# Variables d'environnement
export PYTORCH_ALLOC_CONF="max_split_size_mb:2048"
export TORCH_TF32=1
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=16

# Source venv
source .venv/bin/activate

echo ""
echo "🎯 Relance vers 500,000 steps..."
echo "⏱️  Durée estimée: ~$((($((500000-$STEP))/3.5)/3600)) heures"
echo ""

python3 scripts/train_subtitles_transformer.py \
  --resume-from "$LATEST_CHECKPOINT" \
  --arch-preset "medium" \
  --run-name "french_medium_rtx5080_extended" \
  --batch-size "16" \
  --lr "1.5e-4" \
  --max-steps "500000" \
  --eval-interval "250" \
  --sample-interval "500" \
  --checkpoint-interval "5000" \
  --metrics-log-fraction "0.02" \
  --pretokenized-path "data_clean/conversations_mega_train_mistral_tokenized.pt" \
  --tokenizer-path "data_clean/mistral_tokenizer" \
  --gradient-accumulation-steps "1" \
  --num-workers "8" \
  --prefetch-factor "4" \
  --auto-resource-adapt
