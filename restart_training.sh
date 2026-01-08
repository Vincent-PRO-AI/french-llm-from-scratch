#!/bin/bash
# Script de relance de l'entraînement
# À partir du checkpoint 160000, vers 500000 steps

cd /home/vincent/code/repo/french-llm-from-scratch

echo "🚀 RELANCE DE L'ENTRAÎNEMENT"
echo "═════════════════════════════════════════════════════════════════════════════"

source .venv/bin/activate

# Configuration
CHECKPOINT="trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt"
RUN_NAME="french_medium_rtx5080_batch12"
BATCH_SIZE=12
MAX_STEPS=500000
TARGET_LOSS_REDUCTION=0.3  # Viser une réduction de 30% en loss

echo "📊 Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Run name: $RUN_NAME"
echo "  Batch size: $BATCH_SIZE"
echo "  Max steps: $MAX_STEPS"
echo "  ETA: ~28 heures"
echo ""

# Lancer l'entraînement
nohup python3 scripts/train_subtitles_transformer.py \
    --resume-from "$CHECKPOINT" \
    --arch-preset medium \
    --run-name "$RUN_NAME" \
    --batch-size "$BATCH_SIZE" \
    --lr 1.5e-4 \
    --max-steps "$MAX_STEPS" \
    --eval-interval 250 \
    --sample-interval 500 \
    --checkpoint-interval 5000 \
    --metrics-log-fraction 0.02 \
    --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
    --tokenizer-path data_clean/mistral_tokenizer \
    --gradient-accumulation-steps 1 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --auto-resource-adapt \
    > training_resumed_batch12.log 2>&1 &

PID=$!
sleep 2

echo "✅ Entraînement lancé!"
echo "  PID: $PID"
echo ""
echo "📊 Pour suivre la progression:"
echo "  $ tail -f training_resumed_batch12.log"
echo "  $ watch nvidia-smi"
echo "  $ python3 monitor_training_continuous.py 30"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
