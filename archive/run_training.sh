#!/bin/bash

# 🚀 Script de lancement training + TensorBoard
# Usage: bash run_training.sh

set -e

PROJECT_DIR="/home/vincent/code/repo/french-llm-from-scratch"
CONDA_ENV="french-llm"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         🚀 LANCEMENT TRAINING + TENSORBOARD                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_DIR"

# Activer conda
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

echo "✅ Conda activé: $CONDA_ENV"
echo ""

# Vérifier GPU
echo "🔍 Vérification GPU..."
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
print(f'CUDA: {torch.version.cuda}')
"
echo ""

# Lancer training en arrière-plan
echo "🚀 Lancement training en arrière-plan..."
nohup python scripts/02_train_mistral_tensorboard.py \
  --dataset-path ./data_clean/conversations_mega_train_mistral_tokenized.pt \
  --output-path ./trained_models/mistral_v2_tensorboard \
  --batch-size 16 \
  --num-epochs 3 \
  --learning-rate 1e-4 \
  --device cuda \
  --use-bf16 \
  --num-workers 4 > training.log 2>&1 &

TRAIN_PID=$!
echo "✅ Training PID: $TRAIN_PID"
sleep 2

# Lancer TensorBoard en arrière-plan
echo "📊 Lancement TensorBoard..."
nohup tensorboard --logdir=./trained_models/mistral_v2_tensorboard/logs --port=6006 --bind_all > tensorboard.log 2>&1 &

TB_PID=$!
echo "✅ TensorBoard PID: $TB_PID"
sleep 2

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   ✅ TOUT EST LANCÉ!                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 TensorBoard: http://localhost:6006"
echo "📋 Logs: tail -f training.log"
echo "📈 GPU: nvidia-smi -l 1"
echo ""
echo "PIDs:"
echo "  Training: $TRAIN_PID"
echo "  TensorBoard: $TB_PID"
echo ""
