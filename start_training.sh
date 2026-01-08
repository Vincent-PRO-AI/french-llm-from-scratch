#!/bin/bash
# Wrapper robuste pour l'entraînement
cd /home/vincent/code/repo/french-llm-from-scratch

# Source venv
source .venv/bin/activate

# Variables d'environnement optimisées
export PYTORCH_ALLOC_CONF="max_split_size_mb:2048"
export TORCH_TF32=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

# Lancer l'entraînement
python3 scripts/train_subtitles_transformer.py \
  --resume-from "trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt" \
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
