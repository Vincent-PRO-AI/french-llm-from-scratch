#!/bin/bash
# Lancement du Fine-Tuning Mistral-7B via DDP (2 GPUs)
# Setup pour RTX 4090 (24GB) + RTX 5080 (16GB)

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0

# On utilise l'environnement conda french-llm
CONDA_PATH="/home/vincent/miniconda3/bin/python3"
ENV_PYTHON="/home/vincent/miniconda3/envs/french-llm/bin/python3.10"

echo "🚀 Démarrage du Fine-Tuning Mistral-7B v0.3 (QLoRA)"
echo "----------------------------------------------------"
echo "GPUs: RTX 4090 + RTX 5080"
echo "Mode: DDP (Distributed Data Parallel)"
echo "----------------------------------------------------"

# Lancement avec torchrun pour 50 steps de validation
$ENV_PYTHON -m torch.distributed.run --nproc_per_node=2 --master_port=29509 \
    scripts/train_mistral_ddp.py \
    --model-id "mistralai/Mistral-7B-v0.3" \
    --run-name "mistral_7b_french_v2" \
    --max-steps 50 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 0.00005

echo "----------------------------------------------------"
echo "✅ Test de 50 steps terminé."
