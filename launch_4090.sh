#!/bin/bash
# Simple launcher pour training 4090

cd /home/vincent/code/repo/french-llm-from-scratch

# Activer conda
export PATH="/home/vincent/miniconda3/bin:$PATH"
source activate french-llm

# Lancer le training
python train_4090_optimized.py
