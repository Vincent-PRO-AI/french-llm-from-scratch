#!/bin/bash

# Installation des dépendances pip dans l'environnement Conda

set -e  # Exit on error

ENV_PATH="/home/vincent/miniconda3/envs/french-llm/bin/pip"
PYTHON="/home/vincent/miniconda3/envs/french-llm/bin/python"

echo "🚀 Installation des dépendances Python..."

# Dépendances ML/NLP
$ENV_PATH install -q \
    transformers==4.46.1 \
    datasets==2.21.0 \
    tokenizers==0.20.1 \
    accelerate==1.1.1 \
    peft==0.11.1 \
    bitsandbytes==0.43.3 \
    llama-cpp-python==0.2.85 \
    sentencepiece==0.2.0 \
    numpy==1.26.4 \
    pandas==2.2.0

echo "✅ ML/NLP packages installed"

# API/Serveur
$ENV_PATH install -q \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    flask==3.0.0 \
    flask-cors==4.0.0

echo "✅ API packages installed"

# Monitoring/Logging
$ENV_PATH install -q \
    tensorboard==2.15.1 \
    wandb==0.16.0 \
    tqdm==4.66.1 \
    python-dotenv==1.0.0

echo "✅ Monitoring packages installed"

# Code quality
$ENV_PATH install -q \
    black==23.12.0 \
    flake8==6.1.0 \
    pytest==7.4.3

echo "✅ Code quality packages installed"

# Utils
$ENV_PATH install -q \
    requests==2.31.0 \
    pyyaml==6.0.1 \
    scipy==1.11.4 \
    scikit-learn==1.3.2

echo "✅ Utils packages installed"

echo ""
echo "🎉 Toutes les dépendances sont installées!"

# Vérification
echo ""
echo "📋 Vérification..."
$PYTHON -c "
import torch
import transformers
import datasets
import fastapi
import numpy as np

print(f'✅ torch: {torch.__version__}')
print(f'✅ transformers: {transformers.__version__}')
print(f'✅ datasets: {datasets.__version__}')
print(f'✅ fastapi: {fastapi.__version__}')
print(f'✅ numpy: {np.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
"
