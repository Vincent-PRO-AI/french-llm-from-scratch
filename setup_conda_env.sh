#!/bin/bash
# 🚀 Setup Environnement Conda pour French LLM
# Étape 1: Créer l'env avec les packages essentiels
# Étape 2: Installer les packages via pip

set -e

PROJECT_ROOT="/home/vincent/code/repo/french-llm-from-scratch"
cd "$PROJECT_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "🚀 SETUP ENVIRONNEMENT CONDA - French LLM"
echo "════════════════════════════════════════════════════════════════"
echo ""

# === ÉTAPE 1: Créer l'env avec PyTorch ===
echo "📦 ÉTAPE 1: Créer environnement Conda avec PyTorch..."
echo "⏳ (Ça prend ~5-10 minutes)"
echo ""

conda create -n french-llm python=3.10 -y --quiet

echo "✅ Environnement créé"
echo ""

# === ÉTAPE 2: Activer et installer PyTorch ===
echo "📦 ÉTAPE 2: Installer PyTorch via Conda..."
echo ""

# Source conda dans un subshell
eval "$(conda shell.bash hook)"
conda activate french-llm

# Installer PyTorch (le plus important)
echo "⏳ Installation PyTorch CUDA 12.1..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y --quiet

echo "✅ PyTorch installé"
echo ""

# === ÉTAPE 3: Installer via pip ===
echo "📦 ÉTAPE 3: Installer dépendances via pip..."
echo "⏳ (Ça prend ~5-10 minutes)"
echo ""

pip install --upgrade pip setuptools wheel > /dev/null 2>&1

pip install \
  transformers>=4.36.0 \
  datasets>=2.19.0 \
  tokenizers>=0.15.0 \
  accelerate>=0.27.0 \
  peft>=0.8.0 \
  bitsandbytes>=0.42.0 \
  sentencepiece>=0.2.0 \
  llama-cpp-python>=0.2.80 \
  fastapi>=0.104.0 \
  uvicorn>=0.24.0 \
  pydantic>=2.0 \
  tensorboard>=2.15.0 \
  wandb>=0.16.0 \
  black>=24.1.0 \
  flake8>=7.0.0 \
  pytest>=7.4.0 \
  numpy pandas scipy scikit-learn \
  tqdm pyyaml requests \
  --quiet

echo "✅ Tous les packages installés"
echo ""

# === ÉTAPE 4: Vérifier installation ===
echo "📦 ÉTAPE 4: Vérification de l'installation..."
echo ""

python3 -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA: {torch.version.cuda}')
" || echo "❌ Erreur vérification PyTorch"

python3 -c "
from transformers import __version__ as transformers_version
from datasets import __version__ as datasets_version
print(f'✅ Transformers: {transformers_version}')
print(f'✅ Datasets: {datasets_version}')
" || echo "❌ Erreur vérification transformers/datasets"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ ENVIRONNEMENT PRÊT!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📌 Activation de l'environnement:"
echo "   conda activate french-llm"
echo ""
echo "📌 Vérifier l'env:"
echo "   conda env list"
echo ""
echo "📌 Vérifier les packages:"
echo "   conda list"
echo "   pip list"
echo ""
echo "════════════════════════════════════════════════════════════════"
