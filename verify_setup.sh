#!/bin/bash

# Script de vérification complète de l'environnement Conda

set -e

ENV_PATH="/home/vincent/miniconda3/envs/french-llm"
PYTHON="$ENV_PATH/bin/python"

echo "════════════════════════════════════════════════════"
echo "✅ VÉRIFICATION ENVIRONNEMENT CONDA - french-llm"
echo "════════════════════════════════════════════════════"
echo ""

# 1. Python version
echo "1️⃣  Python Version:"
$PYTHON --version

# 2. PyTorch + CUDA
echo ""
echo "2️⃣  PyTorch & CUDA:"
$PYTHON -c "
import torch
print(f'   PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 3. ML Libraries
echo ""
echo "3️⃣  ML Libraries:"
$PYTHON -c "
import transformers
import datasets
import peft
print(f'   transformers: {transformers.__version__}')
print(f'   datasets: {datasets.__version__}')
import peft
print(f'   peft: {peft.__version__}')
"

# 4. API Framework
echo ""
echo "4️⃣  API Framework:"
$PYTHON -c "
import fastapi
import flask
print(f'   fastapi: {fastapi.__version__}')
print(f'   flask: {flask.__version__}')
"

# 5. Monitoring
echo ""
echo "5️⃣  Monitoring:"
$PYTHON -c "
try:
    import tensorboard
    print(f'   tensorboard: installed')
except ImportError:
    print(f'   tensorboard: not yet installed')
try:
    import wandb
    print(f'   wandb: installed')
except ImportError:
    print(f'   wandb: not yet installed')
"

# 6. Path verification
echo ""
echo "6️⃣  Paths:"
echo "   Conda env: $ENV_PATH"
echo "   Python exe: $PYTHON"
ls -la $ENV_PATH/bin/python 2>/dev/null && echo "   ✅ Python executable verified"

# 7. Project structure
echo ""
echo "7️⃣  Project Structure:"
echo "   scripts/:"
ls -1 /home/vincent/code/repo/french-llm-from-scratch/scripts/*.py 2>/dev/null | wc -l | xargs echo "      Files:"
echo "   data_clean/:"
du -sh /home/vincent/code/repo/french-llm-from-scratch/data_clean 2>/dev/null | cut -f1 | xargs echo "      Size:"

echo ""
echo "════════════════════════════════════════════════════"
echo "🎉 ENVIRONNEMENT PRÊT POUR L'ENTRAÎNEMENT!"
echo "════════════════════════════════════════════════════"
echo ""
echo "➡️  Commande d'activation:"
echo "    conda activate french-llm"
echo ""
echo "➡️  Vérification rapide:"
echo "    source ~/.bashrc && conda activate french-llm && python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
