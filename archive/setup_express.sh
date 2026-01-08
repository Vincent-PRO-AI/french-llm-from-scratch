#!/bin/bash

# Setup ULTRA-RAPIDE: PyTorch + dépendances essentielles
# Destiné à remplacer setup_conda_env.sh si trop lent

set -e

CONDA_ENV="french-llm"
ENV_BIN="/home/vincent/miniconda3/envs/$CONDA_ENV/bin"

echo "🚀 SETUP EXPRESS - French LLM Training"
echo "======================================"
echo ""

# 1. Vérifier que l'env existe
if [ ! -d "/home/vincent/miniconda3/envs/$CONDA_ENV" ]; then
    echo "❌ Erreur: environnement $CONDA_ENV n'existe pas!"
    exit 1
fi

echo "✅ Environnement détecté: $CONDA_ENV"
echo ""

# 2. Installer PyTorch uniquement (plus critique)
echo "📦 Installation PyTorch (peut prendre 5-10 min)..."
echo "   Cette étape est critique pour le GPU"
echo ""

$ENV_BIN/pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --quiet

echo "✅ PyTorch installé"
echo ""

# 3. Vérifier PyTorch
echo "🔍 Vérification PyTorch..."
$ENV_BIN/python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA version: {torch.version.cuda}')
"

echo ""
echo "🎉 SETUP EXPRESSS TERMINÉ!"
echo ""
echo "➡️  Prochaines étapes:"
echo "   1. Ferme Chrome pour libérer RAM"
echo "   2. Exécute les scripts Phase 2/3/4:"
echo "      conda activate french-llm"
echo "      python scripts/01_process_data.py"
echo ""
