#!/bin/bash
# 🚀 Script de démarrage V2 Training - French LLM
# RTX 5080 + R7 9700X OC 105W

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "================================"
echo "🚀 FRENCH LLM V2 TRAINING LAUNCHER"
echo "================================"
echo ""
echo "🖥️  Hardware Detected:"
echo "   GPU: RTX 5080 16GB"
echo "   CPU: R7 9700X OC 105W"
echo "   RAM: 64GB DDR5"
echo "   Storage: NVMe PCIe 4.0"
echo ""

# Vérifier Docker
if command -v docker &> /dev/null; then
    echo "✓ Docker trouvé"
    DOCKER_AVAILABLE=1
else
    echo "✗ Docker non trouvé"
    DOCKER_AVAILABLE=0
fi

# Vérifier Python
if command -v python3 &> /dev/null; then
    echo "✓ Python3 trouvé: $(python3 --version)"
    PYTHON_CMD="python3"
else
    echo "✗ Python3 non trouvé"
    PYTHON_CMD="python"
fi

# Chercher conda
if command -v conda &> /dev/null; then
    echo "✓ Conda trouvé"
    CONDA_AVAILABLE=1
else
    echo "✗ Conda non trouvé"
    CONDA_AVAILABLE=0
fi

echo ""
echo "================================"
echo "📂 Vérification des données..."
echo "================================"

if [ -d "data_clean/v2_tokenized" ]; then
    SIZE=$(du -sh data_clean/v2_tokenized | cut -f1)
    echo "✓ v2_tokenized: $SIZE"
else
    echo "✗ Données V2 manquantes"
fi

if [ -f "data_clean/conversations_mega_train.txt" ]; then
    SIZE=$(du -sh data_clean/conversations_mega_train.txt | cut -f1)
    echo "✓ conversations_mega_train.txt: $SIZE"
fi

if [ -f "data_clean/conversations_mega_test.txt" ]; then
    SIZE=$(du -sh data_clean/conversations_mega_test.txt | cut -f1)
    echo "✓ conversations_mega_test.txt: $SIZE"
fi

echo ""
echo "================================"
echo "🎯 Options de démarrage:"
echo "================================"
echo ""
echo "1. Lancer avec Docker (recommandé)"
echo "2. Lancer avec Python system"
echo "3. Lancer le Dashboard PyTorch seul"
echo "4. Vérifier configuration matériel"
echo "5. Quitter"
echo ""

read -p "Choix (1-5): " CHOICE

case $CHOICE in
    1)
        if [ $DOCKER_AVAILABLE -eq 1 ]; then
            echo ""
            echo "🐳 Lancement avec Docker..."
            echo ""
            docker-compose up -d
            echo ""
            echo "✓ Conteneurs démarrés!"
            echo "📊 Dashboard: http://localhost:5000"
            echo "🔥 Training logs: docker-compose logs -f"
        else
            echo "✗ Docker non disponible"
            exit 1
        fi
        ;;
    2)
        echo ""
        echo "🐍 Lancement avec Python system..."
        echo ""
        
        # Vérifier les dépendances
        if $PYTHON_CMD -c "import torch" 2>/dev/null; then
            echo "✓ PyTorch disponible"
            
            # Lancer le launcher
            $PYTHON_CMD launch_v2_training_optimized.py
        else
            echo "✗ PyTorch non installé"
            echo ""
            echo "Installation requise:"
            echo "  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
            echo "  ou"
            echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            exit 1
        fi
        ;;
    3)
        echo ""
        echo "🎛️  Lancement Dashboard PyTorch..."
        echo ""
        
        if $PYTHON_CMD -c "import flask" 2>/dev/null; then
            $PYTHON_CMD dashboard_pytorch.py
        else
            echo "✗ Flask non installé"
            echo "Installation: pip install flask"
            exit 1
        fi
        ;;
    4)
        echo ""
        echo "🔍 Vérification configuration..."
        echo ""
        $PYTHON_CMD launch_optimized_v2_training.py 2>/dev/null || true
        ;;
    5)
        echo "Au revoir!"
        exit 0
        ;;
    *)
        echo "Choix invalide"
        exit 1
        ;;
esac
