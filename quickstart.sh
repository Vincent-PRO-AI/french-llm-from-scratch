#!/bin/bash
# 🚀 QUICKSTART - French LLM Mistral Training Pipeline
# PHASE 2-4: Data Processing → Training → Export GGUF

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "🚀 FRENCH LLM - MISTRAL TRAINING PIPELINE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# ÉTAPE 1: Vérifier et configurer l'environnement
# ============================================================================

echo "📌 ÉTAPE 1: Vérifier environnement..."
echo ""

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found!"
    exit 1
fi

echo "✅ Python: $(python3 --version)"

# Créer venv si n'existe pas
if [ ! -d ".venv" ]; then
    echo "📦 Créer environnement virtuel..."
    python3 -m venv .venv
fi

# Activer venv
source .venv/bin/activate
echo "✅ Environnement activé"

# ============================================================================
# ÉTAPE 2: Installer dépendances
# ============================================================================

echo ""
echo "📌 ÉTAPE 2: Installer dépendances..."
echo ""

if [ -f "requirements.txt" ]; then
    echo "📥 Installer requirements.txt..."
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt --quiet
    echo "✅ Dépendances installées"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# ============================================================================
# ÉTAPE 3: Vérifier GPU
# ============================================================================

echo ""
echo "📌 ÉTAPE 3: Vérifier GPU..."
echo ""

python3 << 'PYTHON_EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  WARNING: No GPU available. Training will be very slow.")
PYTHON_EOF

# ============================================================================
# ÉTAPE 4: Vérifier import transformers
# ============================================================================

echo ""
echo "📌 ÉTAPE 4: Vérifier dépendances clés..."
echo ""

python3 << 'PYTHON_EOF'
try:
    from transformers import MistralConfig, MistralForCausalLM, AutoTokenizer
    print("✅ Transformers (Mistral support)")
except ImportError as e:
    print(f"❌ Transformers error: {e}")
    exit(1)

try:
    from datasets import load_dataset, load_from_disk
    print("✅ Datasets")
except ImportError as e:
    print(f"❌ Datasets error: {e}")
    exit(1)

try:
    import accelerate
    print("✅ Accelerate")
except ImportError as e:
    print(f"❌ Accelerate error: {e}")
    exit(1)

print("\n✅ ALL DEPENDENCIES OK!")
PYTHON_EOF

if [ $? -ne 0 ]; then
    echo "❌ Dependencies check failed!"
    exit 1
fi

# ============================================================================
# ÉTAPE 5: Afficher informations de configuration
# ============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "📋 CONFIGURATION"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Architecture Mistral Tiny:"
echo "  • Hidden Size: 1024"
echo "  • Num Layers: 12"
echo "  • Num Heads: 8"
echo "  • Vocab Size: 32000"
echo "  • Context Window: 2048"
echo "  • Parameters: ~125M"
echo ""
echo "Optimisations:"
echo "  • Precision: BF16"
echo "  • Gradient Checkpointing: Yes"
echo "  • torch.compile: Optional"
echo ""
echo "Estimations de temps:"
echo "  • Phase 2 (Data): ~30-60 minutes"
echo "  • Phase 3 (Training 80k steps): ~12-16 heures"
echo "  • Phase 4 (Export GGUF): ~10 minutes"
echo "  • Total: ~14-18 heures"
echo ""

# ============================================================================
# ÉTAPE 6: Préparer répertoires
# ============================================================================

echo "📌 ÉTAPE 6: Préparer répertoires..."
echo ""

mkdir -p data/processed_mistral
mkdir -p models/mistral_tiny
mkdir -p checkpoints/mistral_tiny
mkdir -p logs

echo "✅ Répertoires créés"

# ============================================================================
# ÉTAPE 7: Prêt à démarrer!
# ============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "✅ ENVIRONNEMENT PRÊT!"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "🚀 PROCHAINES ÉTAPES:"
echo ""
echo "1️⃣  PHASE 2 - Data Processing (Data Pipeline):"
echo "   python3 scripts/01_process_data.py \\"
echo "     --dataset wikitext \\"
echo "     --output-path ./data/processed_mistral"
echo ""
echo "2️⃣  PHASE 3 - Training (Mistral Model):"
echo "   python3 scripts/02_train_mistral.py \\"
echo "     --batch-size 8 \\"
echo "     --num-epochs 1 \\"
echo "     --dataset-path ./data/processed_mistral \\"
echo "     --output-path ./models/mistral_tiny"
echo ""
echo "3️⃣  PHASE 4 - Export GGUF (LM Studio Ready):"
echo "   python3 scripts/03_export_gguf.py \\"
echo "     --model-path ./models/mistral_tiny \\"
echo "     --output-path ./models/mistral_tiny.gguf \\"
echo "     --quantize Q4_K_M"
echo ""
echo "4️⃣  Test dans LM Studio:"
echo "   • Ouvrir LM Studio"
echo "   • Charger: models/mistral_tiny.gguf"
echo "   • Générer du texte!"
echo ""
echo "📚 Documentation:"
echo "   • PLAN_ACTION.md         - Plan complet"
echo "   • PHASE1_COMPLETED.md    - Résumé setup"
echo "   • scripts/*.py --help    - Aide détaillée"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
