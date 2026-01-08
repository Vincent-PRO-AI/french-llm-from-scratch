#!/bin/bash

# 🚀 COPY-PASTE READY - Commande pour lancer l'entraînement Mistral

set -e

echo "═══════════════════════════════════════════════════════════════════"
echo "🧠 PHASE 3 - LANCER L'ENTRAÎNEMENT MISTRAL"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Step 1: Activate conda
echo "Step 1: Activation Conda"
echo "────────────────────────"
eval "$(conda shell.bash hook)"
conda activate french-llm
echo "✅ Conda activated: french-llm"
echo ""

# Step 2: Go to project
echo "Step 2: Accéder au répertoire du projet"
echo "──────────────────────────────────────"
cd /home/vincent/code/repo/french-llm-from-scratch
echo "✅ Working directory: $(pwd)"
echo ""

# Step 3: Run training
echo "Step 3: Lancer l'entraînement Mistral"
echo "──────────────────────────────────────"
echo ""
echo "📝 Commande:"
echo ""
echo "python scripts/02_train_mistral_fixed.py \\"
echo "  --dataset-path ./data_clean/conversations_mega_train_mistral_tokenized.pt \\"
echo "  --output-path ./trained_models/french_medium_v2 \\"
echo "  --batch-size 16 \\"
echo "  --num-epochs 3 \\"
echo "  --use-bf16 \\"
echo "  --device cuda"
echo ""
echo "─" * 60
echo ""

python scripts/02_train_mistral_fixed.py \
  --dataset-path ./data_clean/conversations_mega_train_mistral_tokenized.pt \
  --output-path ./trained_models/french_medium_v2 \
  --batch-size 16 \
  --num-epochs 3 \
  --use-bf16 \
  --device cuda

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "✅ ENTRAÎNEMENT TERMINÉ!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Modèle sauvegardé dans: ./trained_models/french_medium_v2/final_model"
echo ""
echo "Prochaine étape: Export GGUF pour LM Studio"
echo ""
echo "python scripts/03_export_gguf.py \\"
echo "  --model-path ./trained_models/french_medium_v2/final_model \\"
echo "  --output-dir ./models_gguf"
echo ""
