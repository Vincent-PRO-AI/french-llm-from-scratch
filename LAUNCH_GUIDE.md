#!/bin/bash

# 🚀 Guide rapide pour lancer l'entraînement

set -e

CONDA_ENV="french-llm"
WORKSPACE="/home/vincent/code/repo/french-llm-from-scratch"

echo "════════════════════════════════════════════════"
echo "🚀 FRENCH LLM - GUIDE DE LANCEMENT"
echo "════════════════════════════════════════════════"
echo ""

# Afficher le statut
echo "📊 Statut du projet:"
echo "   GPU: RTX 5080 (17.1GB VRAM)"
echo "   Python: 3.10.19"
echo "   PyTorch: 2.6.0 + CUDA 12.4"
echo "   Données: 28GB (pre-tokenized Mistral)"
echo ""

echo "════════════════════════════════════════════════"
echo "🎯 ÉTAPES"
echo "════════════════════════════════════════════════"
echo ""

# Option 1
echo "Option 1️⃣  LANCER PHASE 3 (Entraînement Mistral)"
echo "   Durée: ~12-16 heures"
echo "   Commande:"
echo "   ────────────────────"
echo "   conda activate $CONDA_ENV"
echo "   cd $WORKSPACE"
echo "   python scripts/02_train_mistral.py \\"
echo "     --data-path ./data_clean/conversations_mega_train_mistral_tokenized.pt \\"
echo "     --output-dir ./trained_models/french_medium_v2 \\"
echo "     --max-steps 100000"
echo ""

# Option 2
echo "Option 2️⃣  LANCER PHASE 2 (Traitement Données - si données brutes)"
echo "   Durée: ~2-4 heures"
echo "   Commande:"
echo "   ────────────────────"
echo "   conda activate $CONDA_ENV"
echo "   cd $WORKSPACE"
echo "   python scripts/01_process_data.py \\"
echo "     --dataset fineweb \\"
echo "     --output-path ./data_clean/processed_mistral"
echo ""

# Option 3
echo "Option 3️⃣  LANCER PHASE 4 (Export GGUF - après entraînement)"
echo "   Durée: ~10 minutes"
echo "   Commande:"
echo "   ────────────────────"
echo "   conda activate $CONDA_ENV"
echo "   cd $WORKSPACE"
echo "   python scripts/03_export_gguf.py \\"
echo "     --model-path ./trained_models/french_medium_v2/checkpoint_final \\"
echo "     --output-dir ./models_gguf"
echo ""

echo "════════════════════════════════════════════════"
echo "✅ PRÉREQUIS COMPLÉTÉS"
echo "════════════════════════════════════════════════"
echo "✅ Miniconda: installé"
echo "✅ Conda env: créé (french-llm)"
echo "✅ PyTorch: 2.6.0 + CUDA 12.4"
echo "✅ GPU: RTX 5080 détecté et testé"
echo "✅ Dépendances: transformers, datasets, accelerate, etc."
echo ""

echo "════════════════════════════════════════════════"
echo "💡 CONSEILS"
echo "════════════════════════════════════════════════"
echo "1. Ferme Chrome avant de lancer (libère ~2-4GB RAM)"
echo "2. Si entrainement long: utilise screen ou tmux"
echo "3. Monitoring GPU: nvidia-smi -l 1"
echo "4. Monitoring entrainement: tail -f metrics.jsonl"
echo ""

echo "════════════════════════════════════════════════"
echo "🎯 CHOIX RECOMMANDÉ"
echo "════════════════════════════════════════════════"
echo ""
echo "→ SI TU VEUX COMMENCER IMMÉDIATEMENT:"
echo "  Ferme Chrome + lance PHASE 3 (entraînement)"
echo ""
echo "→ SI TU VEUX DES DONNÉES NOUVELLES:"
echo "  Lance PHASE 2 d'abord (traitement données)"
echo ""
echo "→ SI TU AS UN MODÈLE ENTRAÎNÉ:"
echo "  Lance PHASE 4 (export GGUF pour LM Studio)"
echo ""
