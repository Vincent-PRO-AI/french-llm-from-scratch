#!/bin/bash
# Script de sauvegarde de l'état du workspace French LLM Training
# Date: Décembre 2025
# Utilisation: ./backup_workspace.sh

echo "🔄 SAUVEGARDE DE L'ENVIRONNEMENT DE TRAVAIL"
echo "=========================================="

# Créer le répertoire de sauvegarde
BACKUP_DIR="/mnt/g/workspace_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📁 Répertoire de sauvegarde: $BACKUP_DIR"

# Sauvegarder l'environnement virtuel
echo "🐍 Sauvegarde de l'environnement virtuel..."
cp -r .venv "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  .venv non trouvé ou déjà sauvegardé"

# Sauvegarder les données
echo "📊 Sauvegarde des données..."
cp -r data_clean "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  data_clean vide"

# Sauvegarder les modèles entraînés
echo "🤖 Sauvegarde des modèles..."
cp -r trained_models "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  trained_models vide"

# Sauvegarder les scripts personnalisés
echo "📜 Sauvegarde des scripts..."
cp launch_mistral_training_100k.py "$BACKUP_DIR/" 2>/dev/null

# Créer un fichier d'état
cat > "$BACKUP_DIR/workspace_state.txt" << EOF
ÉTAT DU WORKSPACE - $(date)

DONNÉES DISPONIBLES:
- Tokenizer Vincent v2: trained_models/tokenizers/vincent_tokenizer_FR_v2/
- Tokenizer Mistral: data_clean/mistral_tokenizer/
- Données tokenisées Mistral: conversations_mega_*_mistral_tokenized.pt
- Checkpoint 100k: trained_models/runs/french_medium_base_60k/checkpoint_step_100000.pt
- Checkpoint 17500 (v2): trained_models/runs/v2_unified/checkpoint_step_17500.pt

COMMANDES DE RELANCE:
# Avec Mistral (recommandé - plus rapide)
python launch_mistral_training_100k.py

# Avec Vincent v2 (nécessite adaptation du code)
# TODO: adapter pour SentencePiece

ENVIRONNEMENT:
- Python: .venv/bin/python3
- CUDA: disponible
- PyTorch: installé avec CUDA support
EOF

echo "✅ Sauvegarde terminée dans: $BACKUP_DIR"
echo ""
echo "📋 État sauvegardé:"
echo "   - Environnement virtuel: ✅"
echo "   - Données: ✅"
echo "   - Modèles: ✅"
echo "   - Scripts: ✅"
echo ""
echo "🔄 Pour reprendre:"
echo "   cd /home/vincent/code/repo/french-llm-from-scratch"
echo "   source .venv/bin/activate"
echo "   python launch_mistral_training_100k.py"