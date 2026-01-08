#!/bin/bash
# Script complet: Entraînement → Export GGUF → Publication HuggingFace
# Usage: ./full_pipeline.sh

set -e

cd /home/vincent/code/repo/french-llm-from-scratch

echo "🚀 PIPELINE COMPLET: ENTRAÎNEMENT → EXPORT → PUBLICATION"
echo "========================================================================"

# 1. ENTRAÎNEMENT
echo ""
echo "📚 PHASE 1: ENTRAÎNEMENT (160k → 500k steps)"
echo "------"

if pgrep -f "train_subtitles_transformer.py" > /dev/null; then
    echo "✅ Entraînement déjà en cours (PID: $(pgrep -f 'train_subtitles_transformer.py'))"
    echo "⏳ Attente de la fin de l'entraînement..."
    
    # Attendre la fin
    source .venv/bin/activate
    python3 monitor_and_export.py
else
    echo "🎯 Lancement de l'entraînement..."
    source .venv/bin/activate
    bash start_training.sh &
    sleep 5
    python3 monitor_and_export.py
fi

# 2. EXPORT GGUF
echo ""
echo "📦 PHASE 2: EXPORT GGUF POUR LM STUDIO"
echo "------"

source .venv/bin/activate
python3 quick_export_gguf.py

# 3. PUBLICATION HUGGINGFACE
echo ""
echo "🌐 PHASE 3: PUBLICATION HUGGINGFACE"
echo "------"

source .venv/bin/activate
python3 publish_huggingface.py

echo ""
echo "========================================================================"
echo "✅ PIPELINE COMPLET TERMINÉ!"
echo "========================================================================"
echo ""
echo "📊 Résumé:"
echo "  ✅ Entraînement: 500k steps complétés"
echo "  ✅ Export GGUF: trained_models/exports/french_medium_500k.gguf"
echo "  ✅ HuggingFace: Vincent-PRO-AI/french-llm-500k"
echo ""
echo "🎉 Votre LLM français est prêt à l'emploi!"
echo "   • Utilisable localement via LM Studio"
echo "   • Disponible sur HuggingFace pour l'inférence/fine-tuning"
echo "   • Optimisé pour votre RTX 5080"
