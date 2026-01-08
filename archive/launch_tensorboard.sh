#!/bin/bash

# 🎯 Lance TensorBoard pour monitoring
# Usage: bash launch_tensorboard.sh

LOG_DIR="./trained_models/mistral_v1_gpu/logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Dossier logs introuvable: $LOG_DIR"
    echo "Assure-toi que l'entraînement est lancé."
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         🎨 TENSORBOARD - Monitoring Entraînement              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Logs directory: $LOG_DIR"
echo ""
echo "🚀 Lancement TensorBoard..."
echo ""
echo "   URL: http://localhost:6006"
echo ""
echo "Ouvre http://localhost:6006 dans ton navigateur"
echo "Appuie sur Ctrl+C pour arrêter TensorBoard"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

conda activate french-llm
tensorboard --logdir="$LOG_DIR" --port=6006 --bind_all
