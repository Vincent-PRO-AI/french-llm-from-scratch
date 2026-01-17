#!/bin/bash
# Launcher pour training 4090 optimisé avec monitoring

set -e

cd /home/vincent/code/repo/french-llm-from-scratch

echo "🎯 Pré-vérifications..."
if [ ! -f "trained_models/tiny_subtitles_transformer.pt" ]; then
    echo "❌ Checkpoint manquant!"
    exit 1
fi

echo "✅ Checkpoint trouvé"
echo "✅ Env conda: french-llm"

# Lancer l'entraînement
echo ""
echo "════════════════════════════════════════════════"
echo "🚀 LANCEMENT TRAINING RTX 4090 OPTIMISÉ"
echo "════════════════════════════════════════════════"
echo ""

nohup /home/vincent/miniconda3/envs/french-llm/bin/python train_4090_optimized.py \
    > training_4090_optimized.log 2>&1 &

TRAIN_PID=$!
echo "📌 PID: $TRAIN_PID"
echo "📊 Log: tail -f training_4090_optimized.log"
echo ""
echo "Monitoring activé. Appuie Ctrl+C pour quitter le monitoring (pas le training)"
echo ""

# Monitoring
while kill -0 $TRAIN_PID 2>/dev/null; do
    clear
    echo "🎯 TRAINING EN COURS..."
    echo "════════════════════════════════════════════════"
    tail -n 15 training_4090_optimized.log | grep -E "step=|val_loss|Checkpoint|Phase"
    echo "════════════════════════════════════════════════"
    sleep 30
done

echo ""
echo "✅ Training terminé (PID $TRAIN_PID)"
tail -n 20 training_4090_optimized.log
