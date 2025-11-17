#!/bin/bash
# Monitoring en temps réel du fine-tuning

METRICS_FILE="trained_models/runs/french_medium_finetune_conversations/metrics.jsonl"
SAMPLES_FILE="trained_models/runs/french_medium_finetune_conversations/samples.txt"
PID=9309

clear
echo "========================================"
echo "🎯 Monitoring Fine-tuning en temps réel"
echo "========================================"
echo ""

while true; do
    # Position curseur en haut
    tput cup 5 0
    
    # État du processus
    if ps -p $PID > /dev/null 2>&1; then
        PROCESS_INFO=$(ps -p $PID -o etime,%cpu,%mem --no-headers)
        echo "📊 Processus (PID: $PID): ✅ Actif"
        echo "   Durée: $PROCESS_INFO"
    else
        echo "❌ Processus terminé ou arrêté"
        exit 0
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 Métriques (10 derniers steps):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -f "$METRICS_FILE" ]; then
        tail -10 "$METRICS_FILE" | python3 -c "
import sys, json
for line in sys.stdin:
    if line.strip():
        d = json.loads(line)
        step = d.get('step', '?')
        loss = d.get('loss', 0)
        print(f'Step {step:>6} | Loss: {loss:.4f}')
" 2>/dev/null || echo "En attente des métriques..."
    else
        echo "⏳ Fichier de métriques non encore créé..."
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💬 Dernier échantillon généré:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -f "$SAMPLES_FILE" ]; then
        # Prendre le dernier échantillon et le nettoyer
        tail -1 "$SAMPLES_FILE" | sed 's/Ġ/ /g; s/Ċ/\n/g; s/âĢĻ/'\''/g; s/âĢĶ/—/g' | fold -w 70 -s
    else
        echo "⏳ Aucun échantillon généré pour l'instant..."
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 Actualisation dans 20s... (Ctrl+C pour quitter)"
    echo ""
    
    sleep 20
done
