#!/bin/bash
# 🎯 Dashboard temps réel de l'entraînement

clear

while true; do
    clear
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     🔥 ENTRAÎNEMENT 5000 STEPS + SERVICES EN DIRECT 🔥     ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Timestamp
    echo "⏰ $(date '+%H:%M:%S') | $(date '+%A %d %B %Y')"
    echo ""
    
    # ============== ENTRAÎNEMENT ==============
    echo "┌─ 🔴 ENTRAÎNEMENT ────────────────────────────────────────"
    
    # Vérifier si le processus tourne
    PID=$(pgrep -f "continue_training_5k.py" | head -1)
    if [ -z "$PID" ]; then
        echo "│ Status: ❌ ARRÊTÉ"
    else
        echo "│ Status: ✅ ACTIF (PID: $PID)"
        
        # Récupérer les infos du log
        if [ -f training_5k_extended.log ]; then
            # Chercher le dernier step
            LAST_LOG=$(tail -1 training_5k_extended.log)
            echo "│ Log dernière ligne: $LAST_LOG" | head -c 68
            echo ""
            
            # Chercher les étapes
            STEPS=$(grep -o "\\[[0-9]*/5000\\]" training_5k_extended.log 2>/dev/null | tail -1)
            if [ ! -z "$STEPS" ]; then
                echo "│ Steps: $STEPS"
            fi
            
            # Chercher la loss
            LOSS=$(grep -oE "loss.*[0-9]+\.[0-9]+" training_5k_extended.log 2>/dev/null | tail -1)
            if [ ! -z "$LOSS" ]; then
                echo "│ $LOSS"
            fi
        fi
    fi
    echo "│"
    
    # ============== GPU ==============
    echo "│ 🖥️  GPU RTX 5080:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | while read util_gpu util_mem mem_used mem_total temp; do
        # Barre de progression GPU
        bar_gpu=$((util_gpu / 10))
        bar_mem=$((util_mem / 10))
        GPU_BAR=""
        MEM_BAR=""
        for ((i=0; i<10; i++)); do
            [ $i -lt $bar_gpu ] && GPU_BAR="${GPU_BAR}█" || GPU_BAR="${GPU_BAR}░"
            [ $i -lt $bar_mem ] && MEM_BAR="${MEM_BAR}█" || MEM_BAR="${MEM_BAR}░"
        done
        echo "│    GPU: ${GPU_BAR} ${util_gpu}% | Mem: ${MEM_BAR} ${util_mem}% (${mem_used}MB/${mem_total}MB) | Temp: ${temp}°C"
    done
    echo "│"
    
    # ============== SERVICES ==============
    echo "│ 🌐 SERVICES:"
    
    # FastAPI
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "│    ✅ FastAPI (localhost:8000)"
    else
        echo "│    ❌ FastAPI (localhost:8000)"
    fi
    
    # TorchServe
    if pgrep -f "java.*torchserve" > /dev/null 2>&1; then
        echo "│    ✅ TorchServe (localhost:8080)"
    else
        echo "│    ❌ TorchServe (localhost:8080)"
    fi
    
    # Dashboard
    if pgrep -f "dashboard/server.py" > /dev/null 2>&1; then
        echo "│    ✅ Dashboard (localhost:5000)"
    else
        echo "│    ❌ Dashboard (localhost:5000)"
    fi
    echo "│"
    
    # ============== INFOS ==============
    echo "│ 📊 STATISTIQUES:"
    
    # CPU
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print int($2)}')
    echo "│    CPU: ${CPU_USAGE}%"
    
    # Mémoire
    MEM_INFO=$(free -h | grep Mem)
    echo "│    RAM: $MEM_INFO" | cut -d' ' -f1-8
    echo "│"
    
    # ============== CHECKPOINTS ==============
    echo "│ 💾 CHECKPOINTS:"
    CHECKPOINT_COUNT=$(find trained_models/runs/french_llm_hf_extended -name "checkpoint-*" -type d 2>/dev/null | wc -l)
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        echo "│    Créés: $CHECKPOINT_COUNT"
        LATEST=$(find trained_models/runs/french_llm_hf_extended -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 | xargs basename)
        echo "│    Dernier: $LATEST"
    else
        echo "│    Aucun checkpoint encore"
    fi
    echo "│"
    
    # ============== COMMANDES ==============
    echo "└─ 📖 COMMANDES UTILES ─────────────────────────────────────"
    echo ""
    echo "  Voir les logs:          tail -f training_5k_extended.log"
    echo "  Arrêter l'entraînement: kill \$(pgrep -f continue_training_5k.py)"
    echo "  Voir GPU en direct:     watch -n1 nvidia-smi"
    echo "  Test API:               curl http://localhost:8000/info"
    echo "  Quitter ce dashboard:   CTRL+C"
    echo ""
    
    sleep 3
done
