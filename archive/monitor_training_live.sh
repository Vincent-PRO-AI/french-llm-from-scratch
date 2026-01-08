#!/bin/bash
# Real-time training monitor

LOG_FILE="/home/vincent/code/repo/french-llm-from-scratch/training_V3_mistral.log"

echo "🚀 Training Monitor - Live Progress"
echo "===================================="

while true; do
    clear
    echo "🚀 Training Monitor - Live Progress"
    echo "===================================="
    echo "Updated: $(date '+%H:%M:%S')"
    echo ""
    
    # Show last 5 steps
    echo "📊 Last 5 steps:"
    grep "step=" "$LOG_FILE" 2>/dev/null | tail -5 | sed 's/^/  /'
    
    echo ""
    
    # Get current step and calculate progress
    CURRENT_STEP=$(grep "step=" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP 'step=\K[0-9]+' || echo "0")
    MAX_STEPS=100000
    PERCENT=$((CURRENT_STEP * 100 / MAX_STEPS))
    
    # Progress bar
    BAR_LENGTH=50
    FILLED=$((CURRENT_STEP * BAR_LENGTH / MAX_STEPS))
    EMPTY=$((BAR_LENGTH - FILLED))
    
    echo "📈 Progress: $CURRENT_STEP / $MAX_STEPS ($PERCENT%)"
    printf "   ["
    printf "%${FILLED}s" | tr ' ' '█'
    printf "%${EMPTY}s" | tr ' ' '░'
    printf "]\n"
    echo ""
    
    # Time estimation
    if [ "$CURRENT_STEP" -gt 0 ]; then
        ELAPSED=$(grep "step=" "$LOG_FILE" 2>/dev/null | wc -l)
        if [ "$ELAPSED" -gt 10 ]; then
            FIRST_STEP_TIME=$(head -1 "$LOG_FILE" | grep -oP 'step=\K[0-9]+' || echo "")
            TIME_PER_STEP=$(echo "scale=3; $(date +%s) / $ELAPSED" | bc 2>/dev/null || echo "N/A")
            REMAINING=$((MAX_STEPS - CURRENT_STEP))
            ETA_SECONDS=$(echo "$REMAINING * $TIME_PER_STEP" | bc 2>/dev/null || echo "N/A")
            
            echo "⏱️  Estimated time remaining: ~$(echo "$ETA_SECONDS / 3600" | bc 2>/dev/null || echo "?") hours"
        fi
    fi
    echo ""
    
    # GPU status
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -i 0 2>/dev/null | tail -1 | sed 's/^/  /'
    
    echo ""
    echo "📋 Process Status:"
    ps aux | grep "train_subtitles_transformer.py" | grep -v grep | awk '{print "  PID: " $2 " | CPU: " $3 "% | MEM: " $4 "%"}' || echo "  ❌ Not running"
    
    echo ""
    echo "Press Ctrl+C to exit. Refresh in 30 seconds..."
    sleep 30
done
