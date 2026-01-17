#!/bin/bash
# Monitor training progress

LOG_FILE="/home/vincent/code/repo/french-llm-from-scratch/trained_models/logs/french_llm_4090_optimized_250k_to_500k/metrics.jsonl"

echo "Monitoring training..."
while true; do
    if [ -f "$LOG_FILE" ]; then
        LINES=$(wc -l < "$LOG_FILE")
        echo "Metrics logged: $LINES steps"
        tail -1 "$LOG_FILE" 2>/dev/null | head -c 200
        echo ""
    fi
    sleep 10
done
