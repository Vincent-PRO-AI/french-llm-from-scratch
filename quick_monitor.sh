#!/bin/bash
# Simple one-liner monitor
watch -n 5 "echo '=== TRAINING MONITOR ===' && echo \"Updated: \$(date)\" && echo '' && tail -1 /home/vincent/code/repo/french-llm-from-scratch/training_V3_mistral.log && echo '' && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,nounits,noheader -i 0 | awk '{print \"GPU: \" \$1 \"% | VRAM: \" \$2 \"MB\"}' && ps aux | grep train_subtitles_transformer.py | grep -v grep | head -1 | awk '{print \"PID: \" \$2 \" | CPU: \" \$3 \"% | MEM: \" \$4 \"%\"}'"
