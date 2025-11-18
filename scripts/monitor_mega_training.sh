#!/bin/bash
# Monitoring temps réel du fine-tuning méga (115k→180k)

clear
echo "🚀 MONITORING FINE-TUNING MÉGA (115k→180k)"
echo "=========================================="
echo ""

while true; do
    # Position curseur en haut
    tput cup 0 0
    
    echo "🚀 MONITORING FINE-TUNING MÉGA (115k→180k)   $(date +%H:%M:%S)"
    echo "=========================================="
    echo ""
    
    # Processus
    if ps aux | grep -q "[t]rain_subtitles_transformer.py"; then
        PROC_INFO=$(ps aux | grep "[t]rain_subtitles_transformer.py" | awk '{printf "PID %s | CPU %s%% | RAM %.1f GB", $2, $3, $6/1024/1024}')
        echo "✅ Processus actif: $PROC_INFO"
    else
        echo "❌ Processus arrêté"
        break
    fi
    
    # GPU
    GPU_INFO=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    GPU_MEM=$(echo "$GPU_INFO" | cut -d',' -f1)
    GPU_UTIL=$(echo "$GPU_INFO" | cut -d',' -f2)
    echo "🎮 GPU: ${GPU_MEM} MB VRAM | ${GPU_UTIL}% utilisation"
    echo ""
    
    # Métriques d'entraînement
    if [ -f "trained_models/runs/french_medium_mega_finetune/metrics.jsonl" ]; then
        python3 << 'PYTHON'
import json
from pathlib import Path

metrics_file = Path("trained_models/runs/french_medium_mega_finetune/metrics.jsonl")
lines = metrics_file.read_text().strip().split('\n')

train_metrics = [json.loads(line) for line in lines if 'train' in json.loads(line).get('split', '')]
val_metrics = [json.loads(line) for line in lines if 'val' in json.loads(line).get('split', '')]

if train_metrics and val_metrics:
    last_train = train_metrics[-1]
    last_val = val_metrics[-1]
    
    print(f"📊 MÉTRIQUES")
    print(f"   Step: {last_val['step']:,} / 180,000")
    print(f"   Train loss: {last_train['loss']:.4f}")
    print(f"   Val loss: {last_val['loss']:.4f}")
    
    # Progression
    progress = (last_val['step'] - 115000) / (180000 - 115000) * 100
    bar_len = 40
    filled = int(bar_len * progress / 100)
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f"   Progression: [{bar}] {progress:.1f}%")
    
    # Vitesse et ETA
    if len(val_metrics) >= 10:
        recent = val_metrics[-10:]
        first = recent[0]
        last = recent[-1]
        
        steps_done = last['step'] - first['step']
        time_taken = last['timestamp'] - first['timestamp']
        steps_per_sec = steps_done / time_taken
        steps_per_hour = steps_per_sec * 3600
        
        remaining_steps = 180000 - last['step']
        eta_hours = remaining_steps / steps_per_hour
        eta_min = eta_hours * 60
        
        print(f"\n⚡ VITESSE")
        print(f"   {steps_per_hour:.0f} steps/h")
        print(f"   ETA: {int(eta_hours)}h{int(eta_min % 60):02d}min")
        
        # Loss improvement
        if len(val_metrics) >= 100:
            baseline_loss = val_metrics[0]['loss']
            current_loss = last['loss']
            improvement = (baseline_loss - current_loss) / baseline_loss * 100
            print(f"\n📉 AMÉLIORATION")
            print(f"   Baseline (step 115k): {baseline_loss:.4f}")
            print(f"   Actuel: {current_loss:.4f}")
            print(f"   Gain: {improvement:.1f}%")

PYTHON
    fi
    
    echo ""
    echo "📝 Dernière génération:"
    if [ -f "trained_models/runs/french_medium_mega_finetune/samples.txt" ]; then
        tail -5 "trained_models/runs/french_medium_mega_finetune/samples.txt" | head -3
    fi
    
    echo ""
    echo "🔄 Refresh automatique toutes les 15s (Ctrl+C pour arrêter)"
    sleep 15
done
