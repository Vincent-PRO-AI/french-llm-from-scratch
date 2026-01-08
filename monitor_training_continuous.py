#!/usr/bin/env python3
"""
Script de monitoring continu pour l'entraînement
Détecte et alerte sur les problèmes en temps réel
"""

import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys

class TrainingMonitor:
    def __init__(self, base_path="/home/vincent/code/repo/french-llm-from-scratch"):
        self.base_path = Path(base_path)
        self.log_file = self.base_path / "training_corrected.log"
        self.metrics_file = self.base_path / "trained_models/runs/french_medium_rtx5080_batch8/metrics.jsonl"
        self.last_log_pos = 0
        self.last_step = 0
        self.step_times = []
        self.start_time = datetime.now()
        
    def get_process_info(self):
        """Récupère les infos du processus d'entraînement"""
        result = subprocess.run(
            ["pgrep", "-f", "train_subtitles_transformer"],
            capture_output=True,
            text=True
        )
        
        if not result.stdout:
            return None
            
        pids = result.stdout.strip().split('\n')
        main_pid = pids[0]
        
        # Get process details
        result = subprocess.run(
            ["ps", "-p", main_pid, "-o", "pid,vsz,rss,%cpu,%mem,etime"],
            capture_output=True,
            text=True
        )
        
        return result.stdout.strip().split('\n')[-1] if result.stdout else None
    
    def check_log_errors(self):
        """Vérifie les erreurs dans les logs"""
        if not self.log_file.exists():
            return []
            
        errors = []
        with open(self.log_file, 'r') as f:
            f.seek(self.last_log_pos)
            new_lines = f.readlines()
            self.last_log_pos = f.tell()
        
        for line in new_lines:
            if any(x in line for x in ["Error", "Traceback", "CUDA", "out of memory", "Exception"]):
                errors.append(line.strip())
        
        return errors
    
    def check_metrics(self):
        """Vérifie les métriques d'entraînement"""
        if not self.metrics_file.exists():
            return None
            
        try:
            with open(self.metrics_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return None
                
            last_line = json.loads(lines[-1])
            current_step = last_line.get('step', 0)
            
            if current_step > self.last_step:
                self.last_step = current_step
                return last_line
            
            return None
        except:
            return None
    
    def check_gpu_memory(self):
        """Vérifie l'utilisation GPU"""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            parts = result.stdout.strip().split(',')
            return {
                'used': parts[0].strip(),
                'total': parts[1].strip(),
                'util': parts[2].strip()
            }
        return None
    
    def check_checkpoint_progress(self):
        """Vérifie la progression des checkpoints"""
        run_dir = self.base_path / "trained_models/runs/french_medium_rtx5080_batch8"
        if not run_dir.exists():
            return None
            
        checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"))
        if checkpoints:
            last = checkpoints[-1]
            return {
                'path': last.name,
                'size': f"{last.stat().st_size / (1024**3):.2f} GB",
                'step': int(last.stem.split('_')[-1])
            }
        return None
    
    def print_status(self):
        """Affiche le status"""
        now = datetime.now()
        elapsed = now - self.start_time
        
        print("\n" + "="*80)
        print(f"⏰ {now.strftime('%Y-%m-%d %H:%M:%S')} | Elapsed: {str(elapsed).split('.')[0]}")
        print("="*80)
        
        # Process info
        proc_info = self.get_process_info()
        if proc_info:
            print(f"✅ PROCESS RUNNING")
            print(f"   {proc_info}")
        else:
            print(f"❌ NO PROCESS FOUND")
            return False
        
        # Errors
        errors = self.check_log_errors()
        if errors:
            print(f"\n⚠️  ERRORS DETECTED:")
            for error in errors[-3:]:  # Show last 3 errors
                print(f"   {error[:100]}")
            return False
        
        # Metrics
        metrics = self.check_metrics()
        if metrics:
            print(f"\n📊 METRICS (Step {metrics.get('step')})")
            print(f"   Loss: {metrics.get('train_loss', 'N/A'):.4f}")
            print(f"   Time: {metrics.get('time', 'N/A')}")
        else:
            print(f"\n⏳ NO NEW METRICS YET")
        
        # GPU
        gpu = self.check_gpu_memory()
        if gpu:
            print(f"\n🎮 GPU MEMORY")
            print(f"   Used: {gpu['used']} / {gpu['total']}")
            print(f"   Util: {gpu['util']}")
        
        # Checkpoint
        ckpt = self.check_checkpoint_progress()
        if ckpt:
            print(f"\n💾 LATEST CHECKPOINT")
            print(f"   {ckpt['path']} ({ckpt['size']})")
        
        print("="*80)
        return True
    
    def run_continuous(self, interval=30):
        """Lance le monitoring continu"""
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔍 MONITORING CONTINU EN COURS                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Intervalle: {interval}s
Fichiers monitorés:
  - {self.log_file}
  - {self.metrics_file}

Appuyez sur Ctrl+C pour arrêter.
""")
        
        try:
            while True:
                self.print_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n✅ Monitoring arrêté")
            sys.exit(0)

if __name__ == "__main__":
    monitor = TrainingMonitor()
    
    # Paramètre optionnel: intervalle de refresh
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    monitor.run_continuous(interval)
