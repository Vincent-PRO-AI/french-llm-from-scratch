#!/usr/bin/env python3
"""
🎯 PYTORCH TRAINING DASHBOARD
============================
Real-time monitoring interface for PyTorch training
Flask web UI + NVIDIA GPU metrics + Training logs
"""

import torch
import json
import psutil
import subprocess
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from threading import Thread
import time
import os

app = Flask(__name__)

# Training state
TRAINING_STATE = {
    "running": False,
    "step": 0,
    "max_steps": 80000,
    "loss": 0.0,
    "lr": 0.0,
    "eval_loss": 0.0,
    "start_time": None,
    "checkpoint_dir": "trained_models/runs/french_v3_llama_260m",
}

def get_gpu_stats():
    """Get GPU memory and utilization"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split('\n')[1].split(',')
            return {
                "memory_used_mb": int(parts[0]),
                "memory_total_mb": int(parts[1]),
                "gpu_util": int(parts[2]),
            }
    except:
        pass
    return {"memory_used_mb": 0, "memory_total_mb": 16000, "gpu_util": 0}

def get_cpu_stats():
    """Get CPU and system stats"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_gb": psutil.virtual_memory().used / (1024**3),
    }

def parse_training_log():
    """Parse training log file for metrics"""
    log_file = Path("training_pytorch.log")
    if not log_file.exists():
        return TRAINING_STATE
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Parse last few lines for metrics
        for line in reversed(lines[-50:]):
            if "Training:" in line:
                # Extract step info from progress bar
                try:
                    parts = line.split('|')
                    if len(parts) > 1:
                        step_part = parts[0].split('Training:')[1].strip()
                        # Extract loss, lr from postfix
                        if "loss:" in line:
                            loss_str = line.split("loss:")[1].split(',')[0].strip()
                            TRAINING_STATE["loss"] = float(loss_str)
                        if "lr:" in line:
                            lr_str = line.split("lr:")[1].split(',')[0].strip()
                            TRAINING_STATE["lr"] = float(lr_str)
                except:
                    pass
            
            elif "Step" in line and "Eval loss:" in line:
                try:
                    step_str = line.split('[Step')[1].split(']')[0].strip()
                    loss_str = line.split('Eval loss:')[1].strip()
                    TRAINING_STATE["step"] = int(step_str)
                    TRAINING_STATE["eval_loss"] = float(loss_str)
                except:
                    pass
    
    except Exception as e:
        pass
    
    return TRAINING_STATE

def monitor_training():
    """Background thread to monitor training process"""
    while True:
        try:
            # Check if training process is running
            result = subprocess.run(
                ["pgrep", "-f", "train_pytorch_native.py"],
                capture_output=True,
                text=True
            )
            TRAINING_STATE["running"] = result.returncode == 0
            
            # Parse log file
            parse_training_log()
            
            # Calculate ETA
            if TRAINING_STATE["step"] > 0 and TRAINING_STATE["start_time"]:
                elapsed = time.time() - TRAINING_STATE["start_time"]
                if TRAINING_STATE["step"] > 0:
                    avg_time_per_step = elapsed / TRAINING_STATE["step"]
                    remaining_steps = TRAINING_STATE["max_steps"] - TRAINING_STATE["step"]
                    TRAINING_STATE["eta_seconds"] = int(remaining_steps * avg_time_per_step)
        
        except Exception as e:
            print(f"Monitor error: {e}")
        
        time.sleep(2)

# HTML TEMPLATE
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 PyTorch Training Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #00ff88;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #00ff88;
            padding-bottom: 20px;
        }
        
        .header h1 {
            font-size: 2.5em;
            text-shadow: 0 0 10px #00ff88;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #00ccff;
            font-size: 1.1em;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(20, 20, 40, 0.8);
            border: 2px solid #00ff88;
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        .card h2 {
            color: #00ffff;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 1px solid #00ff88;
            padding-bottom: 10px;
        }
        
        .metric {
            margin: 12px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .metric-label {
            color: #00ccff;
            font-weight: bold;
        }
        
        .metric-value {
            color: #00ff88;
            font-size: 1.2em;
            font-weight: bold;
            text-shadow: 0 0 5px #00ff88;
        }
        
        .bar {
            width: 100%;
            height: 25px;
            background: rgba(0, 100, 100, 0.3);
            border: 1px solid #00ff88;
            border-radius: 5px;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #00ffff);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #000;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .status {
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin: 10px 0;
            font-weight: bold;
        }
        
        .status.running {
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
            border: 1px solid #00ff88;
        }
        
        .status.stopped {
            background: rgba(255, 0, 100, 0.2);
            color: #ff0064;
            border: 1px solid #ff0064;
        }
        
        .logs {
            grid-column: 1 / -1;
            background: rgba(20, 20, 40, 0.9);
            border: 2px solid #00ff88;
            border-radius: 10px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .logs h2 {
            color: #00ffff;
            margin-bottom: 15px;
            border-bottom: 1px solid #00ff88;
            padding-bottom: 10px;
        }
        
        .log-line {
            color: #00ff88;
            margin: 5px 0;
            padding: 5px;
            border-left: 2px solid #00ffff;
            padding-left: 10px;
            font-size: 0.95em;
        }
        
        .log-error {
            color: #ff0064;
        }
        
        .log-info {
            color: #00ccff;
        }
        
        .actions {
            grid-column: 1 / -1;
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        
        button {
            padding: 12px 30px;
            font-size: 1em;
            border: 2px solid #00ff88;
            background: rgba(0, 255, 136, 0.1);
            color: #00ff88;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        button:hover {
            background: rgba(0, 255, 136, 0.3);
            box-shadow: 0 0 10px #00ff88;
        }
        
        .warning {
            color: #ffff00;
            background: rgba(255, 255, 0, 0.1);
            padding: 10px;
            border-left: 3px solid #ffff00;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 PYTORCH TRAINING DASHBOARD</h1>
            <p>French LLM V3 - LLaMA 260M en temps réel</p>
        </div>
        
        <div class="grid">
            <!-- Training Status -->
            <div class="card">
                <h2>📊 STATUS</h2>
                <div id="status" class="status running">▶️ EN COURS</div>
                <div class="metric">
                    <span class="metric-label">Processus:</span>
                    <span class="metric-value" id="process-status">PID: 29738</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Temps écoulé:</span>
                    <span class="metric-value" id="elapsed-time">00:05:30</span>
                </div>
                <div class="metric">
                    <span class="metric-label">ETA:</span>
                    <span class="metric-value" id="eta-time">~4 jours 12h</span>
                </div>
            </div>
            
            <!-- Training Progress -->
            <div class="card">
                <h2>📈 PROGRESSION</h2>
                <div class="metric">
                    <span class="metric-label">Étape:</span>
                    <span class="metric-value" id="step">0 / 80,000</span>
                </div>
                <div class="bar">
                    <div class="bar-fill" id="progress-bar" style="width: 0%">0%</div>
                </div>
                <div class="metric" style="margin-top: 15px;">
                    <span class="metric-label">Perte (Loss):</span>
                    <span class="metric-value" id="loss">—</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Perte Eval:</span>
                    <span class="metric-value" id="eval-loss">—</span>
                </div>
            </div>
            
            <!-- Learning Rate -->
            <div class="card">
                <h2>⚡ OPTIMISATION</h2>
                <div class="metric">
                    <span class="metric-label">Learning Rate:</span>
                    <span class="metric-value" id="lr">2.0e-4</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Batch Size:</span>
                    <span class="metric-value">12 × 2 (eff: 24)</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Seq Length:</span>
                    <span class="metric-value">2048</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Précision:</span>
                    <span class="metric-value">BF16 + TF32</span>
                </div>
            </div>
            
            <!-- GPU Stats -->
            <div class="card">
                <h2>🎮 GPU (RTX 5080)</h2>
                <div class="metric">
                    <span class="metric-label">Utilisation:</span>
                    <span class="metric-value" id="gpu-util">99%</span>
                </div>
                <div class="bar">
                    <div class="bar-fill" id="gpu-bar" style="width: 99%">99%</div>
                </div>
                <div class="metric" style="margin-top: 15px;">
                    <span class="metric-label">Mémoire:</span>
                    <span class="metric-value" id="gpu-mem">15.8 / 16.3 GB</span>
                </div>
                <div class="bar">
                    <div class="bar-fill" id="gpu-mem-bar" style="width: 97%">97%</div>
                </div>
            </div>
            
            <!-- CPU Stats -->
            <div class="card">
                <h2>💻 SYSTÈME</h2>
                <div class="metric">
                    <span class="metric-label">CPU:</span>
                    <span class="metric-value" id="cpu-util">45%</span>
                </div>
                <div class="bar">
                    <div class="bar-fill" id="cpu-bar" style="width: 45%">45%</div>
                </div>
                <div class="metric" style="margin-top: 15px;">
                    <span class="metric-label">RAM:</span>
                    <span class="metric-value" id="ram-util">32.5 / 64 GB</span>
                </div>
                <div class="bar">
                    <div class="bar-fill" id="ram-bar" style="width: 51%">51%</div>
                </div>
            </div>
            
            <!-- Dataset Info -->
            <div class="card">
                <h2>📚 DATASET</h2>
                <div class="metric">
                    <span class="metric-label">Tokens:</span>
                    <span class="metric-value">85.5M</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Langue:</span>
                    <span class="metric-value">100% Français</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Train/Test:</span>
                    <span class="metric-value">85% / 15%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Séquences:</span>
                    <span class="metric-value">70,963</span>
                </div>
            </div>
            
            <!-- Model Info -->
            <div class="card">
                <h2>🧠 MODÈLE</h2>
                <div class="metric">
                    <span class="metric-label">Architecture:</span>
                    <span class="metric-value">LLaMA 260M</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Couches:</span>
                    <span class="metric-value">18</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Têtes:</span>
                    <span class="metric-value">32</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Paramètres:</span>
                    <span class="metric-value">334.7M</span>
                </div>
            </div>
            
            <!-- Throughput -->
            <div class="card">
                <h2>⚙️ PERFORMANCE</h2>
                <div class="metric">
                    <span class="metric-label">Tokens/sec:</span>
                    <span class="metric-value" id="tokens-sec">24,576</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Steps/min:</span>
                    <span class="metric-value" id="steps-min">2.4</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Power:</span>
                    <span class="metric-value">69W</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Grad Ckpt:</span>
                    <span class="metric-value">✅ Activé</span>
                </div>
            </div>
            
            <!-- Training Logs -->
            <div class="logs">
                <h2>📝 LOGS EN DIRECT</h2>
                <div id="logs-content">
                    <div class="log-line log-info">▶️ Démarrage du training...</div>
                    <div class="log-line log-info">📊 Dataset chargé: 85,490,944 tokens</div>
                    <div class="log-line log-info">🏗️ Modèle compilé: 334,795,776 params</div>
                    <div class="log-line log-info">⚡ GPU initialisé: RTX 5080 (16GB)</div>
                    <div class="log-line">Training:   0%|▏         | 0/80000 [00:00<?, ?it/s]</div>
                </div>
            </div>
        </div>
        
        <div class="actions">
            <button onclick="pauseTraining()">⏸️ PAUSE</button>
            <button onclick="resumeTraining()">▶️ REPRENDRE</button>
            <button onclick="stopTraining()">⛔ ARRÊTER</button>
            <button onclick="refreshMetrics()">🔄 ACTUALISER</button>
        </div>
    </div>
    
    <script>
        // Auto-refresh metrics every 2 seconds
        setInterval(refreshMetrics, 2000);
        
        function refreshMetrics() {
            fetch('/api/metrics')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('step').textContent = `${data.step.toLocaleString()} / 80,000`;
                    document.getElementById('loss').textContent = data.loss.toFixed(4);
                    document.getElementById('eval-loss').textContent = data.eval_loss.toFixed(4);
                    document.getElementById('lr').textContent = data.lr.toExponential(2);
                    
                    // Progress bar
                    const progress = (data.step / 80000) * 100;
                    document.getElementById('progress-bar').style.width = progress + '%';
                    document.getElementById('progress-bar').textContent = progress.toFixed(1) + '%';
                    
                    // GPU stats
                    document.getElementById('gpu-util').textContent = data.gpu_util + '%';
                    document.getElementById('gpu-bar').style.width = data.gpu_util + '%';
                    document.getElementById('gpu-mem').textContent = `${(data.gpu_mem/1024).toFixed(1)} / 16.3 GB`;
                    
                    // CPU stats
                    document.getElementById('cpu-util').textContent = data.cpu_percent.toFixed(1) + '%';
                    document.getElementById('cpu-bar').style.width = data.cpu_percent + '%';
                    document.getElementById('ram-util').textContent = `${data.ram_gb.toFixed(1)} / 64 GB`;
                    
                    // Status
                    const status = document.getElementById('status');
                    if (data.running) {
                        status.className = 'status running';
                        status.textContent = '▶️ EN COURS';
                    } else {
                        status.className = 'status stopped';
                        status.textContent = '⛔ ARRÊTÉ';
                    }
                })
                .catch(e => console.error('Erreur:', e));
        }
        
        function pauseTraining() {
            alert('Pause non implémentée');
        }
        
        function resumeTraining() {
            alert('Reprendre non implémenté');
        }
        
        function stopTraining() {
            if (confirm('Êtes-vous sûr de vouloir arrêter l\'entraînement?')) {
                alert('Stop non implémenté');
            }
        }
        
        // Initial load
        refreshMetrics();
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics"""
    gpu_stats = get_gpu_stats()
    cpu_stats = get_cpu_stats()
    
    return jsonify({
        "running": TRAINING_STATE["running"],
        "step": TRAINING_STATE["step"],
        "max_steps": TRAINING_STATE["max_steps"],
        "loss": TRAINING_STATE["loss"],
        "eval_loss": TRAINING_STATE["eval_loss"],
        "lr": TRAINING_STATE["lr"],
        "gpu_util": gpu_stats["gpu_util"],
        "gpu_mem": gpu_stats["memory_used_mb"],
        "cpu_percent": cpu_stats["cpu_percent"],
        "ram_gb": cpu_stats["ram_gb"],
    })

if __name__ == '__main__':
    # Start monitoring thread
    TRAINING_STATE["start_time"] = time.time()
    monitor_thread = Thread(target=monitor_training, daemon=True)
    monitor_thread.start()
    
    print("\n" + "="*70)
    print("🎯 PYTORCH TRAINING DASHBOARD")
    print("="*70)
    print("📊 Dashboard disponible sur: http://127.0.0.1:5000")
    print("🔗 Ouvre dans le navigateur pour voir le monitoring en temps réel")
    print("="*70 + "\n")
    
    # Start Flask app
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
