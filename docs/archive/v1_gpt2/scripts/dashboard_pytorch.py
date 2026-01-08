#!/usr/bin/env python3
"""
Dashboard PyTorch optimisé pour monitoring V2 training en temps réel.
Port: 5000 (Flask API)
Affiche: Loss, LR, GPU mem, temps, samples
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, render_template_string
import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "trained_models" / "runs"
LATEST_RUN = RUNS_DIR / "french_v2_optimized_rtx5080"

app = Flask(__name__)

# HTML Dashboard inline
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PyTorch Training Dashboard - V2</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        header {
            color: white;
            margin-bottom: 30px;
            text-align: center;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        .card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        .metric {
            margin: 10px 0;
            padding: 8px;
            background: #f5f5f5;
            border-left: 3px solid #667eea;
            border-radius: 4px;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            font-weight: 500;
        }
        .metric-value {
            color: #333;
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 5px;
            font-family: 'Courier New', monospace;
        }
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            position: relative;
            height: 400px;
        }
        .chart-title {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
            font-weight: bold;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            margin: 5px;
        }
        .status.running {
            background: #4CAF50;
            color: white;
        }
        .status.paused {
            background: #FFC107;
            color: black;
        }
        .status.stopped {
            background: #f44336;
            color: white;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #eee;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.5s ease;
        }
        .footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.8;
        }
        .hardware-info {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 PyTorch Training Dashboard</h1>
            <p class="subtitle">French LLM V2 - RTX 5080 | R7 9700X OC 105W</p>
        </header>

        <div class="hardware-info">
            <strong>🖥️ System:</strong> RTX 5080 16GB | R7 9700X OC 105W | 64GB DDR5 | NVMe PCIe 4.0<br>
            <strong>⚡ Optimisations:</strong> AMP | Gradient Checkpointing | torch.compile | Flash Attention
        </div>

        <div class="grid">
            <div class="card">
                <h3>📊 Training Status</h3>
                <div id="training-status" class="status"></div>
                <div class="metric">
                    <div class="metric-label">Current Step</div>
                    <div class="metric-value" id="step">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Training Loss</div>
                    <div class="metric-value" id="loss">-</div>
                </div>
            </div>

            <div class="card">
                <h3>💾 GPU Memory</h3>
                <div class="metric">
                    <div class="metric-label">Used</div>
                    <div class="metric-value" id="gpu-used">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Allocated</div>
                    <div class="metric-value" id="gpu-alloc">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Reserved</div>
                    <div class="metric-value" id="gpu-reserved">-</div>
                </div>
            </div>

            <div class="card">
                <h3>⏱️ Timing</h3>
                <div class="metric">
                    <div class="metric-label">Steps/sec</div>
                    <div class="metric-value" id="throughput">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Elapsed Time</div>
                    <div class="metric-value" id="elapsed">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">ETA</div>
                    <div class="metric-value" id="eta">-</div>
                </div>
            </div>

            <div class="card">
                <h3>🔧 Learning Rate</h3>
                <div class="metric">
                    <div class="metric-label">Current LR</div>
                    <div class="metric-value" id="lr">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Warmup</div>
                    <div class="metric-value" id="warmup">-</div>
                </div>
            </div>

            <div class="card">
                <h3>📈 Progress</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill" style="width: 0%">0%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Completion</div>
                    <div class="metric-value" id="completion">0 / 200000</div>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">📉 Loss Curve</div>
            <canvas id="lossChart"></canvas>
        </div>

        <div class="footer">
            <p>Last updated: <span id="timestamp">-</span> | Auto-refresh every 2 seconds</p>
        </div>
    </div>

    <script>
        let lossChart = null;
        const updateInterval = 2000; // 2 secondes

        async function fetchMetrics() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();
                updateDashboard(data);
            } catch (e) {
                console.error('Erreur:', e);
                setTimeout(fetchMetrics, updateInterval);
            }
        }

        function updateDashboard(data) {
            // Status
            document.getElementById('training-status').textContent = data.status || 'Unknown';
            document.getElementById('training-status').className = 'status ' + (data.status || 'stopped').toLowerCase();

            // Métriques
            document.getElementById('step').textContent = (data.step || 0).toLocaleString();
            document.getElementById('loss').textContent = (data.loss || 0).toFixed(4);
            document.getElementById('lr').textContent = (data.lr || 0).toExponential(2);

            // GPU
            document.getElementById('gpu-used').textContent = (data.gpu_used_gb || 0).toFixed(1) + ' GB';
            document.getElementById('gpu-alloc').textContent = (data.gpu_alloc_gb || 0).toFixed(1) + ' GB';
            document.getElementById('gpu-reserved').textContent = (data.gpu_reserved_gb || 0).toFixed(1) + ' GB';

            // Timing
            document.getElementById('throughput').textContent = (data.steps_per_sec || 0).toFixed(2);
            document.getElementById('elapsed').textContent = formatTime(data.elapsed_sec || 0);
            document.getElementById('eta').textContent = formatTime(data.eta_sec || 0);

            // Progress
            const progress = Math.min(100, ((data.step || 0) / 200000) * 100);
            document.getElementById('progress-fill').style.width = progress + '%';
            document.getElementById('progress-fill').textContent = progress.toFixed(1) + '%';
            document.getElementById('completion').textContent = (data.step || 0).toLocaleString() + ' / 200,000';

            // Timestamp
            document.getElementById('timestamp').textContent = new Date().toLocaleTimeString('fr-FR');

            // Charts
            updateChart(data.loss_history || []);
        }

        function formatTime(seconds) {
            if (seconds < 0) return '-';
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return `${h}h ${m}m ${s}s`;
        }

        function updateChart(losses) {
            const ctx = document.getElementById('lossChart').getContext('2d');
            
            if (lossChart) {
                lossChart.data.labels = losses.map((_, i) => i);
                lossChart.data.datasets[0].data = losses;
                lossChart.update('none');
            } else {
                lossChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: losses.map((_, i) => i),
                        datasets: [{
                            label: 'Training Loss',
                            data: losses,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0,
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            y: {
                                beginAtZero: false,
                                ticks: { color: '#666' },
                                grid: { color: '#eee' }
                            },
                            x: {
                                ticks: { color: '#666' },
                                grid: { color: '#eee' }
                            }
                        }
                    }
                });
            }
        }

        // Auto-refresh
        fetchMetrics();
        setInterval(fetchMetrics, updateInterval);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Affiche le dashboard"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def get_metrics():
    """Retourne les métriques d'entraînement"""
    try:
        metrics_file = LATEST_RUN / "metrics.jsonl"
        
        if not metrics_file.exists():
            return jsonify({
                "status": "waiting",
                "step": 0,
                "loss": 0,
                "gpu_used_gb": 0,
                "loss_history": []
            })
        
        # Lire les dernières métriques
        losses = []
        last_metric = None
        
        with open(metrics_file) as f:
            for line in f:
                if line.strip():
                    metric = json.loads(line)
                    last_metric = metric
                    if 'loss' in metric:
                        losses.append(metric['loss'])
        
        # Infos GPU
        gpu_stats = {
            "gpu_used_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_alloc_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        }
        
        return jsonify({
            "status": "running",
            "step": last_metric.get('step', 0) if last_metric else 0,
            "loss": last_metric.get('loss', 0) if last_metric else 0,
            "lr": 1e-4,  # À obtenir du trainer
            "steps_per_sec": 2.5,  # À calculer
            "elapsed_sec": 3600,  # À calculer
            "eta_sec": 100000,  # À calculer
            "loss_history": losses[-100:],  # Derniers 100 steps
            **gpu_stats
        })
    except Exception as e:
        print(f"Erreur metrics: {e}")
        return jsonify({"status": "error", "error": str(e)})

@app.route('/api/status')
def get_status():
    """Retourne le statut du training"""
    return jsonify({
        "hardware": {
            "gpu": "RTX 5080",
            "gpu_vram": "16GB",
            "cpu": "R7 9700X OC 105W",
            "ram": "64GB DDR5",
            "storage": "NVMe PCIe 4.0",
        },
        "optimizations": [
            "Mixed Precision (AMP)",
            "Gradient Checkpointing",
            "torch.compile",
            "Flash Attention",
            "Persistent Workers (4)",
            "Pin Memory"
        ]
    })

def main():
    print("\n" + "="*70)
    print("🎛️  PYTORCH TRAINING DASHBOARD")
    print("="*70)
    print(f"\n📍 URL: http://localhost:5000")
    print(f"📊 Monitoring: {LATEST_RUN}")
    print(f"\n✓ Dashboard prêt!")
    print(f"✓ Auto-refresh: 2 secondes")
    print(f"✓ Hardware: RTX 5080 + R7 9700X OC 105W")
    print("\nAppuyez sur Ctrl+C pour arrêter\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )

if __name__ == '__main__':
    main()
