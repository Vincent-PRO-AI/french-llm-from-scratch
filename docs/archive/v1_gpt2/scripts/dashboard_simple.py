#!/usr/bin/env python3
"""Simple web dashboard pour visualiser le training en temps réel."""

import json
import threading
import time
from pathlib import Path
from datetime import datetime
from collections import deque

try:
    from flask import Flask, render_template_string, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask not installed - using simple HTTP server instead")

# Configuration
METRICS_FILE = Path("trained_models/runs/french_v2_phase2a_final/metrics.jsonl")
MAX_POINTS = 500  # Garder les 500 derniers points pour le graphique

app = Flask(__name__) if FLASK_AVAILABLE else None
metrics_cache = {"steps": [], "losses": [], "speeds": [], "etas": [], "last_update": None}
metrics_lock = threading.Lock()


def load_metrics():
    """Charge les métriques depuis le fichier JSONL."""
    global metrics_cache
    
    if not METRICS_FILE.exists():
        return
    
    with metrics_lock:
        steps = deque(maxlen=MAX_POINTS)
        losses = deque(maxlen=MAX_POINTS)
        speeds = deque(maxlen=MAX_POINTS)
        etas = deque(maxlen=MAX_POINTS)
        
        try:
            with open(METRICS_FILE, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        steps.append(data.get("step", 0))
                        losses.append(round(float(data.get("loss", 0)), 2))
                        speeds.append(round(float(data.get("speed_steps_per_sec", 0)), 4))
                        etas.append(round(float(data.get("eta_hours", 0)), 2))
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            metrics_cache = {
                "steps": list(steps),
                "losses": list(losses),
                "speeds": list(speeds),
                "etas": list(etas),
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Erreur chargement métriques: {e}")


def update_metrics_background():
    """Met à jour les métriques en arrière-plan."""
    while True:
        load_metrics()
        time.sleep(5)  # Rafraîchir toutes les 5 secondes


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>French LLM Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
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
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-card h3 {
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        
        .metric-unit {
            font-size: 0.8em;
            color: #999;
            margin-left: 5px;
        }
        
        .metric-subtext {
            font-size: 0.85em;
            color: #666;
            margin-top: 8px;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        
        .chart-container h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.2em;
        }
        
        .chart-wrapper {
            position: relative;
            height: 300px;
        }
        
        .status {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        
        .status h2 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .status-item {
            padding: 10px;
            background: #f5f5f5;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }
        
        .status-label {
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
        }
        
        .status-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        
        .refresh-time {
            text-align: center;
            color: white;
            font-size: 0.9em;
            margin-top: 20px;
        }
        
        .progress-bar {
            background: #e0e0e0;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 French LLM V2 Training</h1>
            <p>Phase 2A: Steps 100k → 110k | Dataset: 100% French | GPU: RTX 5080</p>
        </header>
        
        <div class="metrics-grid" id="metrics"></div>
        
        <div class="status">
            <h2>📊 Training Status</h2>
            <div class="status-grid" id="status"></div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <h2>📉 Loss Curve</h2>
                <div class="chart-wrapper">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>
            <div class="chart-container">
                <h2>⚡ Training Speed</h2>
                <div class="chart-wrapper">
                    <canvas id="speedChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="refresh-time">
            Auto-refresh every 5 seconds | Last update: <span id="lastUpdate">--:--:--</span>
        </div>
    </div>
    
    <script>
        let lossChart = null;
        let speedChart = null;
        
        async function updateDashboard() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                
                // Update metrics cards
                const currentStep = data.steps[data.steps.length - 1] || 100000;
                const currentLoss = data.losses[data.losses.length - 1] || 0;
                const currentSpeed = data.speeds[data.speeds.length - 1] || 0;
                const currentEta = data.etas[data.etas.length - 1] || 0;
                
                const progress = ((currentStep - 100000) / 10000) * 100;
                
                document.getElementById('metrics').innerHTML = `
                    <div class="metric-card">
                        <h3>Current Step</h3>
                        <div class="metric-value">${currentStep.toLocaleString()}<span class="metric-unit">/110,000</span></div>
                        <div class="metric-subtext">${progress.toFixed(1)}% Complete</div>
                    </div>
                    <div class="metric-card">
                        <h3>Loss</h3>
                        <div class="metric-value">${currentLoss.toFixed(2)}</div>
                        <div class="metric-subtext">${data.losses.length > 1 ? (currentLoss < data.losses[data.losses.length-2] ? '📉 Decreasing' : '📈 Increasing') : '🆕 Starting'}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Speed</h3>
                        <div class="metric-value">${currentSpeed.toFixed(3)}<span class="metric-unit">steps/s</span></div>
                        <div class="metric-subtext">~${(currentSpeed * 3600).toFixed(0)} steps/hour</div>
                    </div>
                    <div class="metric-card">
                        <h3>ETA</h3>
                        <div class="metric-value">${currentEta.toFixed(1)}<span class="metric-unit">h</span></div>
                        <div class="metric-subtext">~${Math.floor(currentEta)}h ${Math.round((currentEta % 1) * 60)}m remaining</div>
                    </div>
                `;
                
                document.getElementById('status').innerHTML = `
                    <div class="status-item">
                        <div class="status-label">Total Steps Completed</div>
                        <div class="status-value">${(currentStep - 100000).toLocaleString()}</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Min Loss</div>
                        <div class="status-value">${Math.min(...data.losses).toFixed(2)}</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Avg Speed</div>
                        <div class="status-value">${(data.speeds.reduce((a,b)=>a+b,0)/data.speeds.length).toFixed(3)} steps/s</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Data Points</div>
                        <div class="status-value">${data.steps.length}</div>
                    </div>
                `;
                
                document.getElementById('progressFill').style.width = Math.min(100, progress) + '%';
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                
                // Update charts
                updateCharts(data);
                
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }
        
        function updateCharts(data) {
            const ctx1 = document.getElementById('lossChart');
            const ctx2 = document.getElementById('speedChart');
            
            const commonOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: false
                    }
                }
            };
            
            if (lossChart) lossChart.destroy();
            lossChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: data.steps,
                    datasets: [{
                        label: 'Loss',
                        data: data.losses,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 1,
                        pointHoverRadius: 4
                    }]
                },
                options: commonOptions
            });
            
            if (speedChart) speedChart.destroy();
            speedChart = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: data.steps,
                    datasets: [{
                        label: 'Speed (steps/sec)',
                        data: data.speeds,
                        borderColor: '#764ba2',
                        backgroundColor: 'rgba(118, 75, 162, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 1,
                        pointHoverRadius: 4
                    }]
                },
                options: commonOptions
            });
        }
        
        // Initial update and auto-refresh
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""


if FLASK_AVAILABLE:
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/api/metrics')
    def api_metrics():
        with metrics_lock:
            return jsonify(metrics_cache)


def main():
    global app
    
    # Start metrics loader thread
    loader_thread = threading.Thread(target=update_metrics_background, daemon=True)
    loader_thread.start()
    
    if FLASK_AVAILABLE:
        print("\n" + "="*60)
        print("✅ Dashboard Flask lancé!")
        print("="*60)
        print("\n🌐 Accès via:")
        print("   http://localhost:5000")
        print("\n📊 Interface temps réel avec:")
        print("   • Courbes de loss et vitesse")
        print("   • Statut du training")
        print("   • Progression en temps réel")
        print("\n" + "="*60 + "\n")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Flask non disponible")
        print("Installation: pip install flask")


if __name__ == '__main__':
    main()
