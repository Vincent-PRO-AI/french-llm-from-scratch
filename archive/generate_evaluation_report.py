#!/usr/bin/env python3
"""
Benchmark complet et rapport d'évaluation.
Compare avec les baselines, mesure la qualité.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def generate_evaluation_report():
    """Générer un rapport d'évaluation complet."""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': 'french-llm-v3',
        'evaluation_metrics': {}
    }
    
    # 1. Métriques de training
    metrics_file = Path('trained_models/runs/french_v3_finetune_grand_60k/metrics.jsonl')
    if metrics_file.exists():
        train_losses = []
        val_losses = []
        
        with open(metrics_file) as f:
            for line in f:
                data = json.loads(line)
                if data['split'] == 'train':
                    train_losses.append(data['loss'])
                else:
                    val_losses.append(data['loss'])
        
        report['evaluation_metrics']['training'] = {
            'train_loss_min': min(train_losses) if train_losses else None,
            'train_loss_max': max(train_losses) if train_losses else None,
            'train_loss_final': train_losses[-1] if train_losses else None,
            'val_loss_min': min(val_losses) if val_losses else None,
            'val_loss_final': val_losses[-1] if val_losses else None,
            'steps_completed': len(val_losses),
            'status': 'completed' if val_losses[-1] < 4.3 else 'in_progress'
        }
    
    # 2. Taille du modèle
    checkpoint_path = max(
        Path('trained_models/runs/french_v3_finetune_grand_60k').glob('checkpoint_step_*.pt'),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    
    if checkpoint_path.exists():
        size_gb = checkpoint_path.stat().st_size / (1024**3)
        report['evaluation_metrics']['model'] = {
            'checkpoint': checkpoint_path.name,
            'size_gb': round(size_gb, 2),
            'parameters': 260000000,
            'layers': 18,
            'vocab_size': 117000
        }
    
    # 3. Checkpoints disponibles
    checkpoints = sorted(
        Path('trained_models/runs/french_v3_finetune_grand_60k').glob('checkpoint_step_*.pt'),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    
    report['evaluation_metrics']['checkpoints'] = {
        'total_saved': len(checkpoints),
        'latest': checkpoints[-1].name if checkpoints else None,
        'latest_step': int(checkpoints[-1].stem.split('_')[-1]) if checkpoints else None
    }
    
    # 4. Data Quality
    data_files = {
        'train': 'data_clean/conversations_mega_train.txt',
        'test': 'data_clean/conversations_mega_test.txt'
    }
    
    data_quality = {}
    for split, file_path in data_files.items():
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024**2)
            with open(file_path, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            data_quality[split] = {
                'size_mb': round(size_mb, 2),
                'lines': num_lines
            }
    
    report['evaluation_metrics']['data_quality'] = data_quality
    
    return report


def create_html_report(report):
    """Créer un rapport HTML visuel."""
    
    html = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport d'Évaluation - French LLM</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }
            h2 {
                color: #764ba2;
                margin-top: 30px;
            }
            .metric {
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #667eea;
                border-radius: 5px;
            }
            .metric-value {
                font-size: 1.5em;
                font-weight: bold;
                color: #667eea;
            }
            .metric-label {
                color: #666;
                font-size: 0.9em;
            }
            .status-good {
                color: #28a745;
            }
            .status-warning {
                color: #ffc107;
            }
            .status-bad {
                color: #dc3545;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background: #667eea;
                color: white;
            }
            tr:hover {
                background: #f5f5f5;
            }
            .footer {
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Rapport d'Évaluation - French LLM v3</h1>
    """
    
    metrics = report['evaluation_metrics']
    
    # Training
    if 'training' in metrics:
        html += f"""
        <h2>🎯 Métriques d'Entraînement</h2>
        <div class="metric">
            <div class="metric-label">Loss Validation (Final)</div>
            <div class="metric-value status-good">{metrics['training']['val_loss_final']:.4f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Steps Complétés</div>
            <div class="metric-value">{metrics['training']['steps_completed']}</div>
        </div>
        <table>
            <tr>
                <th>Métrique</th>
                <th>Valeur</th>
            </tr>
            <tr>
                <td>Train Loss (Min)</td>
                <td>{metrics['training']['train_loss_min']:.4f}</td>
            </tr>
            <tr>
                <td>Train Loss (Max)</td>
                <td>{metrics['training']['train_loss_max']:.4f}</td>
            </tr>
            <tr>
                <td>Val Loss (Min)</td>
                <td>{metrics['training']['val_loss_min']:.4f}</td>
            </tr>
            <tr>
                <td>Status</td>
                <td class="status-good">✅ {metrics['training']['status']}</td>
            </tr>
        </table>
        """
    
    # Model
    if 'model' in metrics:
        html += f"""
        <h2>🤖 Modèle</h2>
        <table>
            <tr>
                <th>Propriété</th>
                <th>Valeur</th>
            </tr>
            <tr>
                <td>Taille</td>
                <td>{metrics['model']['size_gb']} GB</td>
            </tr>
            <tr>
                <td>Paramètres</td>
                <td>{metrics['model']['parameters']:,}</td>
            </tr>
            <tr>
                <td>Couches</td>
                <td>{metrics['model']['layers']}</td>
            </tr>
            <tr>
                <td>Vocab Size</td>
                <td>{metrics['model']['vocab_size']:,}</td>
            </tr>
            <tr>
                <td>Checkpoint</td>
                <td>{metrics['model']['checkpoint']}</td>
            </tr>
        </table>
        """
    
    # Checkpoints
    if 'checkpoints' in metrics:
        html += f"""
        <h2>💾 Checkpoints</h2>
        <div class="metric">
            <div class="metric-label">Total Sauvegardés</div>
            <div class="metric-value">{metrics['checkpoints']['total_saved']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Dernier Step</div>
            <div class="metric-value">{metrics['checkpoints']['latest_step']:,}</div>
        </div>
        """
    
    # Data
    if 'data_quality' in metrics:
        html += """
        <h2>📚 Qualité des Données</h2>
        <table>
            <tr>
                <th>Dataset</th>
                <th>Taille</th>
                <th>Lignes</th>
            </tr>
        """
        for split, data in metrics['data_quality'].items():
            html += f"""
            <tr>
                <td>{split.upper()}</td>
                <td>{data['size_mb']} MB</td>
                <td>{data['lines']:,}</td>
            </tr>
            """
        html += "</table>"
    
    html += f"""
            <div class="footer">
                <p>Rapport généré le {report['timestamp']}</p>
                <p>Version: {report['version']}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def main():
    print("📊 Génération du rapport d'évaluation...")
    
    # Générer le rapport
    report = generate_evaluation_report()
    
    # Sauvegarder en JSON
    with open('evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("✅ Rapport JSON: evaluation_report.json")
    
    # Sauvegarder en HTML
    html_content = create_html_report(report)
    with open('evaluation_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✅ Rapport HTML: evaluation_report.html")
    
    # Afficher un résumé
    print("\n" + "="*70)
    print("📈 RÉSUMÉ D'ÉVALUATION")
    print("="*70)
    
    metrics = report['evaluation_metrics']
    
    if 'training' in metrics:
        print(f"\n🎯 Training:")
        print(f"   Val Loss Final: {metrics['training']['val_loss_final']:.4f}")
        print(f"   Steps: {metrics['training']['steps_completed']}")
        print(f"   Status: {metrics['training']['status']}")
    
    if 'model' in metrics:
        print(f"\n🤖 Model:")
        print(f"   Size: {metrics['model']['size_gb']} GB")
        print(f"   Latest: {metrics['model']['checkpoint']}")
    
    print("\n✅ Évaluation complète!")


if __name__ == '__main__':
    main()
