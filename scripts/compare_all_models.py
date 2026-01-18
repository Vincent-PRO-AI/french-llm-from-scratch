#!/usr/bin/env python3
"""
Script pour analyser et comparer tous les modèles entraînés.
Génère un rapport complet avec métriques, dates et recommandations.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import argparse


class ModelAnalyzer:
    """Analyse tous les checkpoints disponibles"""
    
    def __init__(self, base_path: str = "trained_models"):
        self.base_path = Path(base_path)
        self.runs: List[Dict] = []
        
    def analyze_all_runs(self) -> List[Dict]:
        """Analyse tous les runs dans trained_models/runs/"""
        runs_dir = self.base_path / "runs"
        
        for metrics_file in runs_dir.glob("*/metrics.jsonl"):
            run_info = self._analyze_single_run(metrics_file)
            if run_info:
                self.runs.append(run_info)
        
        # Trier par qualité (loss la plus faible)
        self.runs.sort(key=lambda x: x['final_loss'])
        return self.runs
    
    def _analyze_single_run(self, metrics_file: Path) -> Dict:
        """Analyse un run individuel"""
        run_name = metrics_file.parent.name
        
        try:
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return None
            
            # Parse première et dernière ligne
            first_metric = json.loads(lines[0])
            last_metric = json.loads(lines[-1])
            
            # Extraire métriques de validation
            val_losses = []
            train_losses = []
            
            for line in lines:
                data = json.loads(line)
                if data['split'] == 'val':
                    val_losses.append((data['step'], data['loss']))
                elif data['split'] == 'train':
                    train_losses.append((data['step'], data['loss']))
            
            # Trouver le checkpoint le plus récent
            checkpoint_files = list(metrics_file.parent.glob("checkpoint*.pt"))
            latest_checkpoint = None
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            
            # Statistiques
            final_loss = val_losses[-1][1] if val_losses else last_metric['loss']
            first_loss = val_losses[0][1] if val_losses else first_metric['loss']
            best_loss = min(v[1] for v in val_losses) if val_losses else final_loss
            improvement = ((first_loss - final_loss) / first_loss * 100) if first_loss > 0 else 0
            
            return {
                'name': run_name,
                'metrics_file': str(metrics_file),
                'checkpoint_file': str(latest_checkpoint) if latest_checkpoint else None,
                'final_step': last_metric['step'],
                'final_loss': final_loss,
                'first_loss': first_loss,
                'best_loss': best_loss,
                'improvement_pct': improvement,
                'total_steps': len(lines),
                'val_steps': len(val_losses),
                'train_steps': len(train_losses),
                'timestamp': last_metric.get('timestamp', 0),
                'date': datetime.fromtimestamp(last_metric.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M'),
            }
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"⚠️  Erreur parsing {run_name}: {e}")
            return None
    
    def print_ranking(self, top_n: int = 10):
        """Affiche le classement des meilleurs modèles"""
        print("\n" + "="*100)
        print(f"🏆 TOP {top_n} MODÈLES PAR QUALITÉ (Loss de validation)")
        print("="*100)
        
        header = f"{'Rang':<6} {'Run Name':<45} {'Step':>10} {'Loss':>8} {'Amélioration':>12} {'Date':<20}"
        print(header)
        print("-"*100)
        
        medals = ['🥇', '🥈', '🥉'] + ['  '] * 97
        
        for i, run in enumerate(self.runs[:top_n]):
            medal = medals[i] if i < len(medals) else '  '
            print(f"{medal} {i+1:<3} {run['name']:<45} {run['final_step']:>10,} "
                  f"{run['final_loss']:>8.4f} {run['improvement_pct']:>11.1f}% {run['date']:<20}")
    
    def print_detailed_report(self, run_name: str):
        """Affiche un rapport détaillé pour un run spécifique"""
        run = next((r for r in self.runs if r['name'] == run_name), None)
        if not run:
            print(f"❌ Run '{run_name}' non trouvé")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 RAPPORT DÉTAILLÉ: {run['name']}")
        print(f"{'='*80}")
        print(f"Checkpoint:     {run['checkpoint_file'] or 'N/A'}")
        print(f"Metrics file:   {run['metrics_file']}")
        print(f"\nTraining:")
        print(f"  Final step:   {run['final_step']:,}")
        print(f"  Total steps:  {run['total_steps']:,} (train: {run['train_steps']}, val: {run['val_steps']})")
        print(f"  Date:         {run['date']}")
        print(f"\nPerformance:")
        print(f"  First loss:   {run['first_loss']:.4f}")
        print(f"  Final loss:   {run['final_loss']:.4f}")
        print(f"  Best loss:    {run['best_loss']:.4f}")
        print(f"  Improvement:  {run['improvement_pct']:.1f}%")
        
        # Statut
        if run['final_loss'] < 2.0:
            status = "✅ EXCELLENT - Production Ready"
        elif run['final_loss'] < 4.0:
            status = "🟢 BON - Utilisable"
        elif run['final_loss'] < 6.0:
            status = "🟡 MOYEN - Nécessite amélioration"
        else:
            status = "🔴 FAIBLE - Problème d'entraînement"
        
        print(f"\nStatut: {status}")
    
    def generate_markdown_report(self, output_file: str = "MODEL_COMPARISON.md"):
        """Génère un rapport markdown complet"""
        with open(output_file, 'w') as f:
            f.write("# 📊 Comparaison Automatique des Modèles\n\n")
            f.write(f"**Généré le:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Nombre de runs:** {len(self.runs)}\n\n")
            
            f.write("## 🏆 Classement Global\n\n")
            f.write("| Rang | Run Name | Step Final | Loss | Amélioration | Date |\n")
            f.write("|------|----------|------------|------|--------------|------|\n")
            
            for i, run in enumerate(self.runs[:20], 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
                f.write(f"| {medal} {i} | {run['name']} | {run['final_step']:,} | "
                       f"{run['final_loss']:.4f} | {run['improvement_pct']:.1f}% | {run['date']} |\n")
            
            f.write("\n## 📈 Statistiques Globales\n\n")
            
            if self.runs:
                best_run = self.runs[0]
                worst_run = self.runs[-1]
                avg_loss = sum(r['final_loss'] for r in self.runs) / len(self.runs)
                
                f.write(f"- **Meilleure loss:** {best_run['final_loss']:.4f} ({best_run['name']})\n")
                f.write(f"- **Pire loss:** {worst_run['final_loss']:.4f} ({worst_run['name']})\n")
                f.write(f"- **Loss moyenne:** {avg_loss:.4f}\n")
                f.write(f"- **Runs < 2.0 loss:** {sum(1 for r in self.runs if r['final_loss'] < 2.0)}\n")
                f.write(f"- **Runs < 4.0 loss:** {sum(1 for r in self.runs if r['final_loss'] < 4.0)}\n")
            
            f.write("\n## 🎯 Recommandations\n\n")
            
            if self.runs and self.runs[0]['final_loss'] < 2.0:
                best = self.runs[0]
                f.write(f"### ✅ Production Ready\n\n")
                f.write(f"Le modèle **{best['name']}** (loss: {best['final_loss']:.4f}) "
                       f"est recommandé pour déploiement en production.\n\n")
                f.write(f"**Checkpoint:** `{best['checkpoint_file']}`\n\n")
            
            f.write("\n---\n\n")
            f.write("*Rapport généré automatiquement par compare_all_models.py*\n")
        
        print(f"✅ Rapport markdown généré: {output_file}")
    
    def export_json(self, output_file: str = "models_analysis.json"):
        """Exporte les résultats en JSON"""
        with open(output_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'total_runs': len(self.runs),
                'runs': self.runs
            }, f, indent=2)
        
        print(f"✅ Données exportées en JSON: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare tous les modèles entraînés")
    parser.add_argument('--base-path', default='trained_models', 
                       help='Chemin vers le dossier trained_models')
    parser.add_argument('--top', type=int, default=10, 
                       help='Nombre de modèles à afficher dans le top')
    parser.add_argument('--detail', type=str, 
                       help='Afficher le détail pour un run spécifique')
    parser.add_argument('--markdown', action='store_true', 
                       help='Générer un rapport markdown')
    parser.add_argument('--json', action='store_true', 
                       help='Exporter en JSON')
    parser.add_argument('--output', default='MODEL_COMPARISON.md', 
                       help='Fichier de sortie pour le rapport markdown')
    
    args = parser.parse_args()
    
    print("🔍 Analyse de tous les modèles...")
    
    analyzer = ModelAnalyzer(args.base_path)
    analyzer.analyze_all_runs()
    
    if not analyzer.runs:
        print("❌ Aucun run trouvé")
        return
    
    print(f"✅ {len(analyzer.runs)} runs analysés\n")
    
    # Afficher le classement
    analyzer.print_ranking(args.top)
    
    # Détail d'un run spécifique
    if args.detail:
        analyzer.print_detailed_report(args.detail)
    
    # Générer rapport markdown
    if args.markdown:
        analyzer.generate_markdown_report(args.output)
    
    # Export JSON
    if args.json:
        analyzer.export_json()


if __name__ == "__main__":
    main()
