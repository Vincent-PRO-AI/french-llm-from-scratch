#!/usr/bin/env python3
"""
🚀 LANCEMENT ENTRAÎNEMENT V2 OPTIMISÉ - RTX 5080 + R7 9700X
Configuration matériel prise en compte:
- GPU: RTX 5080 16GB VRAM
- CPU: R7 9700X OC 105W (8 cores)
- RAM: 64GB DDR5
- Storage: NVMe PCIe 4.0
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent

# ============================================================================
# CONFIGURATION ENV
# ============================================================================

# CUDA optimisations pour RTX 5080
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
os.environ['TORCH_CUDNN_ENABLED'] = '1'
os.environ['TORCH_BACKENDS_CUDNN_BENCHMARK'] = '1'

# CPU optimisations pour R7 9700X
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_NUM_THREADS'] = '8'

# Logging
os.environ['TORCH_PROFILER_ENABLED'] = '1'

class V2Trainer:
    def __init__(self):
        self.root = ROOT
        self.data_dir = ROOT / "data_clean"
        self.model_dir = ROOT / "trained_models"
        self.checkpoint_dir = self.model_dir / "runs" / "french_v2_optimized_rtx5080"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.processes = []
        
    def print_banner(self):
        """Affiche le banner de démarrage"""
        print("\n" + "="*80)
        print("🚀 ENTRAÎNEMENT V2 - OPTIMISÉ RTX 5080 + R7 9700X")
        print("="*80)
        print(f"\n📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Répertoire: {self.root}")
        print(f"\n🖥️  HARDWARE:")
        print(f"   GPU: RTX 5080 16GB VRAM")
        print(f"   CPU: R7 9700X OC 105W (8 cores, 12 threads)")
        print(f"   RAM: 64GB DDR5 (88GB/s bandwidth)")
        print(f"   Storage: NVMe PCIe 4.0 (7GB/s theoretical)")
        print(f"\n⚙️  OPTIMISATIONS ACTIVÉES:")
        print(f"   ✓ Mixed Precision (AMP) - FP16 + FP32")
        print(f"   ✓ Gradient Checkpointing - -30% mémoire")
        print(f"   ✓ torch.compile - +30% perfs")
        print(f"   ✓ Flash Attention - RTX 5080 compatible")
        print(f"   ✓ CUDNN Benchmark - auto-tuning")
        print(f"   ✓ Persistent Workers - DDR5 optimisé")
        print(f"   ✓ Pin Memory - NVMe ↔ RAM")
        print(f"\n📊 CONFIGURATION ENTRAÎNEMENT:")
        print(f"   Batch Size: 16")
        print(f"   Gradient Accumulation: 4 (batch effectif: 64)")
        print(f"   Learning Rate: 1e-4 (avec warmup)")
        print(f"   Max Steps: 200,000")
        print(f"   Checkpoints: Tous les 2,500 steps")
        print(f"   Evaluation: Tous les 1,000 steps")
        print("="*80 + "\n")
    
    def verify_data(self):
        """Vérifie que les données sont présentes"""
        print("\n📂 VÉRIFICATION DES DONNÉES...")
        
        essential_data = [
            self.data_dir / "v2_tokenized",
            self.data_dir / "conversations_mega_train.txt",
            self.data_dir / "conversations_mega_test.txt",
        ]
        
        all_exist = True
        for path in essential_data:
            if path.exists():
                if path.is_file():
                    size = path.stat().st_size / 1e9
                    print(f"   ✓ {path.name}: {size:.2f}GB")
                else:
                    # Dossier
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1e9
                    print(f"   ✓ {path.name}/: {size:.2f}GB")
            else:
                print(f"   ✗ {path.name}: MANQUANT")
                all_exist = False
        
        return all_exist
    
    def verify_dependencies(self):
        """Vérifie que les dépendances sont installées"""
        print("\n📦 VÉRIFICATION DES DÉPENDANCES...")
        
        required = ['torch', 'transformers', 'accelerate', 'flask']
        
        for pkg in required:
            try:
                __import__(pkg)
                print(f"   ✓ {pkg}")
            except ImportError:
                print(f"   ✗ {pkg}: À installer")
                return False
        
        return True
    
    def launch_dashboard(self):
        """Lance le dashboard Flask en background"""
        print("\n🎛️  LANCEMENT DASHBOARD...")
        
        dashboard_script = self.root / "dashboard_pytorch.py"
        
        if not dashboard_script.exists():
            print(f"   ✗ Dashboard non trouvé: {dashboard_script}")
            return None
        
        try:
            proc = subprocess.Popen(
                [sys.executable, str(dashboard_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.root)
            )
            self.processes.append(proc)
            time.sleep(2)  # Attendre que Flask démarre
            print(f"   ✓ Dashboard lancé (PID: {proc.pid})")
            print(f"   📍 Accès: http://localhost:5000")
            return proc
        except Exception as e:
            print(f"   ✗ Erreur dashboard: {e}")
            return None
    
    def launch_training(self):
        """Lance l'entraînement PyTorch"""
        print("\n🔥 LANCEMENT ENTRAÎNEMENT...")
        
        # Utiliser le script d'optimisation
        optimizer_script = self.root / "launch_optimized_v2_training.py"
        
        if not optimizer_script.exists():
            print(f"   ✗ Script optimizer non trouvé: {optimizer_script}")
            return False
        
        try:
            print(f"   Exécution: {optimizer_script}")
            proc = subprocess.run(
                [sys.executable, str(optimizer_script)],
                cwd=str(self.root),
                capture_output=False,
                text=True
            )
            return proc.returncode == 0
        except Exception as e:
            print(f"   ✗ Erreur: {e}")
            return False
    
    def generate_config(self):
        """Génère les fichiers de configuration V2"""
        print("\n⚙️  GÉNÉRATION CONFIGURATION...")
        
        config = {
            "model": {
                "vocab_size": 32000,
                "hidden_size": 1024,
                "num_hidden_layers": 18,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
            },
            "training": {
                "batch_size": 16,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 64,
                "learning_rate": 1e-4,
                "warmup_steps": 500,
                "max_steps": 200000,
            },
            "hardware": {
                "gpu": "RTX 5080 (16GB)",
                "cpu": "R7 9700X OC 105W",
                "ram": "64GB DDR5",
                "storage": "NVMe PCIe 4.0",
            },
            "optimizations": [
                "mixed_precision",
                "gradient_checkpointing",
                "torch_compile",
                "flash_attention",
                "cudnn_benchmark",
                "persistent_workers",
                "pin_memory",
            ]
        }
        
        config_path = self.checkpoint_dir / "v2_config.json"
        
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✓ Config: {config_path}")
        return config_path
    
    def cleanup(self):
        """Arrête les processus proprement"""
        print("\n🛑 ARRÊT DES PROCESSUS...")
        
        for proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"   ✓ Processus {proc.pid} arrêté")
            except:
                proc.kill()
                print(f"   ✓ Processus {proc.pid} tué")
    
    def run(self):
        """Lance le pipeline complet"""
        try:
            self.print_banner()
            
            # Vérifications
            if not self.verify_data():
                print("\n❌ Données manquantes!")
                return False
            
            if not self.verify_dependencies():
                print("\n⚠️  Dépendances manquantes - installation recommandée")
            
            # Génération config
            self.generate_config()
            
            # Lancer le dashboard
            dashboard_proc = self.launch_dashboard()
            
            if dashboard_proc:
                print("\n✅ SYSTÈME PRÊT!")
                print(f"   📊 Dashboard: http://localhost:5000")
                print(f"   🔥 Entraînement: Prêt à démarrer")
                print(f"\nPour lancer l'entraînement, exécuter dans un autre terminal:")
                print(f"   python scripts/train_subtitles_transformer.py \\")
                print(f"     --config {self.checkpoint_dir}/v2_config.json \\")
                print(f"     --output-dir {self.checkpoint_dir}")
                print(f"\nLe dashboard affichera les métriques en temps réel.")
                
                # Garder le dashboard actif
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n\n⏹️  Arrêt par l'utilisateur...")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Arrêt par l'utilisateur")
            return False
        finally:
            self.cleanup()

def main():
    trainer = V2Trainer()
    success = trainer.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
