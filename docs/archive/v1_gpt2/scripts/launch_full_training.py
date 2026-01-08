#!/usr/bin/env python3
"""
🚀 FRENCH LLM V2 - TRAINING LAUNCHER COMPLET
============================================
Lance automatiquement:
1. Interface PyTorch Dashboard (port 5174)
2. Training 100k steps avec dataset 100% français
3. Tokenization dataset si besoin

Usage: python3 launch_full_training.py
"""
import subprocess
import sys
import time
from pathlib import Path
import os

def main():
    print("\n" + "="*70)
    print("🚀 FRENCH LLM V2 - LANCEMENT COMPLET")
    print("="*70)
    
    # 1. Démarrer Docker Dashboard
    print("\n1️⃣  Démarrage Dashboard PyTorch...")
    try:
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            cwd=Path(__file__).parent,
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            print("   ✅ Dashboard démarré (http://localhost:5174)")
        else:
            print(f"   ⚠️  Docker: {result.stderr.decode()[:100]}")
    except Exception as e:
        print(f"   ⚠️  Docker error: {e}")
    
    # 2. Vérifier dataset français
    print("\n2️⃣  Vérification datasets français...")
    dataset_path = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")
    if dataset_path.exists():
        size_gb = dataset_path.stat().st_size / 1e9
        print(f"   ✅ Dataset prêt: {dataset_path.name} ({size_gb:.1f}GB)")
        use_dataset = str(dataset_path)
    else:
        print(f"   ⚠️  Dataset non trouvé, cherche alternatives...")
        alternatives = list(Path("data_clean").glob("*tokenized.pt"))
        if alternatives:
            use_dataset = str(alternatives[0])
            print(f"   ✅ Utilisation: {alternatives[0].name}")
        else:
            print("   ❌ Aucun dataset tokenisé trouvé!")
            return 1
    
    # 3. Lancer training
    print("\n3️⃣  Lancement Training...")
    print(f"   Dataset: {use_dataset}")
    print(f"   Steps: 100,000 (reprise depuis checkpoint 100k)")
    print(f"   Durée estimée: 24-48h")
    print(f"   GPU: RTX 5080 16GB")
    
    cmd = [
        ".venv/bin/python3",
        "scripts/quick_train.py",
        "--data-dir", "data_clean",
        "--max-steps", "110000",
        "--batch-size", "10",
        "--seq-len", "256",
        "--lr", "1e-6",
        "--device", "cuda",
        "--save-every", "1000"
    ]
    
    print(f"\n   Commande: {' '.join(cmd)}")
    print(f"\n   📊 Dashboard: http://localhost:5174")
    print(f"   📄 Logs: tail -f training.log")
    print(f"   💾 Checkpoints: trained_models/runs/")
    
    # Lancer en arrière-plan
    with open("training.log", "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent
        )
    
    print(f"\n✅ Training lancé (PID: {process.pid})")
    print(f"\n" + "="*70)
    print("🎯 TOUT EST PRÊT - TU PEUX PARTIR AFK 12h!")
    print("="*70)
    print(f"\nAprès 24-48h de training:")
    print(f"  1. Checkpoint @ 110,000 steps")
    print(f"  2. Export HuggingFace format")
    print(f"  3. Conversion GGUF")
    print(f"  4. Publication")
    print(f"\n🇫🇷 MODÈLE V2 EN CONSTRUCTION! 🚀\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
