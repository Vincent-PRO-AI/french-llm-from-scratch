#!/usr/bin/env python3
"""
Script de lancement pour entraînement V2 avec tokenizer Vincent SentencePiece.
Utilise les datasets pré-tokenisés v2 et le checkpoint de base 100k.
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Exécute une commande et affiche le résultat."""
    print(f"\n🔄 {description}")
    print(f"   Commande: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print("   ✅ Succès")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()[:200]}...")
        else:
            print(f"   ❌ Échec (code {result.returncode})")
            if result.stderr:
                print(f"   Erreur: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False
    return True

def main():
    print("🚀 LANCEMENT ENTRAÎNEMENT V2 - TOKENIZER VINCENT")
    print("=" * 60)

    # Vérifier l'environnement virtuel
    venv_python = Path(".venv/bin/python3")
    if not venv_python.exists():
        venv_python = Path(".venv/Scripts/python.exe")  # Windows fallback
    if not venv_python.exists():
        print("❌ Environnement virtuel non trouvé dans .venv/")
        return

    print(f"🐍 Utilisation de Python: {venv_python}")

    # Vérifier les fichiers requis
    required_files = [
        "data_clean/tokenizers/tokenizer.model",
        "data_clean/v2_tokenized/french_large_tokenized.pt",
        "trained_models/runs/checkpoint_step_100000.pt"
    ]

    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Fichier requis manquant: {file_path}")
            return
        else:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")

    print("\n📊 Configuration V2:")
    print("   - Tokenizer: Vincent SentencePiece (32k vocab)")
    print("   - Dataset: french_large_tokenized.pt (4GB pré-tokenisé)")
    print("   - Checkpoint: checkpoint_step_100000.pt")
    print("   - Phase: 2A - Introduction progressive format conversation")
    print("   - Steps: 100000 → 105000 (5k steps)")
    print("   - LR: 1e-6 (conservateur)")
    print("   - Run: french_v2_instruction_progressive")

    # Commande d'entraînement
    cmd = [
        str(venv_python), "scripts/train_subtitles_transformer.py",
        "--arch-preset", "medium",
        "--pretokenized-path", "data_clean/v2_tokenized/french_large_tokenized.pt",
        "--tokenizer-path", "data_clean/tokenizers/tokenizer.model",
        "--run-name", "french_v2_instruction_progressive",
        "--resume-from", "trained_models/runs/checkpoint_step_100000.pt",
        "--max-steps", "105000",
        "--batch-size", "4",
        "--lr", "1e-6",
        "--checkpoint-interval", "500",
        "--sample-interval", "200",
        "--eval-interval", "100",
        "--device", "cuda",
        "--use-amp",
        "--weight-decay", "0.01"
    ]

    print(f"\n🔄 Commande: {' '.join(cmd)}")
    print("\n⏳ Lancement de l'entraînement V2... (ça va prendre ~1h pour 5k steps)")
    print("   Suivi:")
    print("   - Logs: tail -f trained_models/runs/french_v2_instruction_progressive/metrics.jsonl")
    print("   - Samples: tail -f samples.txt")
    print("   - Dashboard: docker-compose up -d && open http://localhost:5174")

    try:
        # Lancer en arrière-plan
        process = subprocess.Popen(cmd, cwd=Path(__file__).parent)
        print(f"\n✅ Entraînement lancé (PID: {process.pid})")
        print("   L'entraînement continue en arrière-plan.")
        print("   Tu peux fermer ce terminal.")

    except Exception as e:
        print(f"❌ Impossible de lancer l'entraînement: {e}")

if __name__ == "__main__":
    main()