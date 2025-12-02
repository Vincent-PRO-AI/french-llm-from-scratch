#!/usr/bin/env python3
"""
Script de lancement automatique pour entraînement avec tokenizer Mistral.
Prépare les données, tokenise, puis lance l'entraînement 200k→300k.
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
    print("🚀 LANCEMENT AUTOMATIQUE ENTRAÎNEMENT MISTRAL 200k→300k")
    print("=" * 60)

    # Activer l'environnement virtuel
    venv_python = Path(".venv/bin/python3")
    if not venv_python.exists():
        venv_python = Path(".venv/Scripts/python.exe")  # Windows fallback
    if not venv_python.exists():
        print("❌ Environnement virtuel non trouvé dans .venv/")
        return

    print(f"🐍 Utilisation de Python: {venv_python}")

    # Étape 1: Combiner les corpus texte
    if not run_command([str(venv_python), "scripts/combine_text_corpus.py"],
                      "Étape 1: Combinaison des corpus texte"):
        return

    # Étape 2: Tokenisation avec Mistral
    if not run_command([str(venv_python), "scripts/tokenize_with_mistral.py"],
                      "Étape 2: Tokenisation avec Mistral-7B"):
        return

    # Étape 3: Lancement de l'entraînement
    print("\n🔄 Étape 3: Lancement entraînement Mistral 200k→300k")
    print("   Configuration:")
    print("   - Tokenizer: Mistral-7B (117k vocab)")
    print("   - Données: conversations_mega_mistral_train_tokenized.pt")
    print("   - Test: conversations_mega_mistral_test_tokenized.pt")
    print("   - Run: french_medium_mistral_300k")
    print("   - Steps: 200000 → 300000 (100k steps)")
    print("   - Batch size: 4")
    print("   - Optimisations: torch.compile + FusedAdam + AMP")
    print("   - Checkpoint: checkpoint_step_200000.pt")

    cmd = [
        str(venv_python), "scripts/train_subtitles_transformer.py",
        "--arch-preset", "medium",
        "--pretokenized-path", "data_clean/conversations_mega_mistral_train_tokenized.pt",
        "--tokenizer-path", "data_clean/mistral_tokenizer/tokenizer.json",
        "--run-name", "french_medium_mistral_300k",
        "--resume-from", "trained_models/runs/french_medium_mega_finetune_200k/checkpoint_step_200000.pt",
        "--max-steps", "300000",
        "--batch-size", "4",
        "--checkpoint-interval", "2500",
        "--sample-interval", "500",
        "--eval-interval", "1000",
        "--metrics-log-fraction", "0.05"
    ]

    print(f"   Commande: {' '.join(cmd)}")
    print("\n⏳ Lancement de l'entraînement... (ça va prendre ~7h)")
    print("   Tu peux suivre la progression avec:")
    print("   tail -f trained_models/runs/french_medium_mistral_300k/metrics.jsonl")
    print("   Ou utiliser le dashboard: docker-compose up -d && open http://localhost:5174")

    try:
        # Lancer en arrière-plan pour ne pas bloquer
        process = subprocess.Popen(cmd, cwd=Path(__file__).parent)
        print(f"   PID du processus: {process.pid}")
        print("   ✅ Entraînement lancé en arrière-plan!")
        print("   Tu peux fermer ce terminal, l'entraînement continue.")
        print("\n📊 Suivi:")
        print("   - Logs: trained_models/runs/french_medium_mistral_300k/")
        print("   - Métriques: tail -f metrics.jsonl")
        print("   - Samples: tail -f samples.txt")
        print("   - Checkpoints: ls -la checkpoint_step_*.pt")

    except Exception as e:
        print(f"   ❌ Impossible de lancer l'entraînement: {e}")

if __name__ == "__main__":
    main()