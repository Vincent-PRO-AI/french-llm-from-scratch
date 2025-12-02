#!/usr/bin/env python3
"""
Launch training with Mistral tokenizer from 200k to 300k steps.
Uses the new tokenized datasets and Mistral tokenizer.
"""
import subprocess
import sys
from pathlib import Path

def launch_mistral_training():
    print("🚀 LANCEMENT ENTRAÎNEMENT MISTRAL 200k→300k")
    print("=" * 60)

    # Paths
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_subtitles_transformer.py"
    checkpoint_200k = Path("trained_models/runs/french_medium_mega_finetune_200k/checkpoint_step_200000.pt")
    mistral_tokenizer = Path("data_clean/mistral_tokenizer")
    train_data = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")
    test_data = Path("data_clean/conversations_mega_test_mistral_tokenized.pt")

    # Verify paths exist
    missing = []
    if not train_script.exists():
        missing.append(f"Script: {train_script}")
    if not checkpoint_200k.exists():
        missing.append(f"Checkpoint 200k: {checkpoint_200k}")
    if not mistral_tokenizer.exists():
        missing.append(f"Tokenizer Mistral: {mistral_tokenizer}")
    if not train_data.exists():
        missing.append(f"Données train: {train_data}")
    if not test_data.exists():
        missing.append(f"Données test: {test_data}")

    if missing:
        print("❌ Fichiers manquants:")
        for item in missing:
            print(f"   - {item}")
        return

    print("✅ Tous les fichiers requis sont présents")
    print(f"   Script: {train_script}")
    print(f"   Checkpoint: {checkpoint_200k}")
    print(f"   Tokenizer: {mistral_tokenizer}")
    print(f"   Train data: {train_data} ({train_data.stat().st_size / (1024**3):.1f} GB)")
    print(f"   Test data: {test_data} ({test_data.stat().st_size / (1024**3):.1f} GB)")

    # Command to run
    cmd = [
        sys.executable, str(train_script),
        "--resume-from", str(checkpoint_200k),
        "--pretokenized-path", str(train_data),
        "--tokenizer-path", str(mistral_tokenizer),
        "--run-name", "french_medium_mistral_300k",
        "--max-steps", "300000",  # 200k + 100k = 300k
        "--arch-preset", "medium",
        "--batch-size", "4",
        "--lr", "2e-4",
        "--eval-interval", "100",
        "--sample-interval", "250",
        "--checkpoint-interval", "2500",
        "--metrics-log-fraction", "0.05",
        "--sample-prompt", "Utilisateur: Bonjour, comment vas-tu ? Assistant:",
        "--sample-max-new-tokens", "120",
        "--sample-temperature", "0.9",
        "--sample-top-k", "50",
    ]

    print(f"\n⚙️  Commande d'exécution:")
    print(f"   {' '.join(cmd)}")

    print(f"\n🚀 Lancement de l'entraînement...")
    print(f"   De 200k à 300k steps (~7h avec RTX 5080)")
    print(f"   Tokenizer: Mistral-7B (32k vocab)")
    print(f"   Données: {train_data.stat().st_size / (1024**3):.1f} GB tokenisées")
    print(f"   Optimisations: torch.compile + FusedAdam + AMP")
    print()

    # Run the training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Entraînement terminé avec succès!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        return
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu par l'utilisateur")
        return

if __name__ == "__main__":
    launch_mistral_training()