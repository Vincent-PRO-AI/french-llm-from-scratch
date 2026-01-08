#!/usr/bin/env python3
"""
Lance un entraînement de fine-tuning depuis le checkpoint 160k 
avec le nouveau dataset français pur de 530M tokens.
"""
import subprocess
import sys
from pathlib import Path

def launch_training():
    print("🚀 LANCEMENT ENTRAÎNEMENT FRANÇAIS PUR - CHECKPOINT 160k→220k (DONNÉES PROPRES)")
    print("=" * 80)

    # Chemins
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_subtitles_transformer.py"
    checkpoint_160k = Path("trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_160000.pt")
    tokenizer = Path("data_clean/mistral_tokenizer")
    train_data = Path("data_clean/french_large_filtered_mistral_tokenized.pt")

    # Vérifier les fichiers
    missing = []
    if not train_script.exists():
        missing.append(f"Script: {train_script}")
    if not checkpoint_160k.exists():
        missing.append(f"Checkpoint 160k: {checkpoint_160k}")
    if not tokenizer.exists():
        missing.append(f"Tokenizer: {tokenizer}")
    if not train_data.exists():
        missing.append(f"Dataset: {train_data}")

    if missing:
        print("❌ Fichiers manquants:")
        for item in missing:
            print(f"   - {item}")
        return

    print("✅ Tous les fichiers requis sont présents")
    print(f"   Checkpoint: {checkpoint_160k}")
    print(f"   (Avant entraînement corrompu d'hier)")
    print(f"   Dataset: {train_data}")
    print(f"   - Tokens: 530.1M (FRANÇAIS PUR)")
    print(f"   - Taille: {train_data.stat().st_size / (1024**3):.1f} GB")
    print(f"   Tokenizer: Mistral 32k\n")

    # Commande
    cmd = [
        sys.executable, str(train_script),
        "--resume-from", str(checkpoint_160k),
        "--pretokenized-path", str(train_data),
        "--tokenizer-path", str(tokenizer),
        "--run-name", "french_large_filtered_160k_220k",
        "--max-steps", "220000",  # 160k + 60k steps
        "--arch-preset", "medium",
        "--batch-size", "4",
        "--gradient-accumulation-steps", "4",
        "--lr", "3e-5",
        "--eval-interval", "500",
        "--sample-interval", "500",
        "--checkpoint-interval", "2500",
        "--metrics-log-fraction", "0.05",
        "--sample-prompt", "Utilisateur: Explique-moi l'histoire de France. Assistant:",
        "--sample-max-new-tokens", "150",
        "--sample-temperature", "0.8",
        "--sample-top-k", "50",
    ]

    print("⚙️  Configuration d'entraînement:")
    print(f"   De 160k à 220k steps (+60k)")
    print(f"   Batch: 4 × 4 accumulation = effective 16")
    print(f"   LR: 3e-5 avec cosine annealing")
    print(f"   Dataset: 530M tokens FRANÇAIS PUR (filtré de french_large)")
    print(f"   Estimation: ~10-12h sur RTX 5080")
    print()

    print("🚀 Lancement...")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Entraînement terminé avec succès!")
        print(f"   Checkpoint sauvegardé dans trained_models/runs/french_large_filtered_160k_220k/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        return
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu par l'utilisateur")
        return


if __name__ == "__main__":
    launch_training()
