#!/usr/bin/env python3
"""
Lancement adaptatif entraînement Mistral 200k→300k avec auto-tune RAM.
Utilise les nouveaux paramètres DataLoader et le tokenizer Mistral.
"""
import sys
from pathlib import Path
import subprocess

TRAIN_SCRIPT = Path(__file__).parent / "train_subtitles_transformer.py"
CHECKPOINT = Path("trained_models/runs/french_medium_mega_finetune_200k/checkpoint_step_200000.pt")
TOKENIZER = Path("data_clean/mistral_tokenizer")
PRETOKENIZED = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")


def ram_gb():
    try:
        with open('/proc/meminfo','r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    return kb / (1024*1024)
    except Exception:
        return 0.0
    return 0.0


def build_command():
    r = ram_gb()
    # Base params
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--resume-from", str(CHECKPOINT),
        "--pretokenized-path", str(PRETOKENIZED),
        "--tokenizer-path", str(TOKENIZER),
        "--run-name", "french_medium_mistral_300k_adaptive",
        "--max-steps", "300000",
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
    # Ajustements selon RAM
    if r >= 64:
        cmd += ["--num-workers", "8", "--prefetch-factor", "6"]
    elif r >= 48:
        cmd += ["--num-workers", "6", "--prefetch-factor", "5"]
    elif r >= 32:
        cmd += ["--num-workers", "6", "--prefetch-factor", "4"]
    else:
        cmd += ["--num-workers", "4", "--prefetch-factor", "2"]
    return cmd, r


def main():
    missing = [p for p in [TRAIN_SCRIPT, CHECKPOINT, TOKENIZER, PRETOKENIZED] if not p.exists()]
    if missing:
        print("❌ Fichiers manquants:")
        for m in missing:
            print(" -", m)
        return
    cmd, r = build_command()
    print("🚀 LANCEMENT ADAPTATIF MISTRAL 200k→300k")
    print("RAM détectée: {:.1f} GB".format(r))
    print("Commande:")
    print(" ", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("⏹️  Interrompu")
    except subprocess.CalledProcessError as e:
        print("❌ Erreur:", e)

if __name__ == "__main__":
    main()
