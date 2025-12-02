#!/usr/bin/env python3
"""
Entraînement FROM SCRATCH avec tokenizer Mistral 0→200k.
Recommandé pour convergence propre sans mismatch de tokenizer.
"""
import sys
from pathlib import Path
import subprocess

TRAIN_SCRIPT = Path(__file__).parent / "train_subtitles_transformer.py"
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
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--pretokenized-path", str(PRETOKENIZED),
        "--tokenizer-path", str(TOKENIZER),
        "--run-name", "french_medium_mistral_from_scratch",
        "--max-steps", "200000",
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
    missing = [p for p in [TRAIN_SCRIPT, TOKENIZER, PRETOKENIZED] if not p.exists()]
    if missing:
        print("❌ Fichiers manquants:")
        for m in missing:
            print(" -", m)
        return
    cmd, r = build_command()
    print("🚀 ENTRAÎNEMENT FROM SCRATCH MISTRAL 0→200k")
    print("RAM détectée: {:.1f} GB".format(r))
    print("NOUVEAU: Tokenizer Mistral dès le départ (pas de mismatch)")
    print("Durée estimée: ~11h (200k steps)")
    print("Commande:")
    print(" ", " ".join(cmd))
    print("\n⚠️  Attention: Ceci démarre un nouvel entraînement complet (0→200k)")
    print("Les checkpoints 200k/300k précédents ne seront PAS utilisés.")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("⏹️  Interrompu")
    except subprocess.CalledProcessError as e:
        print("❌ Erreur:", e)

if __name__ == "__main__":
    main()
