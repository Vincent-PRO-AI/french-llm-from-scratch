#!/usr/bin/env python3
"""Fine-tuning du modèle de base avec dataset français pur."""
import argparse
import sys
from pathlib import Path

# Ajouter le root au path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_subtitles_transformer import main as train_main

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune avec dataset français")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint de base")
    parser.add_argument("--data", required=True, help="Fichier texte français")
    parser.add_argument("--tokenizer-dir", required=True, help="Dossier tokenizer")
    parser.add_argument("--run-name", required=True, help="Nom du run")
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Construire les arguments pour train_subtitles_transformer
    train_args = [
        "--data-file", args.data,
        "--tokenizer-path", str(Path(args.tokenizer_dir) / "tokenizer.json"),
        "--resume-from", args.checkpoint,
        "--run-name", args.run_name,
        "--max-steps", str(args.max_steps),
        "--save-every", str(args.save_every),
        "--lr", str(args.learning_rate),
        "--batch-size", str(args.batch_size),
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--device", args.device,
        "--num-workers", "4",
        "--log-every", "50",
    ]
    
    if args.compile:
        train_args.append("--compile")
    
    # Remplacer sys.argv et lancer
    sys.argv = [sys.argv[0]] + train_args
    train_main()
