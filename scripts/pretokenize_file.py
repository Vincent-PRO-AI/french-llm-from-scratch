#!/usr/bin/env python3
"""
Pré-tokenise un unique fichier texte et sauvegarde les token IDs dans un fichier .pt
Usage:
  python scripts/pretokenize_file.py \
    --input-file data_clean/conversations_massive/ultrachat_fr.txt \
    --tokenizer trained_models/tokenizers/fineweb-32k/tokenizer.json \
    --output data_clean/ultrachat_fr_tokenized.pt
"""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer


def pretokenize_file(input_file: Path, tokenizer_path: Path, output_path: Path) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_file}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    text = input_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise RuntimeError(f"Fichier vide: {input_file}")

    encoding = tokenizer.encode(text)
    ids = encoding.ids
    if not ids:
        raise RuntimeError("Aucun token généré !")

    tensor = torch.tensor(ids, dtype=torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ {input_file.name} → {output_path.name} | tokens={tensor.numel():,} | {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-tokenise un unique fichier .txt")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pretokenize_file(args.input_file, args.tokenizer, args.output)


if __name__ == "__main__":
    main()
