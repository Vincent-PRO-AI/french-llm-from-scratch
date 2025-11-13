#!/usr/bin/env python3
"""Pré-tokenise un corpus de texte et sauvegarde les token IDs dans un fichier .pt"""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tqdm import tqdm


def pretokenize_corpus(
    input_dir: Path,
    tokenizer_path: Path,
    output_path: Path,
    max_files: int | None = None,
) -> None:
    """Tokenize tous les .txt dans input_dir et sauvegarde les IDs dans output_path."""
    
    print(f"📂 Chargement tokenizer depuis {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    txt_files = sorted(input_dir.glob("*.txt"))
    if max_files is not None:
        txt_files = txt_files[:max_files]
    
    print(f"📚 Trouvé {len(txt_files)} fichiers à tokenizer")
    
    all_ids: list[int] = []
    sentinel_id = tokenizer.token_to_id("[SEP]") or tokenizer.token_to_id("<sep>") or 0
    
    for txt_path in tqdm(txt_files, desc="Tokenisation"):
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            
            encoding = tokenizer.encode(text)
            ids = encoding.ids
            
            if ids:
                all_ids.extend(ids)
                # Ajouter un séparateur entre documents
                all_ids.append(sentinel_id)
        except Exception as exc:
            print(f"⚠️  Erreur lecture {txt_path.name}: {exc}")
            continue
    
    if not all_ids:
        raise RuntimeError("Aucun token généré!")
    
    print(f"✅ Total tokens: {len(all_ids):,}")
    print(f"💾 Sauvegarde dans {output_path}")
    
    # Sauvegarder en tant que tensor PyTorch
    tensor = torch.tensor(all_ids, dtype=torch.long)
    torch.save(tensor, output_path)
    
    # Afficher stats
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"📦 Fichier créé: {file_size_mb:.1f} MB")
    print(f"🎯 Shape: {tensor.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-tokenise un corpus")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Répertoire contenant les fichiers .txt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Chemin vers tokenizer.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fichier de sortie .pt",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limite le nombre de fichiers (pour tests)",
    )
    
    args = parser.parse_args()
    
    pretokenize_corpus(
        input_dir=args.input_dir,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
