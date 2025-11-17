#!/usr/bin/env python3
"""
Combine plusieurs datasets tokenisés en un seul fichier.
Utile pour fusionner conversations_all + conversations_extra.
"""
import argparse
import torch
from pathlib import Path


def combine_tokenized_datasets(input_files: list[Path], output_file: Path) -> None:
    """Combine plusieurs .pt en un seul."""
    
    print("🔗 Combinaison des datasets tokenisés...\n")
    
    all_tokens = []
    total_tokens = 0
    
    for input_path in input_files:
        if not input_path.exists():
            print(f"⚠️  Fichier non trouvé: {input_path}")
            continue
        
        print(f"📂 Chargement: {input_path}")
        tokens = torch.load(input_path, map_location='cpu')
        
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        print(f"   ✅ {len(tokens):,} tokens ({input_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        all_tokens.extend(tokens)
        total_tokens += len(tokens)
    
    print(f"\n📊 Total combiné: {total_tokens:,} tokens")
    
    # Convertir en tensor et sauvegarder
    print(f"💾 Sauvegarde: {output_file}")
    combined_tensor = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(combined_tensor, output_file)
    
    output_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Fichier créé: {output_size_mb:.1f} MB")
    print(f"🎯 Shape: {combined_tensor.shape}")
    
    # Stats
    print(f"\n📈 Statistiques:")
    print(f"   - Tokens: {total_tokens:,}")
    print(f"   - Taille: {output_size_mb:.1f} MB")
    print(f"   - Fichiers combinés: {len(input_files)}")


def main():
    parser = argparse.ArgumentParser(description="Combine plusieurs datasets tokenisés")
    parser.add_argument(
        "--inputs",
        nargs='+',
        type=Path,
        required=True,
        help="Fichiers .pt à combiner",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fichier de sortie .pt",
    )
    
    args = parser.parse_args()
    
    combine_tokenized_datasets(args.inputs, args.output)


if __name__ == "__main__":
    main()
