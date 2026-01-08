#!/usr/bin/env python3
"""
Script pour charger et combiner les partitions V2.
"""
import torch
import json
from pathlib import Path

def load_v2_partitions():
    partitioned_dir = Path("data_clean/v2_tokenized/phase2_partitioned")
    metadata_path = partitioned_dir / "metadata.json"

    if not metadata_path.exists():
        print("❌ metadata.json non trouvé")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    partitions = metadata['partitions']
    all_tokens = []

    print(f"🔄 Chargement de {len(partitions)} partitions...")

    for partition in partitions:
        path = Path(partition['path'])
        if path.exists():
            print(f"   Chargement {path.name}...")
            tokens = torch.load(path, map_location='cpu')
            all_tokens.append(tokens)
        else:
            print(f"⚠️  Partition manquante: {path}")

    if all_tokens:
        combined = torch.cat(all_tokens)
        output_path = Path("data_clean/v2_tokenized/combined_phase2.pt")
        torch.save(combined, output_path)
        print(f"✅ Données combinées sauvegardées: {output_path}")
        print(f"   Taille: {len(combined):,} tokens")
        return output_path
    else:
        print("❌ Aucune partition chargée")
        return None

if __name__ == "__main__":
    load_v2_partitions()
