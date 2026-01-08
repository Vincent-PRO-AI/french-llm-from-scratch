#!/usr/bin/env python3
"""
Supprimer TOUS les biaises du modèle pour correspondre à la config sans bias.
"""

import torch
from pathlib import Path


def remove_all_biases(state_dict):
    """Supprimer tous les tenseurs de bias du state dict."""
    print(f"État initial: {len(state_dict)} tenseurs")
    
    # Copier et filtrer
    new_state = {}
    bias_removed = []
    
    for key, tensor in state_dict.items():
        if 'bias' in key:
            bias_removed.append(key)
            print(f"  ❌ Suppression: {key}")
        else:
            new_state[key] = tensor
    
    print(f"\nÉtat final: {len(new_state)} tenseurs (biases supprimés: {len(bias_removed)})")
    return new_state


def main():
    model_path = Path("trained_models/french-llm-v3-mistral-hf-compatible/pytorch_model.bin")
    
    print("=" * 80)
    print("🔧 Suppression de tous les biais du modèle")
    print("=" * 80)
    print(f"\n📁 Modèle: {model_path}\n")
    
    if not model_path.exists():
        print(f"❌ Fichier non trouvé")
        return
    
    # Charger
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Nettoyer
    clean_state = remove_all_biases(state)
    
    # Sauvegarder
    print(f"\n💾 Sauvegarde...")
    torch.save(clean_state, model_path)
    size_mb = model_path.stat().st_size / 1e6
    print(f"✓ {model_path} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 80)
    print("✅ Nettoyage terminé!")
    print("=" * 80)


if __name__ == "__main__":
    main()
