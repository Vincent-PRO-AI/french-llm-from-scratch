#!/usr/bin/env python3
"""
Supprimer les biais de normalisation pour compatibilité Mistral/Llama.
llama.cpp ignore les biais de norme pour l'architecture Llama/Mistral,
ce qui cause une erreur "wrong number of tensors".
"""

import torch
import json
from pathlib import Path
from collections import OrderedDict

def remove_norm_biases(state_dict):
    """
    Supprimer les clés *.norm*.bias.
    """
    new_state = OrderedDict()
    removed_count = 0
    
    for key, tensor in state_dict.items():
        # Identifier les biais de norme
        is_norm_bias = False
        if 'bias' in key:
            if 'input_layernorm' in key or 'post_attention_layernorm' in key or 'model.norm' in key:
                is_norm_bias = True
        
        if is_norm_bias:
            print(f"   - Suppression: {key}")
            removed_count += 1
        else:
            new_state[key] = tensor
            
    return new_state, removed_count

def main():
    print("=" * 80)
    print("🔧 Correction: Suppression des biais de norme pour GGUF")
    print("=" * 80)
    
    model_path = Path("trained_models/french-llm-v3-mistral-hf-compatible/pytorch_model.bin")
    
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    print(f"\n📥 Chargement du modèle...")
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    print(f"   Tenseurs actuels: {len(state)}")
    
    print(f"\n🔧 Nettoyage...")
    new_state, count = remove_norm_biases(state)
    
    print(f"\n✅ Total biais supprimés: {count}")
    print(f"   Tenseurs restants: {len(new_state)}")
    
    # Sauvegarder
    print(f"\n💾 Sauvegarde...")
    torch.save(new_state, model_path)
    size_mb = model_path.stat().st_size / 1e6
    print(f"✓ {model_path} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 80)
    print("✅ Correction terminée!")
    print(f"Reconvertir en GGUF:")
    print(f"   python /home/vincent/llama.cpp/convert_hf_to_gguf.py \\")
    print(f"     --outtype f16 \\")
    print(f"     --outfile trained_models/french-llm-v3-mistral-f16-v5.gguf \\")
    print(f"     trained_models/french-llm-v3-mistral-hf-compatible")
    print("=" * 80)

if __name__ == "__main__":
    main()
