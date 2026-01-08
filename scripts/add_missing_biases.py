#!/usr/bin/env python3
"""
Corriger la conversion en ajoutant TOUS les biais manquants.
Le checkpoint original n'a peut-être pas de biais, mais le format GGUF les attend.
"""

import torch
import json
from pathlib import Path
from collections import OrderedDict


def add_missing_biases(state_dict, config):
    """
    Ajouter les biais manquants pour compatibilité GGUF.
    Stratégie: si un weight existe mais pas son bias, initialiser bias à zéros.
    """
    
    new_state = OrderedDict()
    embed_dim = config['embed_dim']
    ff_hidden = config['ff_hidden_dim']
    vocab_size = config['vocab_size']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    
    print(f"🔧 Ajout des biais manquants")
    print(f"   Config: {num_layers} layers, {embed_dim} embed, {ff_hidden} FF")
    
    # Copier tous les tenseurs existants
    for key, tensor in state_dict.items():
        new_state[key] = tensor
    
    bias_added = 0
    
    # Par couche: ajouter biais manquants
    for layer_num in range(num_layers):
        layer_keys = [k for k in new_state.keys() if f'model.layers.{layer_num}.' in k]
        
        # Attention biases
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            weight_key = f'model.layers.{layer_num}.self_attn.{proj}.weight'
            bias_key = f'model.layers.{layer_num}.self_attn.{proj}.bias'
            
            if weight_key in new_state and bias_key not in new_state:
                # Le bias doit avoir la première dimension du weight
                weight_shape = new_state[weight_key].shape
                bias_shape = (weight_shape[0],)
                new_state[bias_key] = torch.zeros(bias_shape, dtype=new_state[weight_key].dtype)
                bias_added += 1
                print(f"   + layer {layer_num}.self_attn.{proj}.bias {bias_shape}")
        
        # LayerNorm biases
        for norm in ['input_layernorm', 'post_attention_layernorm']:
            weight_key = f'model.layers.{layer_num}.{norm}.weight'
            bias_key = f'model.layers.{layer_num}.{norm}.bias'
            
            if weight_key in new_state and bias_key not in new_state:
                bias_shape = (embed_dim,)
                new_state[bias_key] = torch.zeros(bias_shape, dtype=new_state[weight_key].dtype)
                bias_added += 1
                print(f"   + layer {layer_num}.{norm}.bias {bias_shape}")
        
        # FFN biases
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            weight_key = f'model.layers.{layer_num}.mlp.{proj}.weight'
            bias_key = f'model.layers.{layer_num}.mlp.{proj}.bias'
            
            if weight_key in new_state and bias_key not in new_state:
                weight_shape = new_state[weight_key].shape
                bias_shape = (weight_shape[0],)
                new_state[bias_key] = torch.zeros(bias_shape, dtype=new_state[weight_key].dtype)
                bias_added += 1
                print(f"   + layer {layer_num}.mlp.{proj}.bias {bias_shape}")
    
    # Final norm biases
    if 'model.norm.weight' in new_state and 'model.norm.bias' not in new_state:
        new_state['model.norm.bias'] = torch.zeros(embed_dim, dtype=new_state['model.norm.weight'].dtype)
        bias_added += 1
        print(f"   + model.norm.bias ({embed_dim},)")
    
    print(f"\n✅ Total biases ajoutés: {bias_added}")
    print(f"   Tenseurs avant: {len(state_dict)}")
    print(f"   Tenseurs après: {len(new_state)}")
    
    return new_state


def main():
    print("=" * 80)
    print("🔧 Correction: Ajout des biais manquants pour GGUF")
    print("=" * 80)
    
    # Charger le modèle converti actuel
    model_path = Path("trained_models/french-llm-v3-mistral-hf-compatible/pytorch_model.bin")
    config_path = Path("trained_models/french-llm-v3-mistral-hf-compatible/config.json")
    
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    if not config_path.exists():
        print(f"❌ Config non trouvée: {config_path}")
        return
    
    print(f"\n📥 Chargement du modèle...")
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    print(f"   Tenseurs actuels: {len(state)}")
    
    with open(config_path) as f:
        config_hf = json.load(f)
    
    # Convertir config HF vers config interne
    config = {
        'embed_dim': config_hf['hidden_size'],
        'ff_hidden_dim': config_hf['intermediate_size'],
        'num_heads': config_hf['num_attention_heads'],
        'num_layers': config_hf['num_hidden_layers'],
        'vocab_size': config_hf['vocab_size'],
    }
    
    # Ajouter les biais manquants
    print(f"\n🔧 Traitement...")
    new_state = add_missing_biases(state, config)
    
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
    print(f"     --outfile trained_models/french-llm-v3-mistral-f16-v4.gguf \\")
    print(f"     trained_models/french-llm-v3-mistral-hf-compatible")
    print("=" * 80)


if __name__ == "__main__":
    main()
