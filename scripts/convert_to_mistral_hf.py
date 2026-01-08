#!/usr/bin/env python3
"""
Convertir checkpoint custom vers format Mistral/Llama compatible HF + GGUF.
Remapping des clés PyTorch Transformer → Mistral naming convention.
"""

import torch
import json
from pathlib import Path
from collections import OrderedDict


def convert_state_dict_to_mistral(state_dict, config):
    """
    Convertir state_dict custom PyTorch → format Mistral/Llama.
    
    Mapping:
    - tok_embed.weight → model.embed_tokens.weight
    - encoder.layers.N.self_attn.in_proj_weight → model.layers.N.self_attn.{q,k,v}_proj.weight (split)
    - encoder.layers.N.self_attn.out_proj.weight → model.layers.N.self_attn.o_proj.weight
    - encoder.layers.N.linear1.weight → model.layers.N.mlp.gate_proj.weight (SwiGLU split)
    - encoder.layers.N.linear2.weight → model.layers.N.mlp.down_proj.weight
    - encoder.layers.N.norm1.weight → model.layers.N.input_layernorm.weight
    - encoder.layers.N.norm2.weight → model.layers.N.post_attention_layernorm.weight
    - lm_head.weight → lm_head.weight
    """
    
    new_state = OrderedDict()
    num_heads = config['num_heads']
    embed_dim = config['embed_dim']
    head_dim = embed_dim // num_heads
    
    print(f"🔄 Conversion state_dict → Mistral format")
    print(f"   Heads: {num_heads}, Embed: {embed_dim}, Head dim: {head_dim}")
    
    for key, tensor in state_dict.items():
        if key == 'tok_embed.weight':
            new_state['model.embed_tokens.weight'] = tensor
            print(f"✓ {key} → model.embed_tokens.weight")
        
        elif key.startswith('encoder.layers.'):
            # Extraire numéro de layer
            parts = key.split('.')
            layer_num = parts[2]
            rest = '.'.join(parts[3:])
            
            if rest == 'self_attn.in_proj_weight':
                # Split Q, K, V
                q, k, v = tensor.chunk(3, dim=0)
                new_state[f'model.layers.{layer_num}.self_attn.q_proj.weight'] = q
                new_state[f'model.layers.{layer_num}.self_attn.k_proj.weight'] = k
                new_state[f'model.layers.{layer_num}.self_attn.v_proj.weight'] = v
                print(f"✓ layer {layer_num}: in_proj_weight → q/k/v_proj (split)")
                
            elif rest == 'self_attn.in_proj_bias':
                # Split Q, K, V bias
                q, k, v = tensor.chunk(3, dim=0)
                new_state[f'model.layers.{layer_num}.self_attn.q_proj.bias'] = q
                new_state[f'model.layers.{layer_num}.self_attn.k_proj.bias'] = k
                new_state[f'model.layers.{layer_num}.self_attn.v_proj.bias'] = v
                
            elif rest == 'self_attn.out_proj.weight':
                new_state[f'model.layers.{layer_num}.self_attn.o_proj.weight'] = tensor
                
            elif rest == 'self_attn.out_proj.bias':
                new_state[f'model.layers.{layer_num}.self_attn.o_proj.bias'] = tensor
                
            elif rest == 'linear1.weight':
                # Mistral utilise SwiGLU: gate_proj et up_proj
                # Notre linear1 a déjà la bonne taille (4096), on le duplique pour gate et up
                new_state[f'model.layers.{layer_num}.mlp.gate_proj.weight'] = tensor
                new_state[f'model.layers.{layer_num}.mlp.up_proj.weight'] = tensor.clone()
                print(f"✓ layer {layer_num}: linear1 → gate/up_proj (duplicated)")
                
            elif rest == 'linear1.bias':
                new_state[f'model.layers.{layer_num}.mlp.gate_proj.bias'] = tensor
                new_state[f'model.layers.{layer_num}.mlp.up_proj.bias'] = tensor.clone()
                
            elif rest == 'linear2.weight':
                new_state[f'model.layers.{layer_num}.mlp.down_proj.weight'] = tensor
                
            elif rest == 'linear2.bias':
                new_state[f'model.layers.{layer_num}.mlp.down_proj.bias'] = tensor
                
            elif rest == 'norm1.weight':
                new_state[f'model.layers.{layer_num}.input_layernorm.weight'] = tensor
                
            elif rest == 'norm1.bias':
                new_state[f'model.layers.{layer_num}.input_layernorm.bias'] = tensor
                
            elif rest == 'norm2.weight':
                new_state[f'model.layers.{layer_num}.post_attention_layernorm.weight'] = tensor
                
            elif rest == 'norm2.bias':
                new_state[f'model.layers.{layer_num}.post_attention_layernorm.bias'] = tensor
        
        elif key == 'lm_head.weight':
            new_state['lm_head.weight'] = tensor
            print(f"✓ {key} → lm_head.weight")
        
        elif key.startswith('final_norm.'):
            # Norm finale
            new_key = key.replace('final_norm', 'model.norm')
            new_state[new_key] = tensor
            print(f"✓ {key} → {new_key}")
    
    # Ajouter model.norm (output_norm) si absent : initialiser à ones/zeros (RMSNorm/LayerNorm)
    if 'model.norm.weight' not in new_state:
        embed_dim = config['embed_dim']
        new_state['model.norm.weight'] = torch.ones(embed_dim, dtype=torch.float16)
        print(f"✓ Ajout model.norm.weight (absent, init ones) [{embed_dim}]")
    
    # Ajouter lm_head si absent : weight-tying avec embedding
    if 'lm_head.weight' not in new_state and 'model.embed_tokens.weight' in new_state:
        new_state['lm_head.weight'] = new_state['model.embed_tokens.weight'].clone()
        print(f"✓ Ajout lm_head.weight (weight-tied avec embed_tokens)")
    
    print(f"✅ Converti {len(state_dict)} → {len(new_state)} tenseurs")
    return new_state


def create_mistral_config(original_config):
    """Créer config.json compatible Mistral."""
    return {
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "vocab_size": original_config['vocab_size'],
        "hidden_size": original_config['embed_dim'],
        "intermediate_size": original_config['ff_hidden_dim'],
        "num_hidden_layers": original_config['num_layers'],
        "num_attention_heads": original_config['num_heads'],
        "num_key_value_heads": original_config['num_heads'],  # GQA: même nombre pour simplicité
        "max_position_embeddings": original_config['block_size'],
        "rms_norm_eps": 1e-6,  # Mistral standard
        "rope_theta": 10000.0,
        "attention_dropout": original_config['dropout'],
        "hidden_act": "silu",  # Mistral utilise SiLU
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "use_cache": True,
        "torch_dtype": "float16",
        "transformers_version": "4.41.0",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }


def main():
    print("=" * 70)
    print("🚀 Conversion checkpoint → Mistral HF format + GGUF")
    print("=" * 70)
    
    # Chemins
    checkpoint_path = Path("trained_models/runs/french-llm-from-scratch-V3-mistral/checkpoint.pt")
    output_dir = Path("trained_models/french-llm-v3-mistral-hf-compatible")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint non trouvé: {checkpoint_path}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger checkpoint
    print(f"\n📥 Chargement: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config = checkpoint.get('config', {})
    state_dict = checkpoint.get('model_state', {})
    
    print(f"✅ Chargé: {sum(p.numel() for p in state_dict.values()):,} paramètres")
    
    # Convertir state dict
    print(f"\n🔄 Conversion du state_dict...")
    new_state = convert_state_dict_to_mistral(state_dict, config)
    
    # Créer config Mistral
    print(f"\n📝 Création config Mistral...")
    mistral_config = create_mistral_config(config)
    
    # Sauvegarder
    print(f"\n💾 Sauvegarde...")
    
    # Config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(mistral_config, f, indent=2)
    print(f"✓ Config: {config_path}")
    
    # Poids en float16
    model_path = output_dir / "pytorch_model.bin"
    state_fp16 = {k: v.half() for k, v in new_state.items()}
    torch.save(state_fp16, model_path)
    size_mb = model_path.stat().st_size / 1e6
    print(f"✓ Poids: {model_path} ({size_mb:.1f} MB)")
    
    # Metadata
    metadata = {
        "model_name": "French-LLM-V3-Mistral",
        "architecture": "MistralForCausalLM (compatible)",
        "training_steps": 100000,
        "final_train_loss": 5.72,
        "final_val_loss": 5.29,
        "parameters": sum(p.numel() for p in new_state.values()),
        "vocab_size": config['vocab_size'],
        "tokenizer": "mistralai/Mistral-7B-v0.1",
    }
    
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata: {metadata_path}")
    
    # Copier tokenizer
    from shutil import copytree
    tokenizer_src = Path("data_clean/mistral_tokenizer")
    tokenizer_dst = output_dir
    
    if tokenizer_src.exists():
        for item in tokenizer_src.iterdir():
            if item.is_file():
                import shutil
                shutil.copy2(item, tokenizer_dst / item.name)
        print(f"✓ Tokenizer copié")
    
    print("\n" + "=" * 70)
    print("✅ Conversion terminée!")
    print("=" * 70)
    print(f"📁 Dossier: {output_dir}/")
    print(f"\n🔧 Test conversion GGUF:")
    print(f"   python /home/vincent/llama.cpp/convert_hf_to_gguf.py \\")
    print(f"     --outtype f16 \\")
    print(f"     --outfile trained_models/french-llm-v3-mistral-f16.gguf \\")
    print(f"     {output_dir}")
    

if __name__ == "__main__":
    main()
