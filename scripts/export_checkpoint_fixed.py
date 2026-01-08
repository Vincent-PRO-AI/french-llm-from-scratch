#!/usr/bin/env python3
"""Export checkpoint vers format HuggingFace avec mapping correct des clés."""
import torch
import os
import shutil
import math
import json
from safetensors.torch import save_file

def _create_pos_encoding(embed_dim: int, max_seq_len: int) -> torch.Tensor:
    position = torch.arange(max_seq_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / embed_dim)
    )
    encoding = torch.zeros(max_seq_len, embed_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding

def convert_state_dict(checkpoint_state_dict):
    """Convertit les clés du checkpoint custom vers format Transformers."""
    new_state = {}
    
    for key, value in checkpoint_state_dict.items():
        # Mapping des clés custom -> HuggingFace Transformers
        if key == "tok_embed.weight":
            # Embedding des tokens
            new_key = "transformer.wte.weight"
            new_state[new_key] = value
            
        elif key == "ln.weight":
            # LayerNorm final
            new_state["transformer.ln_f.weight"] = value
            
        elif key == "ln.bias":
            new_state["transformer.ln_f.bias"] = value
            
        elif key == "head.weight":
            # LM head - Linear standard, pas de transposition
            new_state["lm_head.weight"] = value
            
        elif key.startswith("encoder.layers."):
            # Layers du transformer
            parts = key.replace("encoder.layers.", "").split(".")
            layer_num = parts[0]
            rest = ".".join(parts[1:])
            
            # Mapping des sous-composants
            if rest.startswith("self_attn.in_proj_"):
                # Attention: in_proj -> c_attn (qkv combinés)
                suffix = rest.replace("self_attn.in_proj_", "")
                new_key = f"transformer.h.{layer_num}.attn.c_attn.{suffix}"
                # Conv1D attend [out_features, in_features] donc transpose les weights
                if suffix == "weight":
                    value = value.t().contiguous()
                new_state[new_key] = value
                
            elif rest.startswith("self_attn.out_proj."):
                # Attention: out_proj -> c_proj
                suffix = rest.replace("self_attn.out_proj.", "")
                new_key = f"transformer.h.{layer_num}.attn.c_proj.{suffix}"
                # Conv1D transpose
                if suffix == "weight":
                    value = value.t().contiguous()
                new_state[new_key] = value
                
            elif rest.startswith("norm1."):
                # LayerNorm 1
                suffix = rest.replace("norm1.", "")
                new_key = f"transformer.h.{layer_num}.ln_1.{suffix}"
                new_state[new_key] = value
                
            elif rest.startswith("norm2."):
                # LayerNorm 2
                suffix = rest.replace("norm2.", "")
                new_key = f"transformer.h.{layer_num}.ln_2.{suffix}"
                new_state[new_key] = value
                
            elif rest.startswith("linear1."):
                # FFN: linear1 -> c_fc (expand)
                suffix = rest.replace("linear1.", "")
                new_key = f"transformer.h.{layer_num}.mlp.c_fc.{suffix}"
                # Conv1D transpose
                if suffix == "weight":
                    value = value.t().contiguous()
                new_state[new_key] = value
                
            elif rest.startswith("linear2."):
                # FFN: linear2 -> c_proj (contract)
                suffix = rest.replace("linear2.", "")
                new_key = f"transformer.h.{layer_num}.mlp.c_proj.{suffix}"
                # Conv1D transpose
                if suffix == "weight":
                    value = value.t().contiguous()
                new_state[new_key] = value
                
            else:
                print(f"⚠️ Clé non mappée: {key}")
        else:
            print(f"⚠️ Clé inconnue: {key}")
    
    # Ajouter position embeddings (Sinusoidal fixed)
    if "transformer.wpe.weight" not in new_state:
        # Doit correspondre à max_position_embeddings dans config (1024)
        print(f"⚠️ Génération des embeddings positionnels sinusoïdaux fixes...")
        
        # Récupérer dimensions depuis wte
        wte_shape = new_state["transformer.wte.weight"].shape
        vocab_size, embed_dim = wte_shape
        max_seq_len = 1024 # Valeur par défaut
        
        # Générer matrice sinusoïdale
        pos_enc = _create_pos_encoding(embed_dim, max_seq_len)
        new_state["transformer.wpe.weight"] = pos_enc
        print(f"✅ Generated transformer.wpe.weight: {max_seq_len}x{embed_dim}")
    
    return new_state


def main():
    checkpoint_path = "trained_models/runs/french_v3_finetune_pure_fr_160k_long/checkpoint_step_200000.pt"
    output_dir = "trained_models/hf_converted/french_200k_gpt2"
    tokenizer_dir = "data_clean/mistral_tokenizer"
    
    print("📦 Chargement du checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print("🔄 Conversion du state_dict...")
    new_state_dict = convert_state_dict(checkpoint["model_state"])
    
    print(f"✅ {len(new_state_dict)} clés converties")
    print("\nExemples de clés:")
    for i, key in enumerate(sorted(new_state_dict.keys())[:10]):
        print(f"  - {key}: {new_state_dict[key].shape}")
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder en safetensors
    print(f"\n💾 Sauvegarde en safetensors...")
    # Clone les tensors pour éviter le partage de mémoire
    new_state_dict_cloned = {k: v.clone() for k, v in new_state_dict.items()}
    save_file(new_state_dict_cloned, os.path.join(output_dir, "model.safetensors"))
    
    # Copier les fichiers tokenizer
    print("\n📋 Copie des fichiers tokenizer...")
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    
    for filename in files_to_copy:
        src = os.path.join(tokenizer_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✅ {filename}")
        else:
            print(f"  ⚠️ {filename} non trouvé")
            
    # Créer config.json pour GPT-2
    print("\n⚙️ Création de config.json standard GPT-2...")
    gpt2_config = {
        "activation_function": "gelu_new",
        "architectures": [
            "GPT2LMHeadModel"
        ],
        "attn_pdrop": 0.1,
        "bos_token_id": 1,
        "embd_pdrop": 0.1,
        "eos_token_id": 2,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 1024,
        "n_embd": 1024,
        "n_head": 16,
        "n_layer": 18,
        "n_positions": 1024,
        "vocab_size": 32000,
        "task_specific_params": {
            "text-generation": {
            "do_sample": True,
            "max_length": 50
            }
        }
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(gpt2_config, f, indent=2)
    print("  ✅ config.json créé")
    
    # Créer un README minimal
    readme = f"""---
language: fr
license: mit
tags:
- gpt2
- french
---

# French LLM Fine-tuned (200k steps)

Modèle GPT-2 260M pré-entraîné sur corpus français puis fine-tuné sur conversations.

**Export fixé** - Ce modèle a été exporté avec le mapping correct des clés du checkpoint custom vers HuggingFace Transformers.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

prompt = "Utilisateur: Quelle est la capitale de la France ?\\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Statistiques

- **Architecture**: GPT-2
- **Paramètres**: 260M
- **Layers**: 18
- **Hidden size**: 1024
- **Vocab size**: 32001
- **Context length**: 1024
- **Training steps**: 200k (base) + 200k (fine-tune)
"""
    
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    
    print(f"\n✅ Export terminé: {output_dir}")
    print(f"📊 Taille du modèle: {os.path.getsize(os.path.join(output_dir, 'model.safetensors')) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
