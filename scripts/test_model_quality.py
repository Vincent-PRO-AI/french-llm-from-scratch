#!/usr/bin/env python3
"""
Tester le modèle PyTorch converti pour évaluer sa qualité.
Génère du texte en français avec différents prompts.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json


def test_model_inference():
    """Tester le modèle avec plusieurs prompts français."""
    
    print("=" * 80)
    print("🧪 TEST DU MODÈLE FRANÇAIS LLM V3-MISTRAL")
    print("=" * 80)
    
    model_dir = Path("trained_models/french-llm-v3-mistral-hf-compatible")
    
    if not model_dir.exists():
        print(f"❌ Dossier modèle non trouvé: {model_dir}")
        return
    
    # Charger config et tokenizer
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\n📋 Configuration du modèle:")
    print(f"   Architecture: {config.get('architectures', ['Unknown'])}")
    print(f"   Vocab size: {config['vocab_size']}")
    print(f"   Hidden size: {config['hidden_size']}")
    print(f"   Num layers: {config['num_hidden_layers']}")
    print(f"   Attention heads: {config['num_attention_heads']}")
    print(f"   Max position: {config['max_position_embeddings']}")
    
    print(f"\n📥 Chargement du modèle...")
    try:
        # Charger le modèle
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"   ✅ Modèle chargé")
        
        # Charger tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"   ✅ Tokenizer chargé")
        
    except Exception as e:
        print(f"   ❌ Erreur lors du chargement: {e}")
        return
    
    # Prompts de test
    prompts = [
        "Bonjour, je suis un assistant",
        "Paris est la capitale",
        "La machine learning est",
        "Un jour, il y avait",
    ]
    
    print(f"\n🎯 Génération de texte avec {len(prompts)} prompts:")
    print("-" * 80)
    
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📝 Prompt {i}: \"{prompt}\"")
            
            try:
                # Tokeniser le prompt
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # Générer du texte
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Décoder la sortie
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Afficher seulement la partie générée
                if len(generated_text) > len(prompt):
                    generated_part = generated_text[len(prompt):].strip()
                else:
                    generated_part = generated_text
                
                print(f"   ➜ {generated_part}")
                
            except Exception as e:
                print(f"   ❌ Erreur génération: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Test terminé!")
    print("=" * 80)


def analyze_model_parameters():
    """Analyser les paramètres du modèle."""
    
    print("\n📊 Analyse des paramètres du modèle:")
    print("-" * 80)
    
    model_dir = Path("trained_models/french-llm-v3-mistral-hf-compatible")
    model_path = model_dir / "pytorch_model.bin"
    
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        total_params = sum(p.numel() for p in state_dict.values())
        total_mb = sum(p.nbytes for p in state_dict.values()) / 1e6
        
        print(f"Total tensors: {len(state_dict)}")
        print(f"Total parameters: {total_params:,}")
        print(f"Total size (float16): {total_mb:.1f} MB")
        
        # Détails par type
        weight_params = sum(state_dict[k].numel() for k in state_dict.keys() if 'weight' in k)
        bias_params = sum(state_dict[k].numel() for k in state_dict.keys() if 'bias' in k)
        
        print(f"\nDécomposition:")
        print(f"  - Weights: {weight_params:,} ({weight_params/total_params*100:.1f}%)")
        print(f"  - Biases: {bias_params:,} ({bias_params/total_params*100:.1f}%)")
        
        # Vérifier les dimensions critiques
        print(f"\n🔍 Vérifications dimensionnelles:")
        
        # Embeddings
        embed_weight = state_dict.get('model.embed_tokens.weight')
        if embed_weight is not None:
            print(f"  ✓ Embeddings: {tuple(embed_weight.shape)}")
        
        # Première couche attention
        q_proj = state_dict.get('model.layers.0.self_attn.q_proj.weight')
        if q_proj is not None:
            print(f"  ✓ Q-proj (layer 0): {tuple(q_proj.shape)}")
        
        # FFN
        gate = state_dict.get('model.layers.0.mlp.gate_proj.weight')
        if gate is not None:
            print(f"  ✓ Gate-proj (layer 0): {tuple(gate.shape)}")
        
        # LM Head
        lm_head = state_dict.get('lm_head.weight')
        if lm_head is not None:
            print(f"  ✓ LM Head: {tuple(lm_head.shape)}")
        
        # Norme finale
        norm = state_dict.get('model.norm.weight')
        if norm is not None:
            print(f"  ✓ Final norm: {tuple(norm.shape)}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    analyze_model_parameters()
    test_model_inference()
