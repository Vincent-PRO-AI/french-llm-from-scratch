#!/usr/bin/env python3
"""
Diagnostiquer les tenseurs manquants dans la conversion.
Compare checkpoint original vs GGUF converti.
"""

import json
import subprocess
from pathlib import Path
import re

def get_gguf_tensors(gguf_path):
    """Extraire la liste des tenseurs du GGUF avec gguf-dump."""
    try:
        # Essayer avec le script python gguf-dump
        result = subprocess.run(
            ['python', '/home/vincent/llama.cpp/gguf-py/scripts/gguf-dump.py', str(gguf_path)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
             # Essayer avec le binaire
            result = subprocess.run(
                ['/home/vincent/llama.cpp/build/bin/gguf-dump', str(gguf_path)],
                capture_output=True, text=True, timeout=60
            )
            
        lines = result.stdout.split('\n')
        tensors = []
        for line in lines:
            # Pattern pour trouver les noms de tenseurs
            # Ex: name: 'blk.0.attn_q.weight'
            match = re.search(r"name:\s*'([^']+)'|name:\s*\"([^\"]+)\"", line)
            if match:
                name = match.group(1) or match.group(2)
                tensors.append(name)
        return tensors
    except Exception as e:
        print(f"❌ Erreur gguf-dump: {e}")
        return []

def analyze_expected_tensors(config):
    """Générer la liste attendue des tenseurs pour Mistral."""
    expected = []
    
    # Embeddings
    expected.append('model.embed_tokens.weight')
    
    # 18 couches (num_hidden_layers)
    num_layers = config.get('num_hidden_layers', 18)
    for i in range(num_layers):
        # Attention
        expected.append(f'model.layers.{i}.input_layernorm.weight')
        expected.append(f'model.layers.{i}.self_attn.q_proj.weight')
        expected.append(f'model.layers.{i}.self_attn.k_proj.weight')
        expected.append(f'model.layers.{i}.self_attn.v_proj.weight')
        expected.append(f'model.layers.{i}.self_attn.o_proj.weight')
        expected.append(f'model.layers.{i}.post_attention_layernorm.weight')
        
        # FFN (SwiGLU: gate + up + down)
        expected.append(f'model.layers.{i}.mlp.gate_proj.weight')
        expected.append(f'model.layers.{i}.mlp.up_proj.weight')
        expected.append(f'model.layers.{i}.mlp.down_proj.weight')
    
    # Final norm et head
    expected.append('model.norm.weight')
    expected.append('lm_head.weight')
    
    return expected

def main():
    print("=" * 80)
    print("🔍 DIAGNOSTIC: Comparaison tenseurs attendus vs GGUF")
    print("=" * 80)
    
    # Config attendue
    config_path = Path("trained_models/french-llm-v3-mistral-hf-compatible/config.json")
    if not config_path.exists():
        print(f"❌ Config non trouvée: {config_path}")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\n📋 Config:")
    print(f"   Layers: {config.get('num_hidden_layers', 18)}")
    
    # Vérifier GGUF
    gguf_path = Path("trained_models/french-llm-v3-mistral-f16-v4.gguf")
    if gguf_path.exists():
        print(f"\n📁 Fichier GGUF: {gguf_path}")
        print(f"   Taille: {gguf_path.stat().st_size / 1e6:.1f} MB")
        
        tensors = get_gguf_tensors(gguf_path)
        print(f"\n📊 Tenseurs trouvés dans GGUF: {len(tensors)}")
        
        # Analyser les types de tenseurs
        weights = [t for t in tensors if 'weight' in t]
        biases = [t for t in tensors if 'bias' in t]
        print(f"   - Weights: {len(weights)}")
        print(f"   - Biases: {len(biases)}")
        
        if len(tensors) > 0:
            print("\nExemples de tenseurs:")
            for t in sorted(tensors)[:10]:
                print(f"   - {t}")
                
            print("\nExemples de biais:")
            for t in sorted(biases)[:10]:
                print(f"   - {t}")

if __name__ == "__main__":
    main()
