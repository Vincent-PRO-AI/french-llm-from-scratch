#!/usr/bin/env python3
"""Convertir un checkpoint DDP (avec module.*) en checkpoint simple."""
import torch
import sys

def convert_ddp_checkpoint(input_path, output_path):
    """Enlever les préfixes 'module.' d'un checkpoint DDP."""
    print(f"Chargement checkpoint DDP: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    if 'model_state' in checkpoint:
        model_state = checkpoint['model_state']
    else:
        model_state = checkpoint
    
    # Vérifier si c'est un checkpoint DDP (clés avec 'module.')
    has_module_prefix = any(k.startswith('module.') for k in model_state.keys())
    
    if has_module_prefix:
        print("✓ Checkpoint DDP détecté (clés avec 'module.')")
        # Enlever les préfixes
        new_state = {}
        for key, value in model_state.items():
            new_key = key.replace('module.', '', 1)  # Enlever premier 'module.'
            new_state[new_key] = value
        model_state = new_state
        print(f"✓ Converti {len(new_state)} clés")
    else:
        print("✓ Checkpoint sans DDP")
    
    # Reconstruire le checkpoint si nécessaire
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        checkpoint['model_state'] = model_state
    else:
        checkpoint = model_state
    
    print(f"Sauvegarde: {output_path}")
    torch.save(checkpoint, output_path)
    print("✓ Fait!")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python fix_checkpoint_ddp.py <input> <output>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_ddp_checkpoint(input_path, output_path)
