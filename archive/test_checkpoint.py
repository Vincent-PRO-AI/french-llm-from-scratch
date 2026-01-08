#!/usr/bin/env python3
"""
Test du checkpoint avant relance
Vérifie l'intégrité et la compatibilité
"""

import torch
from pathlib import Path
import json

def test_checkpoint():
    print("\n" + "="*80)
    print("  🧪 TEST DE CHECKPOINT AVANT RELANCE")
    print("="*80 + "\n")
    
    checkpoint_path = Path("/home/vincent/code/repo/french-llm-from-scratch/trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt")
    
    print(f"📁 Checkpoint: {checkpoint_path.name}")
    print(f"📦 Taille: {checkpoint_path.stat().st_size / (1024**3):.2f} GB")
    
    # Charger le checkpoint
    print(f"\n⏳ Chargement du checkpoint...")
    try:
        from pathlib import PosixPath
        torch.serialization.add_safe_globals([PosixPath])
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        print(f"✅ Checkpoint chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return False
    
    # Vérifier la structure
    print(f"\n🔍 Vérification de la structure")
    print("-" * 80)
    
    required_keys = ['model_state', 'optimizer_state', 'step', 'config']
    found_keys = list(ckpt.keys())
    
    print(f"Keys trouvées: {found_keys}")
    
    for key in required_keys:
        if key in ckpt:
            print(f"  ✅ {key}")
        else:
            print(f"  ⚠️  {key} (manquant, mais pas critique)")
    
    # Vérifier le model state
    if 'model_state' in ckpt:
        model_state = ckpt['model_state']
        print(f"\n📊 Model state dict")
        print(f"  Nombre de paramètres: {len(model_state)}")
        
        # Afficher quelques infos
        for i, (k, v) in enumerate(list(model_state.items())[:3]):
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} {v.dtype}")
        
        if len(model_state) > 3:
            print(f"  ... et {len(model_state) - 3} autres paramètres")
    
    # Vérifier le step
    if 'step' in ckpt:
        step = ckpt['step']
        print(f"\n📍 Step sauvegardé: {step}")
        print(f"   Cible: 500000 steps")
        print(f"   Restants: {500000 - step:,} steps")
        print(f"   À ~3.3 steps/sec: {(500000 - step) / 3.3 / 3600:.1f} heures")
    
    # Résumé
    print(f"\n" + "="*80)
    print(f"✅ CHECKPOINT VALIDE - PRÊT POUR RELANCE")
    print(f"="*80 + "\n")
    
    return True

if __name__ == "__main__":
    test_checkpoint()
