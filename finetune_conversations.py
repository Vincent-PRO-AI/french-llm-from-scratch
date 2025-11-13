#!/usr/bin/env python3
"""
Fine-tuning du modèle medium sur conversations
depuis le checkpoint 60k
"""
import sys
from pathlib import Path

# Ajouter le répertoire scripts au path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from train_subtitles_transformer import SubtitleTrainer, Config
import torch

if __name__ == '__main__':
    # Configuration du fine-tuning
    config = Config(
        # Architecture (même que le modèle de base)
        arch_preset='medium',
        num_layers=18,
        num_heads=16,
        embed_dim=1024,
        ff_hidden_dim=4096,
        block_size=1024,
        vocab_size=32000,
        dropout=0.1,
        
        # Données
        tokenizer_path='trained_models/tokenizers/fineweb-32k/tokenizer.json',
        pretokenized_path='data_clean/conversations_tokenized.pt',
        
        # Fine-tuning params
        run_name='french_medium_finetune_chat',
        max_steps=5000,
        batch_size=4,
        lr=0.00001,  # LR très bas pour fine-tuning
        weight_decay=0.0,
        
        # Monitoring
        checkpoint_interval=1000,
        eval_interval=250,
        sample_interval=500,
        sample_prompt="Utilisateur: Bonjour, comment vas-tu ?\nAssistant:",
        sample_max_new_tokens=120,
        sample_temperature=0.8,
        sample_top_k=50,
        
        # GPU
        device='cuda',
        use_amp=True,
    )
    
    print("="*70)
    print("🔥 Fine-tuning sur conversations")
    print("="*70)
    print(f"Checkpoint de base: trained_models/runs/french_medium_50k/checkpoint.pt")
    print(f"Dataset: data_clean/conversations_tokenized.pt (2.4M tokens, 2474 conversations)")
    print(f"Steps: 0 → 5000 (LR: 1e-5)")
    print(f"Run: {config.run_name}")
    print("="*70)
    
    # Créer le trainer
    trainer = SubtitleTrainer(config)
    
    # Charger le checkpoint 60k
    checkpoint_path = Path('trained_models/runs/french_medium_50k/checkpoint_step_60000.pt')
    print(f"\n📦 Chargement checkpoint: {checkpoint_path}")
    
    import pathlib
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    try:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    
    print(f"   Step source: {checkpoint.get('step', '?')}")
    print(f"   Architecture: {checkpoint['config'].get('arch_preset', 'custom')}")
    
    # Charger les poids du modèle (pas l'optimizer pour repartir fresh)
    trainer.model.load_state_dict(checkpoint['model_state'])
    print(f"✅ Modèle chargé\n")
    
    # Lancer le fine-tuning
    print("🚀 Démarrage du fine-tuning...\n")
    trainer.run()
    
    print("\n✅ Fine-tuning terminé!")
    print(f"   Checkpoint final: {trainer.run_checkpoint_path}")
