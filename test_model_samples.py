#!/usr/bin/env python3
"""Test de génération avec le modèle entraîné"""
import re
import torch
import pathlib
from scripts.train_subtitles_transformer import SubtitleTrainer, Config

def clean_bpe_artifacts(text):
    """Nettoie les artefacts BPE du texte généré"""
    text = text.replace('Ġ', ' ')  # Espace GPT-style
    text = text.replace('Ċ', '\n')  # Newline
    text = text.replace('âĢĶ', '—')  # Em-dash
    text = text.replace('âĢĻ', "'")  # Apostrophe
    # Supprimer espaces multiples
    text = re.sub(r' +', ' ', text)
    text = re.sub(r' \n ', '\n', text)
    return text.strip()

if __name__ == '__main__':
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    
    checkpoint_path = 'trained_models/runs/french_medium_50k/checkpoint.pt'
    print(f'Chargement: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    print(f"\n📊 Modèle: {checkpoint['config']['arch_preset']} "
          f"({checkpoint['config']['num_layers']}L, {checkpoint['config']['num_heads']}H, "
          f"{checkpoint['config']['embed_dim']}D) @ step {checkpoint.get('step', '?')}")
    
    cfg = Config(**checkpoint['config'])
    trainer = SubtitleTrainer(cfg)
    trainer.model.load_state_dict(checkpoint['model_state'])
    trainer.model.eval()
    
    print("\n" + "="*70)
    print("🎯 Échantillons générés:")
    print("="*70)
    
    prompts = [
        'Bonjour, je suis',
        'La France est',
        "L'intelligence artificielle",
        'Dans le futur,',
        'Le président a déclaré que'
    ]
    
    for prompt in prompts:
        trainer.cfg.sample_prompt = prompt
        sample = trainer.sample_text()
        clean_sample = clean_bpe_artifacts(sample)
        print(f"\n💬 Prompt: '{prompt}'")
        print(clean_sample)
        print("-"*70)
