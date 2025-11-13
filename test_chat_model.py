#!/usr/bin/env python3
"""Test du modèle fine-tuné sur conversations"""
import re
import torch
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from train_subtitles_transformer import SubtitleTrainer, Config

def clean_bpe_artifacts(text):
    """Nettoie les artefacts BPE"""
    text = text.replace('Ġ', ' ')
    text = text.replace('Ċ', '\n')
    text = text.replace('âĢĶ', '—')
    text = text.replace('âĢĻ', "'")
    text = re.sub(r' +', ' ', text)
    text = re.sub(r' \n ', '\n', text)
    return text.strip()

if __name__ == '__main__':
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    
    checkpoint_path = 'trained_models/runs/french_medium_finetune_chat/checkpoint.pt'
    print(f'📦 Chargement: {checkpoint_path}\n')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"📊 Modèle fine-tuné: {checkpoint['config']['arch_preset']} @ step {checkpoint.get('step', '?')}")
    print(f"   Dataset: conversations (2.4M tokens, 2474 dialogues)")
    print(f"   Fine-tuning: {checkpoint.get('step', 0)} steps supplémentaires\n")
    
    cfg = Config(**checkpoint['config'])
    trainer = SubtitleTrainer(cfg)
    trainer.model.load_state_dict(checkpoint['model_state'])
    trainer.model.eval()
    
    print("="*70)
    print("💬 Test de dialogue:")
    print("="*70)
    
    # Prompts conversationnels
    prompts = [
        'Utilisateur: Bonjour, comment vas-tu ?\nAssistant:',
        'Utilisateur: Quelle est la capitale de la France ?\nAssistant:',
        "Utilisateur: Explique-moi l'intelligence artificielle.\nAssistant:",
        'Utilisateur: Quel temps fait-il aujourd\'hui ?\nAssistant:',
        'Utilisateur: Raconte-moi une blague.\nAssistant:',
    ]
    
    for prompt in prompts:
        trainer.cfg.sample_prompt = prompt
        trainer.cfg.sample_max_new_tokens = 100
        trainer.cfg.sample_temperature = 0.85
        sample = trainer.sample_text()
        clean_sample = clean_bpe_artifacts(sample)
        
        # Extraire uniquement la réponse de l'assistant
        if 'Assistant:' in clean_sample:
            parts = clean_sample.split('Assistant:', 1)
            if len(parts) > 1:
                response = parts[1].strip()
                # Couper à la prochaine mention "Utilisateur:" si présente
                if 'Utilisateur:' in response:
                    response = response.split('Utilisateur:')[0].strip()
                
                print(f"\n{prompt.split('Assistant:')[0]}Assistant:")
                print(f"  {response}")
                print("-"*70)
        else:
            print(f"\n{prompt}")
            print(f"  {clean_sample}")
            print("-"*70)
