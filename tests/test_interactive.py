#!/usr/bin/env python3
"""Test interactif du modèle base 100k"""
import re
import torch
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from train_subtitles_transformer import SubtitleTrainer, Config

def clean_text(text):
    """Nettoie les artefacts BPE"""
    text = text.replace('Ġ', ' ').replace('Ċ', '\n')
    text = text.replace('âĢĶ', '—').replace('âĢĻ', "'")
    text = text.replace('Â«', '«').replace('Â»', '»')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n +', '\n', text)
    return text.strip()

def load_model(checkpoint_path):
    """Charge le modèle depuis un checkpoint"""
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    print(f'📦 Chargement: {checkpoint_path}')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    cfg = Config(**checkpoint['config'])
    trainer = SubtitleTrainer(cfg)
    trainer.model.load_state_dict(checkpoint['model_state'])
    trainer.model.eval()
    
    print(f"✅ Modèle: {cfg.arch_preset} @ step {checkpoint.get('step', 'unknown')}\n")
    return trainer

def generate(trainer, prompt, max_tokens=150, temperature=0.8):
    """Génère du texte à partir d'un prompt"""
    trainer.cfg.sample_prompt = prompt
    trainer.cfg.sample_max_new_tokens = max_tokens
    trainer.cfg.sample_temperature = temperature
    sample = trainer.sample_text()
    return clean_text(sample)

if __name__ == '__main__':
    # Charger le modèle
    checkpoint_path = 'trained_models/runs/french_medium_base_60k/checkpoint_step_100000.pt'
    trainer = load_model(checkpoint_path)
    
    print("="*70)
    print("🤖 Mode test interactif - Modèle base 100k steps")
    print("="*70)
    print("Commandes:")
    print("  - Entrez un prompt pour générer du texte")
    print("  - 'quit' ou 'exit' pour quitter")
    print("  - 'temp X' pour changer la température (ex: temp 1.0)")
    print("  - 'tokens X' pour changer le nombre de tokens (ex: tokens 200)")
    print("="*70)
    
    temperature = 0.8
    max_tokens = 150
    
    while True:
        try:
            prompt = input(f"\n💬 Prompt [temp={temperature}, tokens={max_tokens}]: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir!")
                break
            
            if prompt.startswith('temp '):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"✅ Température mise à {temperature}")
                except:
                    print("❌ Format invalide. Utiliser: temp 0.8")
                continue
            
            if prompt.startswith('tokens '):
                try:
                    max_tokens = int(prompt.split()[1])
                    print(f"✅ Max tokens mis à {max_tokens}")
                except:
                    print("❌ Format invalide. Utiliser: tokens 200")
                continue
            
            print(f"\n🔄 Génération en cours...")
            result = generate(trainer, prompt, max_tokens, temperature)
            print(f"\n📝 Résultat:\n{result}")
            print("-"*70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
