#!/usr/bin/env python3
"""
Test le modèle fine-tuné sur des conversations
"""
import torch
import sys
import re
import pathlib
from pathlib import Path

# Import du modèle et du tokenizer
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from train_subtitles_transformer import SubtitleTrainer, Config

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

def load_model(checkpoint_path, device='cuda'):
    """Charge le modèle depuis un checkpoint"""
    print(f"🔄 Chargement du modèle depuis {checkpoint_path}...")
    
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg = Config(**checkpoint['config'])
    trainer = SubtitleTrainer(cfg)
    trainer.model.load_state_dict(checkpoint['model_state'])
    trainer.model.eval()
    
    params = sum(p.numel() for p in trainer.model.parameters()) / 1e6
    print(f"✅ Modèle chargé : {params:.1f}M paramètres @ step {checkpoint.get('step', '?')}")
    
    return trainer

def generate_conversation(trainer, prompt, max_tokens=200):
    """Génère une réponse conversationnelle"""
    trainer.cfg.sample_prompt = prompt
    trainer.cfg.sample_max_new_tokens = max_tokens
    trainer.cfg.sample_temperature = 0.85
    trainer.cfg.sample_top_k = 50
    
    sample = trainer.sample_text()
    clean_sample = clean_bpe_artifacts(sample)
    
    return clean_sample

def test_conversations():
    """Test le modèle avec plusieurs prompts conversationnels"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("🧪 TEST DES MODÈLES : Base (100k) vs Fine-tuné (115k)")
    print("=" * 80)
    
    # Prompts de test
    test_prompts = [
        "Utilisateur: Bonjour, comment vas-tu ?\nAssistant:",
        "Utilisateur: Explique-moi ce qu'est l'intelligence artificielle.\nAssistant:",
        "Utilisateur: Quel temps fait-il aujourd'hui ?\nAssistant:",
        "Utilisateur: Peux-tu m'aider à résoudre un problème de maths ?\nAssistant:",
        "Utilisateur: Raconte-moi une blague.\nAssistant:",
    ]
    
    # Tester le modèle de base (100k)
    print("\n📊 MODÈLE DE BASE (100,000 steps)")
    print("-" * 80)
    base_trainer = load_model(
        "trained_models/runs/french_medium_base_60k/checkpoint_step_100000.pt",
        device
    )
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n💬 Test {i}:")
        print(f"Prompt: {prompt}")
        output = generate_conversation(base_trainer, prompt, max_tokens=150)
        print(f"\n{output}")
        print("-" * 80)
    
    # Libérer la mémoire
    del base_trainer
    torch.cuda.empty_cache()
    
    # Tester le modèle fine-tuné (115k)
    print("\n\n🎯 MODÈLE FINE-TUNÉ (115,000 steps)")
    print("-" * 80)
    finetuned_trainer = load_model(
        "trained_models/runs/french_medium_finetune_conversations/checkpoint_step_115000.pt",
        device
    )
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n💬 Test {i}:")
        print(f"Prompt: {prompt}")
        output = generate_conversation(finetuned_trainer, prompt, max_tokens=150)
        print(f"\n{output}")
        print("-" * 80)
    
    print("\n✅ Tests terminés !")
    print("\n📈 Comparaison :")
    print("- Le modèle de base (100k) a été entraîné sur du texte général")
    print("- Le modèle fine-tuné (115k) a été entraîné sur 15M tokens de conversations")
    print("- Le fine-tuné devrait montrer une meilleure structure conversationnelle")

if __name__ == "__main__":
    test_conversations()
