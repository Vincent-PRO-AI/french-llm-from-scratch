#!/usr/bin/env python3
"""
Comparaison des modèles 115k vs 180k steps
Test sur conversations françaises pour mesurer l'amélioration qualitative
"""

import torch
from pathlib import Path
import sys

# Ajouter le chemin pour importer les classes nécessaires
sys.path.append(str(Path(__file__).parent))
from scripts.train_subtitles_transformer import TinyTransformerLM, Config

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Charge un modèle depuis un checkpoint"""
    print(f"📥 Chargement: {Path(checkpoint_path).name}")
    
    # Config pour modèle medium (260M params)
    cfg = Config(
        num_layers=18,
        num_heads=16,
        embed_dim=1024,
        ff_hidden_dim=4096,
        vocab_size=32000,
        block_size=1024,
        dropout=0.1,
    )
    
    # Créer modèle
    model = TinyTransformerLM(cfg).to(device)
    
    # Charger checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"   ✅ Chargé (step {checkpoint['step']:,})")
    return model, cfg

def generate_conversation(model: TinyTransformerLM, device: str, prompt: str, max_seq_len: int = 1024, max_tokens: int = 300) -> str:
    """Génère une conversation à partir d'un prompt"""
    
    # Encoder le prompt
    input_bytes = prompt.encode("utf-8")
    tokens = torch.tensor(list(input_bytes), dtype=torch.long, device=device).unsqueeze(0)
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            if tokens.size(1) > max_seq_len:
                tokens = tokens[:, -max_seq_len:]
            
            logits = model(tokens)
            logits = logits[:, -1, :]  # Dernier token
            
            # Sampling
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            token_val = int(next_token.item())
            if token_val < 256:  # Byte valide
                generated.append(token_val)
                
                # Stop si double retour ligne
                if len(generated) >= 2 and generated[-2:] == [10, 10]:
                    break
    
    # Décoder
    full_text = (input_bytes + bytes(generated)).decode("utf-8", errors="ignore")
    return full_text

def test_conversations():
    """Compare les deux modèles sur plusieurs prompts de test"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎮 Device: {device}\n")
    
    # Chemins des checkpoints
    checkpoint_115k = "trained_models/runs/french_medium_finetune_conversations/checkpoint_step_115000.pt"
    checkpoint_180k = "trained_models/runs/french_medium_mega_finetune/checkpoint_step_180000.pt"
    
    # Charger les modèles
    print("=" * 70)
    print("CHARGEMENT DES MODÈLES")
    print("=" * 70)
    model_115k, cfg = load_model(checkpoint_115k, device)
    model_180k, _ = load_model(checkpoint_180k, device)
    print()
    
    # Prompts de test variés
    test_prompts = [
        "Utilisateur: Explique-moi comment fonctionne l'intelligence artificielle\nAssistant:",
        "Utilisateur: Quelle est la capitale de la France ?\nAssistant:",
        "Utilisateur: Donne-moi une recette de crêpes\nAssistant:",
        "Utilisateur: Comment apprendre la programmation Python ?\nAssistant:",
        "Utilisateur: Raconte-moi une blague\nAssistant:",
    ]
    
    # Tester chaque prompt
    for i, prompt in enumerate(test_prompts, 1):
        print("=" * 70)
        print(f"TEST {i}/5")
        print("=" * 70)
        print(f"💬 Prompt: {prompt[:50]}...")
        print()
        
        # Génération 115k
        print("📌 MODÈLE 115k (15M tokens):")
        print("-" * 70)
        response_115k = generate_conversation(model_115k, device, prompt, max_tokens=200)
        # Nettoyer et afficher
        clean_115k = response_115k.replace("Ġ", " ").replace("Ċ", "\n")
        print(clean_115k)
        print()
        
        # Génération 180k
        print("📌 MODÈLE 180k (197M tokens):")
        print("-" * 70)
        response_180k = generate_conversation(model_180k, device, prompt, max_tokens=200)
        # Nettoyer et afficher
        clean_180k = response_180k.replace("Ġ", " ").replace("Ċ", "\n")
        print(clean_180k)
        print()
        print()
    
    print("=" * 70)
    print("✅ COMPARAISON TERMINÉE")
    print("=" * 70)
    print("\n📊 MÉTRIQUES:")
    print("   Modèle 115k: Loss 4.88 | Dataset 15M tokens")
    print("   Modèle 180k: Loss 4.89 | Dataset 197M tokens (×13)")
    print("\n💡 OBSERVATIONS:")
    print("   • Comparer la cohérence des réponses")
    print("   • Vérifier la structure Utilisateur/Assistant")
    print("   • Évaluer la qualité du français")
    print("   • Détecter les répétitions ou artefacts")

if __name__ == "__main__":
    test_conversations()
