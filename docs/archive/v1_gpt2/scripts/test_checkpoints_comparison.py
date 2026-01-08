#!/usr/bin/env python3
"""Script pour comparer les générations de différents checkpoints."""

import torch
import sys
from pathlib import Path
from tokenizers import Tokenizer

# Ajouter le path pour importer les modules
sys.path.insert(0, str(Path(__file__).parent))

from scripts.train_subtitles_transformer import TinyTransformerLM, Config

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Charge un checkpoint et retourne le modèle."""
    print(f"\n{'='*60}")
    print(f"Chargement: {checkpoint_path}")
    print(f"{'='*60}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = ckpt.get("config", {})
    
    config = Config(
        vocab_size=config_dict.get("vocab_size", 32000),
        embed_dim=config_dict.get("embed_dim", 1024),
        num_heads=config_dict.get("num_heads", 16),
        num_layers=config_dict.get("num_layers", 18),
        ff_hidden_dim=config_dict.get("ff_hidden_dim", 4096),
        block_size=config_dict.get("block_size", 1024),
        dropout=config_dict.get("dropout", 0.1)
    )
    
    model = TinyTransformerLM(config).to(device)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError("model_state absent dans le checkpoint")
    model.load_state_dict(state)
    model.eval()

    step = ckpt.get("step", "unknown")
    loss = ckpt.get("loss", "N/A")
    print(f"✅ Checkpoint step: {step}, loss: {loss}")

    return model, config

def generate_text(model, tokenizer, prompt: str, max_tokens: int = 100, 
                  temperature: float = 0.8, top_k: int = 50, device: str = "cuda"):
    """Génère du texte à partir d'un prompt."""
    model.eval()
    
    # Tokenize
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoding.ids, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Limite au context window
            context = input_ids[:, -model.cfg.block_size:]
            
            # Forward
            logits = model(context)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop si EOS (token 3)
            if next_token.item() == 3:
                break
    
    # Decode
    generated_ids = input_ids[0].tolist()
    text = tokenizer.decode(generated_ids)
    
    return text

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Tokenizer
    tokenizer_path = "trained_models/tokenizers/fineweb-32k/tokenizer.json"
    print(f"📚 Chargement tokenizer: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Checkpoints à tester
    checkpoints = {
        "100k (Base - ✅ Marche)": "trained_models/runs/french_medium_base_60k/checkpoint_step_100000.pt",
        "195k (Extended)": "trained_models/runs/french_medium_mega_finetune_extended/checkpoint_step_195000.pt",
        "200k (LinkedIn - ❌ Cassé)": "trained_models/runs/french_medium_mega_finetune_200k/checkpoint_step_200000.pt",
    }
    
    # Prompts de test
    test_prompts = [
        "Utilisateur: Bonjour, comment vas-tu ?\nAssistant:",
        "Utilisateur: Quelle est la capitale de la France ?\nAssistant:",
        "Utilisateur: Explique-moi ce qu'est l'intelligence artificielle.\nAssistant:",
    ]
    
    print("\n" + "="*80)
    print("COMPARAISON DES CHECKPOINTS")
    print("="*80)
    
    for name, ckpt_path in checkpoints.items():
        if not Path(ckpt_path).exists():
            print(f"\n⚠️  {name}: Fichier introuvable - {ckpt_path}")
            continue
        
        try:
            model, config = load_model(ckpt_path, device)
            
            print(f"\n📝 Génération avec {name}:")
            print("-" * 60)
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n🔹 Test {i}: {prompt[:50]}...")
                try:
                    output = generate_text(
                        model, tokenizer, prompt,
                        max_tokens=80,
                        temperature=0.8,
                        top_k=50,
                        device=device
                    )
                    # Afficher seulement la réponse (après "Assistant:")
                    if "Assistant:" in output:
                        response = output.split("Assistant:")[-1].strip()
                        print(f"💬 {response[:200]}")
                    else:
                        print(f"💬 {output[:200]}")
                except Exception as e:
                    print(f"❌ Erreur génération: {e}")
            
            # Libérer mémoire
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ Erreur chargement {name}: {e}")
            continue
    
    print("\n" + "="*80)
    print("✅ Comparaison terminée")
    print("="*80)

if __name__ == "__main__":
    main()
