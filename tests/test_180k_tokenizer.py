#!/usr/bin/env python3
"""
Comparaison 115k vs 180k avec le tokenizer fineweb-32k correct
"""

import torch
from pathlib import Path
import sys
from tokenizers import Tokenizer

sys.path.append(str(Path(__file__).parent))
from scripts.train_subtitles_transformer import TinyTransformerLM, Config

def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: str = "cuda"):
    """Charge modèle + tokenizer"""
    print(f"📥 {Path(checkpoint_path).name}")
    
    # Config medium
    cfg = Config(
        num_layers=18,
        num_heads=16,
        embed_dim=1024,
        ff_hidden_dim=4096,
        vocab_size=32000,
        block_size=1024,
        dropout=0.1,
    )
    
    # Charger modèle
    model = TinyTransformerLM(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Charger tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"   ✅ Step {checkpoint['step']:,}")
    return model, tokenizer

def generate(model: TinyTransformerLM, tokenizer: Tokenizer, prompt: str, device: str, max_tokens: int = 150) -> str:
    """Génère avec tokenizer BPE"""
    
    # Encoder
    encoded = tokenizer.encode(prompt)
    tokens = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            if tokens.size(1) > 1024:
                tokens = tokens[:, -1024:]
            
            logits = model(tokens)
            logits = logits[:, -1, :]
            
            # Sample
            probs = torch.softmax(logits / 0.8, dim=-1)  # temperature=0.8
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Stop si EOS ou double \n
            if next_token.item() == tokenizer.token_to_id("</s>"):
                break
    
    # Décoder
    generated_ids = tokens[0].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return text

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_path = "trained_models/tokenizers/fineweb-32k/tokenizer.json"
    
    print("=" * 70)
    print("CHARGEMENT")
    print("=" * 70)
    
    model_115k, tok = load_model_and_tokenizer(
        "trained_models/runs/french_medium_finetune_conversations/checkpoint_step_115000.pt",
        tokenizer_path,
        device
    )
    
    model_180k, _ = load_model_and_tokenizer(
        "trained_models/runs/french_medium_mega_finetune/checkpoint_step_180000.pt",
        tokenizer_path,
        device
    )
    
    print()
    
    # Prompts
    prompts = [
        "Utilisateur: Quelle est la capitale de la France ?\nAssistant:",
        "Utilisateur: Explique-moi l'intelligence artificielle en une phrase\nAssistant:",
        "Utilisateur: Donne-moi une recette rapide\nAssistant:",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print("=" * 70)
        print(f"TEST {i}/{len(prompts)}")
        print("=" * 70)
        print(f"💬 {prompt}\n")
        
        print("📌 MODÈLE 115k:")
        print("-" * 70)
        out_115k = generate(model_115k, tok, prompt, device, max_tokens=100)
        print(out_115k)
        print()
        
        print("📌 MODÈLE 180k:")
        print("-" * 70)
        out_180k = generate(model_180k, tok, prompt, device, max_tokens=100)
        print(out_180k)
        print("\n")
    
    print("=" * 70)
    print("✅ TERMINÉ")
    print("=" * 70)
    print("\n📊 Modèle 115k: Loss 4.88 | 15M tokens")
    print("📊 Modèle 180k: Loss 4.89 | 197M tokens (×13 dataset)")

if __name__ == "__main__":
    main()
