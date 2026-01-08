#!/usr/bin/env python3
"""
Script d'inférence pour les modèles V3 (TinyTransformerLM avec 18 layers).
Compatible avec les checkpoints:
- french_medium_optimized_batch4_grad3
- french-llm-from-scratch-V3-mistral
- french_v3_finetune_*
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path
import argparse
from typing import Optional


class TinyTransformerLM(nn.Module):
    """Architecture V3: 18 layers, 260M paramètres"""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 18,
        ff_hidden_dim: int = 4096,
        dropout: float = 0.1
    ):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tok_embed(x)
        x = self.encoder(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits


def generate(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cuda"
) -> str:
    """Génère du texte à partir d'un prompt"""
    model.eval()
    from typing import Any, cast
    tok = cast(Any, tokenizer)
    input_ids = tok.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\n{'='*70}")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")
    generated_text = prompt
    print(prompt, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token_idx)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            token_text = tok.decode(next_token[0])
            print(token_text, end="", flush=True)
            generated_text += token_text
            
            # Stop si EOS
            if hasattr(tok, "eos_token_id") and next_token.item() == tok.eos_token_id:
                break
    
    print("\n" + "="*70)
    return generated_text


def load_model(
    checkpoint_path: str,
    tokenizer_path: str = "data_clean/mistral_tokenizer",
    device: str = "cuda"
) -> tuple[nn.Module, AutoTokenizer]:
    """Charge un modèle V3 depuis un checkpoint"""
    
    # Load tokenizer
    print(f"📥 Chargement tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("✅ Tokenizer chargé")
    
    # Load checkpoint
    print(f"📥 Chargement checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extraire la config
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', 32000)
    embed_dim = config.get('embed_dim', 1024)
    num_heads = config.get('num_heads', 16)
    num_layers = config.get('num_layers', 18)
    ff_hidden_dim = config.get('ff_hidden_dim', 4096)
    
    print(f"🏗️ Architecture:")
    print(f"  - Vocab size: {vocab_size:,}")
    print(f"  - Embed dim: {embed_dim}")
    print(f"  - Num heads: {num_heads}")
    print(f"  - Num layers: {num_layers}")
    print(f"  - FF hidden: {ff_hidden_dim:,}")
    
    # Create model
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_hidden_dim=ff_hidden_dim,
        dropout=0.0  # Désactiver dropout pour inférence
    )
    model.to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    step = checkpoint.get('step', 'unknown')
    print(f"✅ Modèle chargé (step {step:,})")
    
    # Compter paramètres
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Paramètres: {total_params/1e6:.1f}M")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Test d'inférence pour modèles V3")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Chemin vers le checkpoint .pt"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt de test (si non spécifié, teste plusieurs prompts)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data_clean/mistral_tokenizer",
        help="Chemin vers le tokenizer"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Nombre maximum de tokens à générer"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Température de sampling (0.1 = conservateur, 1.0 = créatif)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (0 = désactivé)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    print(f"🔧 Device: {args.device}")
    print()
    
    # Load model
    model, tokenizer = load_model(
        args.checkpoint,
        args.tokenizer,
        args.device
    )
    
    # Test prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        print("\n🎯 Test avec prompts par défaut:\n")
        prompts = [
            "Bonjour, comment allez-vous ?",
            "Paris est la capitale de",
            "L'intelligence artificielle permet de",
            "Je suis très content de vous rencontrer car",
            "La France est un pays",
        ]
    
    # Generate
    for prompt in prompts:
        generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )
        print()


if __name__ == "__main__":
    main()
