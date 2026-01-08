#!/usr/bin/env python3
"""
Script d'évaluation simple pour tester la qualité du modèle.
"""
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=32000, d_model=1024, nhead=16, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.fc(x)
        return logits

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=40):
    """Generate text with better sampling parameters"""
    model.eval()
    device = next(model.parameters()).device
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_idx]
            
            # Append token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--max-length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = Path("data_clean/mistral_tokenizer")
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create model
    print("Creating model...")
    model = SimpleTransformer(vocab_size=len(tokenizer))
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    
    # Handle DDP prefix
    state_dict = checkpoint['model_state']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print("\n" + "="*80)
    print("EVALUATION DU MODÈLE")
    print("="*80)
    
    # Test prompts
    prompts = [
        "Bonjour, comment allez-vous ?",
        "La France est",
        "L'intelligence artificielle",
        "Il était une fois",
        "Le président de la République",
        "Paris est la capitale",
    ]
    
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("-" * 80)
        generated = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(generated)
        print("-" * 80)

if __name__ == "__main__":
    main()
