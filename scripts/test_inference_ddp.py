#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path
from types import SimpleNamespace
import argparse

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=32000, d_model=1024, nhead=16, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.fc(x)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return SimpleNamespace(loss=loss, logits=logits)

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, device="cuda"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    print(prompt, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            token_text = tokenizer.decode(next_token[0])
            print(token_text, end="", flush=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    print("\n" + "-" * 40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Bonjour, comment ça va ?")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer_path = Path("data_clean/mistral_tokenizer")
    if not tokenizer_path.exists():
        print("Loading default Mistral tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    else:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load Model
    print("Creating model...")
    model = SimpleTransformer() # Default params match training script
    model.to(device)

    # Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle DDP prefix
    state_dict = checkpoint['model_state']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    print("Model loaded successfully!")

    # Generate
    generate(model, tokenizer, args.prompt, device=device)
    generate(model, tokenizer, "L'intelligence artificielle est", device=device)
    generate(model, tokenizer, "La France est un pays", device=device)

if __name__ == "__main__":
    main()
