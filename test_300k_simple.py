#!/usr/bin/env python3
"""
Test minimaliste du modèle 300k - charge et fixe le checkpoint DDP.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path

print("🔧 Chargement du modèle 300k (mode minimaliste)...")

# Modèle
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=32000, embed_dim=1024, num_heads=16, num_layers=18, ff_hidden=4096):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_hidden,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, x, padding_mask=None):
        x = self.tok_embed(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.ln(x)
        return self.head(x)

# Charger checkpoint
ckpt = torch.load('trained_models/tiny_subtitles_transformer.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')
model_state = ckpt.get('model_state', ckpt.get('model', {}))

# Fixer les clés DDP (module.*  → *)
fixed_state = {}
for k, v in model_state.items():
    if k.startswith('module.'):
        fixed_state[k[7:]] = v
    else:
        fixed_state[k] = v

# Charger modèle
model = TinyTransformer()
model.load_state_dict(fixed_state)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('data_clean/mistral_tokenizer')

print(f"✅ Modèle chargé ({len(fixed_state)} weights)")
print(f"✅ Tokenizer chargé (vocab={tokenizer.vocab_size})")

# Tests
print("\n" + "="*70)
print("🚀 TESTS D'INFÉRENCE (modèle 300k français)")
print("="*70)

prompts = [
    "Bonjour, comment",
    "L'intelligence artificielle",
    "Bonjour je suis"
]

with torch.no_grad():
    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        generated_ids = input_ids[0].clone().tolist()
        for step in range(60):
            logits = model(torch.tensor([generated_ids]).to(device))
            next_logits = logits[0, -1, :] / 0.8
            top_k_logits, top_k_indices = torch.topk(next_logits, 50)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[torch.multinomial(probs, 1)].item()
            generated_ids.append(next_token)
        
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_text = output_text.replace('Ġ', ' ').replace('Ċ', '\n')
        print(f"📤 {output_text[:120]}...")

print("\n✅ Tests complets!")
