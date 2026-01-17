#!/usr/bin/env python3
"""
Test rapide du modèle 300k avec support des checkpoints DDP.
"""
import torch
import sys
sys.path.insert(0, '/home/vincent/code/repo/french-llm-from-scratch')

from scripts.train_subtitles_transformer import SubtitleTrainer, Config
from transformers import AutoTokenizer
from pathlib import Path

# Config
CHECKPOINT_PATH = Path('trained_models/tiny_subtitles_transformer.pt')
TOKENIZER_PATH = Path('data_clean/mistral_tokenizer')

print("🔧 Chargement du modèle 300k...")
print(f"   Checkpoint: {CHECKPOINT_PATH}")
print(f"   Tokenizer: {TOKENIZER_PATH}")

# Charger via SubtitleTrainer (qui gère déjà les checkpoints DDP)
cfg = Config(
    device='cuda',
    resume_run_dir='trained_models/runs/french_medium_optimized_batch4_grad3'
)

trainer = SubtitleTrainer(cfg)
model = trainer.model
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))

print(f"✅ Modèle chargé (step {trainer.start_step})")
print(f"✅ Tokenizer chargé (vocab={tokenizer.vocab_size})")

# Tests d'inférence
prompts = [
    "Bonjour, je suis",
    "Comment ça marche",
    "L'intelligence artificielle",
]

print("\n" + "="*70)
print("🚀 Tests d'inférence (50 tokens)")
print("="*70)

model.eval()
with torch.no_grad():
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(trainer.device)
        
        generated_ids = input_ids[0].clone().tolist()
        for _ in range(50):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / 0.8
            top_k_logits, top_k_indices = torch.topk(next_token_logits, 50)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[torch.multinomial(probs, 1)].item()
            generated_ids.append(next_token)
            input_ids = torch.tensor([generated_ids]).to(trainer.device)
        
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_text = output_text.replace('Ġ', ' ').replace('Ċ', '\n')
        print(f"📤 {output_text[:150]}...")

print("\n✅ Test complet!")
