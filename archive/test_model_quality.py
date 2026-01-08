#!/usr/bin/env python3
"""
🧪 Model Quality Assessment - Évaluation sur 50 prompts
"""
import sys
sys.path.insert(0, '/home/vincent/code/repo/french-llm-from-scratch')

import torch
from pathlib import Path
from scripts.train_subtitles_transformer import Config, TinyTransformerLM
from transformers import AutoTokenizer
import json
import time
from datetime import datetime

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║         🧪 MODEL QUALITY ASSESSMENT - 50 Common French AI Prompts         ║
║                   Évaluation de la Qualité du Modèle                      ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# Checkpoint path
CHECKPOINT_PATH = Path('trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint.pt')
TOKENIZER_PATH = Path('data_clean/mistral_tokenizer')

print("[1/4] Chargement du checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
ck_cfg = checkpoint.get('config', {})
ck_step = checkpoint.get('step', 'unknown')
print(f"      ✓ Step: {ck_step}")

print("[2/4] Initialisation du modèle avec config du checkpoint...")
cfg = Config()
if isinstance(ck_cfg, dict):
    if 'n_embd' in ck_cfg: cfg.embed_dim = ck_cfg['n_embd']
    if 'n_layer' in ck_cfg: cfg.num_layers = ck_cfg['n_layer']
    if 'n_head' in ck_cfg: cfg.num_heads = ck_cfg['n_head']
    if 'ff_hidden_dim' in ck_cfg: cfg.ff_hidden_dim = ck_cfg['ff_hidden_dim']
    if 'vocab_size' in ck_cfg: cfg.vocab_size = ck_cfg['vocab_size']
    if 'block_size' in ck_cfg: cfg.block_size = ck_cfg['block_size']

print(f"      Architecture: {cfg.num_layers}L {cfg.num_heads}H {cfg.embed_dim}E {cfg.vocab_size}V")

cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.tokenizer_path = TOKENIZER_PATH
model = TinyTransformerLM(cfg).to(cfg.device)
model_state = checkpoint.get('model_state')
model.load_state_dict(model_state, strict=True)
model.eval()
print(f"      ✓ Modèle chargé sur {cfg.device}")

print("[3/4] Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
print(f"      ✓ Vocab: {tokenizer.vocab_size} tokens")

PROMPTS = [
    "L'histoire de la civilisation française est",
    "Les grands auteurs français ont écrit",
    "La Révolution française a changé",
    "Napoléon Bonaparte était",
    "Le Moyen Âge était une période",
    "L'intelligence artificielle peut",
    "La physique quantique révèle",
    "Les mathématiques permettent",
    "L'astronomie montre",
    "La biologie étudie",
    "La philosophie examine",
    "Socrate pensait que",
    "La liberté humaine signifie",
    "La morale nous oblige",
    "La conscience est",
    "La démocratie fonctionne",
    "Les droits humains incluent",
    "L'égalité demande",
    "La justice protège",
    "Un bon gouvernement",
    "L'économie est basée",
    "Le marché encourage",
    "Le commerce bénéficie",
    "L'inflation se produit",
    "La croissance dépend",
    "L'école est importante",
    "L'apprentissage continue",
    "Les universités forment",
    "La formation technique",
    "L'éducation numérique",
    "Le changement climatique menace",
    "La protection exige",
    "Les énergies offrent",
    "La biodiversité est",
    "La pollution détruit",
    "L'art exprime",
    "La musique peut",
    "La peinture montrait",
    "Le cinéma raconte",
    "La littérature permet",
    "Le stress affecte",
    "L'empathie est",
    "La résilience permet",
    "La motivation vient",
    "L'anxiété peut",
    "Les réseaux ont",
    "La cybersécurité protège",
    "La blockchain révolutionne",
    "L'intelligence aide",
    "La réalité virtuelle",
]

print(f"[4/4] Test de {len(PROMPTS)} prompts...")
print()

results = []
latencies = []
token_counts = []
success_count = 0

with torch.no_grad():
    for idx, prompt in enumerate(PROMPTS, 1):
        pct = int(40 * idx / len(PROMPTS))
        bar = "█" * pct + "░" * (40 - pct)
        print(f"\r[{bar}] {idx:2d}/{len(PROMPTS)}", end="", flush=True)
        
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt').to(cfg.device)
            input_len = len(inputs[0])
            
            start_time = time.time()
            generated = model.generate(
                input_ids=inputs,
                max_new_tokens=25,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id or 2
            )
            latency = time.time() - start_time
            
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            tokens_gen = len(generated[0]) - input_len
            
            latencies.append(latency)
            token_counts.append(tokens_gen)
            if tokens_gen >= 3:
                success_count += 1
            
            results.append({'prompt': prompt[:40], 'tokens': tokens_gen, 'latency': round(latency, 2), 'ok': tokens_gen >= 3})
            
        except Exception as e:
            results.append({'prompt': prompt[:40], 'error': str(e)[:40]})

print()
print()
print("="*80)
print("📊 RÉSULTATS")
print("="*80)

print(f"\n✓ Succès: {success_count}/{len(PROMPTS)} ({100*success_count/len(PROMPTS):.0f}%)")

if latencies:
    avg_lat = sum(latencies) / len(latencies)
    print(f"⏱️  Latence: {avg_lat:.2f}s (min: {min(latencies):.2f}s, max: {max(latencies):.2f}s)")

if token_counts:
    avg_tok = sum(token_counts) / len(token_counts)
    print(f"📈 Tokens: {avg_tok:.1f} (min: {min(token_counts)}, max: {max(token_counts)})")

print("\n" + "="*80)
prod_ready = (success_count >= 40)
print(f"{'🚀 PRÊT PRODUCTION' if prod_ready else '⚠️  À AMÉLIORER'}")
print("="*80)

with open('TEST_RESULTS.json', 'w') as f:
    json.dump({
        'total': len(PROMPTS),
        'success': success_count,
        'rate': f"{100*success_count/len(PROMPTS):.0f}%",
        'ready': prod_ready
    }, f)

print(f"\n✓ Résultats: TEST_RESULTS.json")
