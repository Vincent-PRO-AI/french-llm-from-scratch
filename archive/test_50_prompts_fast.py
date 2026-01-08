#!/usr/bin/env python3
"""
🧪 Fast Model Testing - 50 Prompts (Optimized)
Test rapide du modèle sur 50 prompts variés
"""
import sys
sys.path.insert(0, '/home/vincent/code/repo/french-llm-from-scratch')

import torch
from pathlib import Path
from scripts.train_subtitles_transformer import Config, SubtitleTrainer
from transformers import AutoTokenizer
import json
import time
from datetime import datetime

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              🧪 QUICK MODEL TESTING - 50 Common AI Prompts              ║
║                      Évaluation Rapide du Modèle                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

# Configuration
CHECKPOINT_PATH = Path('trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint.pt')
TOKENIZER_PATH = Path('data_clean/mistral_tokenizer')

print("[1/4] Chargement du checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda', weights_only=False)
ck_cfg = checkpoint.get('config', {})
ck_step = checkpoint.get('step', 'unknown')
print(f"      ✓ Step: {ck_step}")

print("[2/4] Initialisation du modèle...")
cfg = Config()
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.tokenizer_path = TOKENIZER_PATH

if isinstance(ck_cfg, dict):
    for k, v in [('embed_dim', 'n_embd'), ('num_layers', 'n_layer'), ('num_heads', 'n_head'),
                  ('ff_hidden_dim', 'ff_hidden_dim'), ('vocab_size', 'vocab_size'), ('block_size', 'block_size')]:
        setattr(cfg, k, ck_cfg.get(v, getattr(cfg, k)))

trainer = SubtitleTrainer(cfg)
model = trainer.model.to(cfg.device).eval()
model_state = checkpoint.get('model_state') or checkpoint.get('model')
model.load_state_dict(model_state, strict=False)
print(f"      ✓ Device: {cfg.device}")

print("[3/4] Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
print(f"      ✓ Vocab: {tokenizer.vocab_size} tokens")

# 50 prompts
PROMPTS = [
    # Littérature (5)
    "L'histoire de la civilisation française",
    "Les grands auteurs de la littérature française",
    "La Révolution française a changé",
    "Napoléon Bonaparte était",
    "Le Moyen Âge était une période",
    # Science (5)
    "L'intelligence artificielle est",
    "La physique quantique explique",
    "Les mathématiques sont le fondement",
    "L'astronomie nous montre que",
    "La biologie moderne révèle",
    # Philosophie (5)
    "La question fondamentale de la philosophie est",
    "Socrate croyait que",
    "La liberté humaine signifie",
    "La morale est basée sur",
    "Qu'est-ce que la conscience",
    # Société (5)
    "La démocratie est un système",
    "Les droits humains sont",
    "L'égalité dans la société signifie",
    "La justice sociale demande",
    "La gouvernance efficace nécessite",
    # Économie (5)
    "L'économie mondiale fonctionne",
    "Le capitalisme encourage",
    "Le commerce international favorise",
    "L'inflation affecte",
    "La productivité augmente quand",
    # Éducation (5)
    "L'éducation est essentielle pour",
    "L'apprentissage tout au long de la vie",
    "Les universités devraient préparer",
    "La formation professionnelle aide",
    "L'éducation numérique transforme",
    # Environnement (5)
    "Le changement climatique est causé par",
    "La protection de l'environnement demande",
    "Les énergies renouvelables offrent",
    "La biodiversité est importante car",
    "La pollution affecte la santé",
    # Arts (5)
    "L'art exprime",
    "La musique a le pouvoir de",
    "La peinture Renaissance était",
    "Le cinéma moderne montre",
    "La danse traduit",
    # Psychologie (5)
    "La psychologie étudie",
    "Le stress mental peut être réduit par",
    "L'empathie est la capacité de",
    "La résilience signifie",
    "La dépression affecte",
    # Technologie (5)
    "Les réseaux sociaux ont transformé",
    "La cybersécurité protège",
    "La blockchain révolutionne",
    "L'Internet des objets connecte",
    "La réalité virtuelle permet",
]

print(f"[4/4] Test de {len(PROMPTS)} prompts...")
print()

results = []
latencies = []
token_counts = []
success_count = 0

with torch.no_grad():
    for idx, prompt in enumerate(PROMPTS, 1):
        # Progress
        pct = int(50 * idx / len(PROMPTS))
        bar = "█" * pct + "░" * (50 - pct)
        print(f"\r[{bar}] {idx:2d}/50", end="", flush=True)
        
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt').to(cfg.device)
            input_len = len(inputs[0])
            
            start_time = time.time()
            generated = model.generate(
                input_id=inputs,
                max_new_tokens=40,  # Réduit pour rapidité
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id or 2
            )
            latency = time.time() - start_time
            
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            tokens_gen = len(generated[0]) - input_len
            
            results.append({
                'prompt': prompt[:40],
                'output': output_text[:120],
                'tokens': tokens_gen,
                'latency': round(latency, 2)
            })
            
            latencies.append(latency)
            token_counts.append(tokens_gen)
            if tokens_gen > 5:
                success_count += 1
                
        except Exception as e:
            results.append({
                'prompt': prompt[:40],
                'error': str(e)[:50]
            })

print()
print()
print("="*80)
print("📊 RÉSULTATS")
print("="*80)

valid = [r for r in results if 'error' not in r]
print(f"\n✓ Succès:      {success_count}/{len(PROMPTS)} ({100*success_count/len(PROMPTS):.0f}%)")
print(f"❌ Erreurs:    {len(PROMPTS)-success_count}/{len(PROMPTS)}")

if latencies:
    avg_lat = sum(latencies) / len(latencies)
    print(f"\n⏱️  Latence moyenne:  {avg_lat:.2f}s/prompt")
    print(f"   Min/Max:        {min(latencies):.2f}s / {max(latencies):.2f}s")

if token_counts:
    avg_tokens = sum(token_counts) / len(token_counts)
    print(f"\n📈 Tokens générés:   {avg_tokens:.1f} tokens/prompt (min: {min(token_counts)}, max: {max(token_counts)})")

print()
print("="*80)
print("✅ VERDICT")
print("="*80)

prod_ready = (success_count >= 40 and avg_tokens >= 8)
print(f"\n{'🚀 PRÊT POUR PRODUCTION' if prod_ready else '⚠️  À AMÉLIORER'}")
print(f"   Taux succès: {100*success_count/len(PROMPTS):.0f}% (cible: 80%+)")
print(f"   Qualité: {'✓' if avg_tokens >= 8 else '✗'}")

# Save results
output_file = Path('TEST_RESULTS_50_PROMPTS.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'model_step': ck_step,
        'total_tests': len(PROMPTS),
        'successful': success_count,
        'success_rate': f"{100*success_count/len(PROMPTS):.1f}%",
        'avg_latency': f"{avg_lat:.2f}s",
        'avg_tokens': f"{avg_tokens:.1f}",
        'results': results[:10]  # First 10 for preview
    }, f, indent=2, ensure_ascii=False)

print(f"\n✓ Résultats sauvegardés: {output_file}")
print("="*80)
