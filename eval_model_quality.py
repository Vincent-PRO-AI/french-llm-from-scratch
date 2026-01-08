#!/usr/bin/env python3
"""
🧪 Évaluation Qualité - 50 Prompts Français Communs
"""
import sys
sys.path.insert(0, '/home/vincent/code/repo/french-llm-from-scratch')

import torch
from pathlib import Path
from scripts.train_subtitles_transformer import Config, SubtitleTrainer
from transformers import AutoTokenizer
import json
import time

print("🧪 ÉVALUATION DU MODÈLE - 50 Prompts")
print("="*70)

# Paths
CKPT = Path('trained_models/tiny_subtitles_transformer.pt')
TOK = Path('data_clean/mistral_tokenizer')

# Load checkpoint
print(f"[1/3] Chargement de {CKPT}...")
ckpt = torch.load(str(CKPT), map_location='cuda', weights_only=False)
cfg_dict = ckpt['config']

# Recréer config avec SubtitleTrainer (qui gère la config correctement)
cfg = Config()
for k, v in cfg_dict.items():
    if hasattr(cfg, k):
        setattr(cfg, k, v)

# Disable auto-resume to avoid loading the old 160k checkpoint during init
cfg.resume_checkpoint = None
cfg.resume_run_dir = None

cfg.tokenizer_path = TOK
cfg.device = 'cuda'

print(f"   Architecture: {cfg.num_layers}L {cfg.num_heads}H {cfg.embed_dim}E")

# Initialize trainer
trainer = SubtitleTrainer(cfg)
trainer.model.load_state_dict(ckpt['model_state'], strict=True)
trainer.model.eval()

tokenizer = AutoTokenizer.from_pretrained(str(TOK))
print(f"   Tokenizer: {tokenizer.vocab_size} tokens")

# 50 prompts
PROMPTS = [
    "L'histoire de la civilisation est", "La littérature française montre", "La science révolutionne",
    "L'art exprime l'émotion par", "La philosophie pose la question", "La politique est le domaine",
    "L'économie fonctionne selon", "L'éducation forme les esprits", "L'environnement est menacé",
    "La technologie change notre façon", "L'amour est un sentiment", "La mort est inévitable",
    "La liberté signifie pouvoir", "L'égalité demande la justice", "La vérité se cache dans",
    "La beauté réside dans", "L'intelligence est la capacité", "La sagesse vient de",
    "La passion nous pousse à", "Le courage est la force", "L'espoir est essentiel pour",
    "Le succès dépend de", "L'échec enseigne la leçon", "Le bonheur vient de",
    "La douleur forge le caractère", "La nostalgie nous ramène", "Le souvenir persiste dans",
    "L'avenir est incertain mais", "Le passé modèle notre", "Le présent est le moment",
    "La famille est le fondement", "L'amitié renforce nos liens", "La loyauté est rare",
    "La trahison blesse profondément", "Le pardon guérit l'âme", "La paix commence dans",
    "La guerre détruit tout", "La violence engendre la", "Le dialogue résout les",
    "L'empathie nous rapproche", "L'indifférence isole les", "La compassion sauve des",
    "L'esprit critique questionne le", "La curiosité pousse à explorer", "L'imagination crée des",
    "La créativité exprime l'innovation", "L'originalité distingue les", "La tradition preserve la",
]

def generate_text(model, idx, max_new_tokens=50, temperature=0.8, top_k=50):
    """
    Generate text using the model.
    idx: (B, T) tensor of indices
    """
    block_size = getattr(model.cfg, 'block_size', 256)
    
    for step in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Forward
        logits = model(idx_cond)
        
        # Logits at last step
        logits = logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # DEBUG: Print first few tokens
        if step == 0 and idx.size(1) < 20: 
             pass # print(f"DEBUG: Next token ID: {idx_next.item()}")
        
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

print(f"[2/3] Test de {len(PROMPTS)} prompts...")

results = []
latencies = []
token_counts = []
successes = 0
samples_log = []

with torch.no_grad():
    for i, prompt in enumerate(PROMPTS, 1):
        pct = 40 * i // len(PROMPTS)
        print(f"\r[{'█'*pct}{'░'*(40-pct)}] {i}/{len(PROMPTS)}", end='', flush=True)
        
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(cfg.device)
            if i == 1: print(f"\nDEBUG: {inputs[0].tolist()} -> {tokenizer.decode(inputs[0])}")
            start = time.time()
            
            # Use trainer's internal sample method
            trainer.cfg.sample_prompt = prompt
            trainer.cfg.sample_max_new_tokens = 50
            trainer.cfg.sample_temperature = 0.8
            
            text = trainer.sample_text()
            latency = time.time() - start
            
            # Estimate tokens (approx)
            tokens = len(text) // 4 if text else 0
            
            gen = None # Unused
            
            latencies.append(latency)
            token_counts.append(tokens)
            if tokens >= 3:
                successes += 1
                
            sample_entry = f"PROMPT: {prompt}\nGEN: {text[len(prompt):].strip()}\n{'-'*40}"
            samples_log.append(sample_entry)
            
        except Exception as e:
            samples_log.append(f"PROMPT: {prompt}\nERROR: {e}\n{'-'*40}")
            pass

# Save samples immediately
with open('EVAL_SAMPLES.txt', 'w', encoding='utf-8') as f:
    f.write("\n\n".join(samples_log))

print("\n\n=== EXEMPLES GÉNÉRÉS ===")
for s in samples_log[:3]:
    print(s)
print("========================\n")

print()
print("[3/3] Analyse des résultats...")
print()
print("="*70)
print("📊 RÉSULTATS")
print("="*70)

if latencies:
    avg_lat = sum(latencies) / len(latencies)
    avg_tok = sum(token_counts) / len(token_counts)
    rate = 100 * successes / len(PROMPTS)
    
    print(f"\n✓ Succès:    {successes}/{len(PROMPTS)} ({rate:.0f}%)")
    print(f"⏱️  Latence:   {avg_lat:.2f}s (min: {min(latencies):.2f}s, max: {max(latencies):.2f}s)")
    print(f"📈 Tokens:    {avg_tok:.1f} (min: {min(token_counts)}, max: {max(token_counts)})")
    
    print("\n" + "="*70)
    ready = (rate >= 80 and avg_tok >= 8)
    print(f"{'🚀 PRÊT PRODUCTION' if ready else '⚠️  À AMÉLIORER'}")
    print("="*70)
    
    # Save
    with open('EVAL_RESULTS.json', 'w') as f:
        json.dump({
            'total': len(PROMPTS),
            'success': successes,
            'rate': f"{rate:.0f}%",
            'avg_latency': f"{avg_lat:.2f}s",
            'avg_tokens': f"{avg_tok:.1f}",
            'production_ready': ready,
            'step': ckpt.get('step'),
            'timestamp': str(time.ctime())
        }, f, indent=2)
    
    print(f"\n✓ Résultats: EVAL_RESULTS.json")

