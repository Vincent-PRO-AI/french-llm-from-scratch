#!/usr/bin/env python3
"""
🧪 Comprehensive Model Testing Suite - 50 Common AI Prompts
Test du modèle French LLM sur 50 prompts variés d'IA commun
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
║                    🧪 COMPREHENSIVE MODEL TESTING                        ║
║                   50 Common AI Prompts Evaluation                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

# Configuration
CHECKPOINT_PATH = Path('trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint.pt')
TOKENIZER_PATH = Path('data_clean/mistral_tokenizer')

# Load checkpoint
print("[1/4] Chargement du checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda', weights_only=False)
ck_cfg = checkpoint.get('config', {})
ck_step = checkpoint.get('step', 'unknown')

# Initialize model
print("[2/4] Initialisation du modèle...")
cfg = Config()
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.tokenizer_path = TOKENIZER_PATH

if isinstance(ck_cfg, dict):
    cfg.embed_dim = ck_cfg.get('embed_dim', cfg.embed_dim)
    cfg.num_layers = ck_cfg.get('num_layers', cfg.num_layers)
    cfg.num_heads = ck_cfg.get('num_heads', cfg.num_heads)
    cfg.ff_hidden_dim = ck_cfg.get('ff_hidden_dim', cfg.ff_hidden_dim)
    cfg.vocab_size = ck_cfg.get('vocab_size', cfg.vocab_size)
    cfg.block_size = ck_cfg.get('block_size', cfg.block_size)

trainer = SubtitleTrainer(cfg)
model = trainer.model.to(cfg.device).eval()
model_state = checkpoint.get('model_state') or checkpoint.get('model')
model.load_state_dict(model_state, strict=False)

# Load tokenizer
print("[3/4] Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))

# 50 Common AI Prompts (French)
PROMPTS = [
    # Littérature et histoire (5)
    "L'histoire de la civilisation française",
    "Les grands auteurs de la littérature française",
    "La Révolution française a changé",
    "Napoléon Bonaparte était",
    "Le Moyen Âge était une période",
    
    # Science et technologie (5)
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
    
    # Société et politique (5)
    "La démocratie est un système",
    "Les droits humains sont",
    "L'égalité dans la société signifie",
    "La justice sociale demande",
    "La gouvernance efficace nécessite",
    
    # Économie et affaires (5)
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
    
    # Arts et culture (5)
    "L'art exprime",
    "La musique a le pouvoir de",
    "La peinture Renaissance était",
    "Le cinéma moderne montre",
    "La danse traduit",
    
    # Psychologie et santé mentale (5)
    "La psychologie étudie",
    "Le stress mental peut être réduit par",
    "L'empathie est la capacité de",
    "La résilience signifie",
    "La dépression affecte",
    
    # Technologie et avenir (5)
    "Les réseaux sociaux ont transformé",
    "La cybersécurité protège",
    "La blockchain révolutionne",
    "L'Internet des objets connecte",
    "La réalité virtuelle permet",
]

print(f"[4/4] Test de {len(PROMPTS)} prompts...")
print()

# Testing loop
results = []
timestamps = []
latencies = []
quality_scores = []

with torch.no_grad():
    for idx, prompt in enumerate(PROMPTS, 1):
        print(f"[{idx:2d}/50] {prompt[:50]}...", end=" ", flush=True)
        
        try:
            # Tokenize
            inputs = tokenizer.encode(prompt, return_tensors='pt').to(cfg.device)
            input_len = len(inputs[0])
            
            # Generate
            start_time = time.time()
            generated = model.generate(
                input_ids=inputs,
                max_new_tokens=60,
                temperature=0.75,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id or 2
            )
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Decode
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            generated_len = len(generated[0]) - input_len
            
            # Simple quality heuristic (word count, diversity, etc.)
            words = output_text.split()
            unique_words = len(set(words))
            avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
            
            # Score (0-100): consider length, uniqueness, diversity
            quality_score = min(100, (generated_len / 60) * 40 + (unique_words / len(words) * 100 if words else 0) * 0.5)
            quality_scores.append(quality_score)
            
            result = {
                'idx': idx,
                'prompt': prompt,
                'output': output_text[:200],  # First 200 chars
                'tokens_generated': generated_len,
                'latency_sec': round(latency, 2),
                'quality_score': round(quality_score, 1),
                'unique_ratio': round(unique_words / len(words) if words else 0, 2),
                'status': '✓ OK' if generated_len > 10 else '⚠ SHORT'
            }
            results.append(result)
            
            print(f"{result['status']:8s} ({generated_len:2d} tokens, {latency:.2f}s, score: {quality_score:.0f}%)")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)[:40]}")
            result = {
                'idx': idx,
                'prompt': prompt,
                'output': '',
                'error': str(e),
                'status': '❌ FAILED'
            }
            results.append(result)

# Statistics
print()
print("="*80)
print("📊 STATISTIQUES")
print("="*80)

valid_results = [r for r in results if 'error' not in r]
total_tests = len(PROMPTS)
successful = len(valid_results)
failed = total_tests - successful

print(f"\n✓ Tests réussis:      {successful}/{total_tests} ({100*successful/total_tests:.1f}%)")
print(f"❌ Tests échoués:      {failed}/{total_tests}")

if latencies:
    print(f"\n⏱️  Latence moyenne:    {sum(latencies)/len(latencies):.2f}s")
    print(f"   Min/Max:           {min(latencies):.2f}s / {max(latencies):.2f}s")

if quality_scores:
    avg_score = sum(quality_scores) / len(quality_scores)
    print(f"\n🎯 Score de qualité moyen: {avg_score:.1f}%")
    print(f"   Min/Max:                {min(quality_scores):.1f}% / {max(quality_scores):.1f}%")

# Quality distribution
if quality_scores:
    excellent = len([s for s in quality_scores if s >= 80])
    good = len([s for s in quality_scores if 60 <= s < 80])
    fair = len([s for s in quality_scores if 40 <= s < 60])
    poor = len([s for s in quality_scores if s < 40])
    
    print(f"\n📈 Distribution de qualité:")
    print(f"   Excellent (80+):    {excellent} ({100*excellent/len(quality_scores):.1f}%)")
    print(f"   Bon (60-80):        {good} ({100*good/len(quality_scores):.1f}%)")
    print(f"   Acceptable (40-60): {fair} ({100*fair/len(quality_scores):.1f}%)")
    print(f"   Faible (-40):       {poor} ({100*poor/len(quality_scores):.1f}%)")

# Save detailed results
print()
print("="*80)
print("💾 SAUVEGARDE DES RÉSULTATS")
print("="*80)

output_file = Path('TEST_RESULTS_50_PROMPTS.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'model_step': ck_step,
        'checkpoint': str(CHECKPOINT_PATH),
        'device': cfg.device,
        'total_tests': total_tests,
        'successful': successful,
        'failed': failed,
        'success_rate': round(100*successful/total_tests, 1),
        'avg_latency': round(sum(latencies)/len(latencies), 2) if latencies else None,
        'avg_quality_score': round(avg_score, 1) if quality_scores else None,
        'results': results
    }, f, indent=2, ensure_ascii=False)

print(f"✓ Résultats sauvegardés: {output_file}")

# Create evaluation summary
print()
print("="*80)
print("✅ ÉVALUATION FINALE")
print("="*80)

production_ready = (successful >= 45 and avg_score >= 65)

print(f"\n🎯 Prêt pour la production: {'✅ OUI' if production_ready else '⚠️  À AMÉLIORER'}")
print(f"\nCritères:")
print(f"  • Taux de succès ≥ 90%:    {'✓' if successful >= 45 else '✗'} ({100*successful/total_tests:.1f}%)")
print(f"  • Score moyen ≥ 65%:       {'✓' if avg_score >= 65 else '✗'} ({avg_score:.1f}%)")
print(f"  • Latence acceptable:      {'✓' if sum(latencies)/len(latencies) < 15 else '✗'} ({sum(latencies)/len(latencies):.2f}s)")

print("\n" + "="*80)
if production_ready:
    print("🚀 LE MODÈLE EST PRÊT POUR LE COMMIT EN PRODUCTION!")
else:
    print("⚠️  Recommandation: Vérifier les cas d'erreur avant production")
print("="*80)
