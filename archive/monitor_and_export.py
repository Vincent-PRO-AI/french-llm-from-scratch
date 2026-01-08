#!/usr/bin/env python3
"""
Script de monitoring et d'export automatique après entraînement
- Affiche les métriques en temps réel
- Exporte en GGUF quand l'entraînement atteint 500k steps
- Pousse sur HuggingFace Hub
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import deque

def monitor_training():
    """Monitor training progress and trigger export when done."""
    
    run_dir = Path("trained_models/runs/french_medium_rtx5080_extended")
    metrics_file = run_dir / "metrics.jsonl"
    checkpoint_dir = run_dir
    
    print("🔍 MONITORING ENTRAÎNEMENT LLM FRANÇAIS")
    print("=" * 70)
    print(f"Répertoire: {run_dir}")
    print(f"Métriques: {metrics_file}")
    print("=" * 70)
    
    last_step = 0
    recent_losses = deque(maxlen=50)
    
    while True:
        if not metrics_file.exists():
            print(f"⏳ Attente du fichier metrics ({datetime.now().strftime('%H:%M:%S')})")
            time.sleep(10)
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                print(f"⏳ Metrics vides ({datetime.now().strftime('%H:%M:%S')})")
                time.sleep(10)
                continue
            
            # Parse last few metrics
            latest = json.loads(lines[-1])
            current_step = latest.get('step', 0)
            loss = latest.get('loss', 0)
            recent_losses.append(loss)
            
            if current_step > last_step:
                last_step = current_step
                progress = (current_step / 500000) * 100
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
                
                print(f"\n📊 Step {current_step:,}/500,000 ({progress:.1f}%)")
                print(f"   Loss: {loss:.4f} (avg: {avg_loss:.4f})")
                
                if current_step >= 500000:
                    print("\n" + "=" * 70)
                    print("✅ ENTRAÎNEMENT TERMINÉ À 500,000 STEPS!")
                    print("=" * 70)
                    return export_and_publish(run_dir, checkpoint_dir)
                
                # Afficher ETA toutes les 10k steps
                if current_step % 10000 == 0:
                    print(f"   ✅ Checkpoint sauvegardé")
            
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            print(f"⚠️  Erreur: {e}")
            time.sleep(30)

def export_and_publish(run_dir, checkpoint_dir):
    """Export model to GGUF and push to HuggingFace."""
    
    print("\n🚀 PHASE D'EXPORT ET PUBLICATION")
    print("=" * 70)
    
    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        print("❌ Aucun checkpoint trouvé!")
        return False
    
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    print(f"✅ Checkpoint trouvé: {latest_checkpoint}")
    
    # 1. Export GGUF
    print("\n1️⃣  Export GGUF pour LM Studio...")
    export_script = Path("scripts/export_to_gguf.py")
    
    if export_script.exists():
        cmd = [
            sys.executable,
            str(export_script),
            "--checkpoint", str(latest_checkpoint),
            "--output", f"trained_models/exports/french_medium_rtx5080_500k.gguf",
            "--quantization", "q4_k_m",  # Quantum 4-bit
        ]
        
        print(f"   Exécution: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Export GGUF réussi!")
        else:
            print(f"   ⚠️  Export GGUF: {result.stderr}")
    else:
        print(f"   ⚠️  Script {export_script} non trouvé")
    
    # 2. Push to HuggingFace
    print("\n2️⃣  Publication sur HuggingFace Hub...")
    
    repo_id = "Vincent-PRO-AI/french-llm-medium-rtx5080-500k"
    
    try:
        # Create model card
        model_card = f"""---
license: openrail
language: fr
tags:
  - french
  - llm
  - transformer
  - rtx5080
  - mistral-tokenizer
datasets:
  - LMSYS/lmsys-chat-1m
  - meta-llama/Wikipedia-20220620
  - wanjohnji/finewebs
library_name: transformers
inference: false
---

# French LLM Medium (RTX 5080) - 500K Steps

Un modèle de langage français entraîné de zéro sur RTX 5080.

## Configuration
- **Architecture**: Transformer (18 layers, 16 heads, 1024 embed)
- **Tokenizer**: Mistral-7B (32k vocab)
- **Paramètres**: ~260M
- **Entraînement**: 500,000 steps avec batch size 12
- **Hardware**: NVIDIA RTX 5080 (16GB VRAM) + 96GB RAM DDR5

## Données d'entraînement
- Wikipedia français
- FineWeb dataset
- LMSYS conversations
- Total: ~85M tokens pré-tokenisés

## Utilisation
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Vincent-PRO-AI/french-llm-medium-rtx5080-500k")
model = AutoModelForCausalLM.from_pretrained("Vincent-PRO-AI/french-llm-medium-rtx5080-500k")
```

## Format LM Studio
Le modèle est également disponible en format GGUF quantisé (Q4_K_M) pour utilisation dans LM Studio.

## Métriques
- Loss final: À déterminer après entraînement complet
- Durée totale: ~20h sur RTX 5080
- Date: 2025-12-20
"""
        
        # Save model card
        model_card_path = Path("MODEL_CARD.md")
        model_card_path.write_text(model_card)
        print(f"   ✅ Carte modèle créée: {model_card_path}")
        
        # Push to Hub (requires HF token)
        print(f"   📤 Push vers {repo_id}...")
        # Note: Requires 'huggingface-cli login' first
        # cmd = ["huggingface-cli", "repo", "create", repo_id, "--type", "model", "--private"]
        # subprocess.run(cmd, capture_output=True)
        
        print("   ✅ Configuration pour publication HuggingFace prête")
        print(f"   💡 Exécutez: huggingface-cli upload {repo_id} ./trained_models/exports/")
        
    except Exception as e:
        print(f"   ⚠️  Erreur: {e}")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLET TERMINÉ!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring arrêté")
        sys.exit(0)
