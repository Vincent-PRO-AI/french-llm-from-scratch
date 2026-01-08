#!/usr/bin/env python3
"""
🎯 FRENCH LLM V2 - PHASE 2A TRAINING LAUNCHER
================================================
Dataset: conversations_mega_train_mistral (100% français, 653MB, ~169M tokens)
Checkpoint: checkpoint_step_100000.pt (base pré-entraînée)
Objectif: 100k → 110k steps (adaptation format conversationnel)
Durée: 2-3h GPU (RTX 5080 16GB)
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    print("\n" + "="*70)
    print("🚀 FRENCH LLM V2 - PHASE 2A TRAINING")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    venv_python = Path(".venv/bin/python3")
    checkpoint = Path("trained_models/runs/checkpoint_step_100000.pt")
    dataset = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")
    tokenizer = Path("trained_models/tokenizers/vincent_tokenizer_FR_v2")
    run_name = "french_v2_phase2a_mega"
    
    # Vérifications
    print("📋 Vérifications:")
    checks = [
        ("Python venv", venv_python),
        ("Checkpoint 100k", checkpoint),
        ("Dataset mega_train", dataset),
        ("Tokenizer V2", tokenizer)
    ]
    
    all_ok = True
    for name, path in checks:
        exists = path.exists()
        icon = "✅" if exists else "❌"
        print(f"  {icon} {name}: {path}")
        if not exists:
            all_ok = False
    
    if not all_ok:
        print("\n❌ Fichiers manquants! Vérifiez les chemins ci-dessus.")
        return 1
    
    # Configuration training
    dataset_size_mb = dataset.stat().st_size / 1024**2
    print(f"\n⚙️  Configuration Phase 2A:")
    print(f"  • Architecture: Medium (260M params)")
    print(f"  • Checkpoint: step_100000 → step_110000")
    print(f"  • Dataset: conversations_mega_train_mistral ({dataset_size_mb:.0f}MB)")
    print(f"  • Learning Rate: 1e-6 (ultra-conservateur)")
    print(f"  • Batch Size: 10 (effective 40 avec accum)")
    print(f"  • Optimizations: AMP FP16, torch.compile, Flash Attention")
    print(f"  • Durée estimée: 2-3h GPU")
    print(f"  • Outputs: trained_models/runs/{run_name}/")
    
    # Confirmation
    print(f"\n🤔 Lancer l'entraînement Phase 2A maintenant?")
    print(f"   Cela va occuper le GPU pendant ~2-3h")
    response = input("   Taper 'oui' pour confirmer: ").strip().lower()
    
    if response not in ['oui', 'o', 'yes', 'y']:
        print("❌ Annulé par l'utilisateur")
        return 0
    
    # Construction commande
    cmd = [
        str(venv_python),
        "scripts/train_subtitles_transformer.py",
        "--arch-preset", "medium",
        "--resume-from", str(checkpoint),
        "--pretokenized-path", str(dataset),
        "--tokenizer-path", str(tokenizer),
        "--run-name", run_name,
        "--max-steps", "110000",
        "--lr", "1e-6",
        "--batch-size", "10",
        "--gradient-accumulation-steps", "4",
        "--checkpoint-interval", "1000",
        "--eval-interval", "500",
        "--sample-interval", "500",
        "--device", "cuda",
        "--use-amp",
        "--metrics-log-fraction", "0.05"
    ]
    
    print(f"\n🚀 Lancement training...\n")
    print(f"📁 Run name: {run_name}")
    print(f"📊 Checkpoints: trained_models/runs/{run_name}/checkpoint_step_*.pt")
    print(f"📈 Metrics: trained_models/runs/{run_name}/metrics.jsonl")
    print(f"📝 Samples: trained_models/runs/{run_name}/samples.txt")
    print()
    print("💡 Monitoring en temps réel:")
    print(f"   tail -f trained_models/runs/{run_name}/metrics.jsonl")
    print(f"   watch -n 1 'tail -5 trained_models/runs/{run_name}/samples.txt'")
    print()
    print("⏳ Training démarré... Durée estimée: 2-3h\n")
    print("="*70)
    print()
    
    # Lancer training (pas en arrière-plan pour voir les logs)
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("\n" + "="*70)
            print("✅ PHASE 2A TERMINÉE AVEC SUCCÈS!")
            print("="*70)
            print(f"\n📦 Checkpoint final: trained_models/runs/{run_name}/checkpoint_step_110000.pt")
            print(f"\n🎯 Prochaine étape: Phase 2B (110k → 130k steps)")
            print(f"   Commande:")
            print(f"   python3 launch_phase2b_training.py")
            return 0
        else:
            print(f"\n❌ Training échoué avec code {result.returncode}")
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrompu par l'utilisateur (Ctrl+C)")
        print(f"   Le dernier checkpoint est sauvegardé dans:")
        print(f"   trained_models/runs/{run_name}/")
        return 130
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
