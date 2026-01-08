#!/usr/bin/env python3
"""
🎯 FRENCH LLM V2 - PHASE 2 COMPLÈTE (2A + 2B)
==============================================
Phase 2A: 100k → 110k steps (LR 1e-6, adaptation conversationnel)
Phase 2B: 110k → 130k steps (LR 5e-7, fine-tuning profond)
Durée totale: 6-10h GPU (RTX 5080 16GB)
Dashboard: http://localhost:5174
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time

def main():
    print("\n" + "="*70)
    print("🚀 FRENCH LLM V2 - PHASE 2 COMPLÈTE (2A + 2B)")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    venv_python = Path(".venv/bin/python3")
    checkpoint_start = Path("trained_models/runs/checkpoint_step_100000.pt")
    dataset = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")
    tokenizer = Path("trained_models/tokenizers/vincent_tokenizer_FR_v2")
    
    # Vérifications
    print("📋 Vérifications:")
    checks = [
        ("Python venv", venv_python),
        ("Checkpoint 100k", checkpoint_start),
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
        print("\n❌ Fichiers manquants!")
        return 1
    
    dataset_size_mb = dataset.stat().st_size / 1024**2
    
    print(f"\n⚙️  Configuration Phase 2 Complète:")
    print(f"  🎯 Phase 2A: 100k → 110k steps (LR 1e-6, 2-3h)")
    print(f"  🎯 Phase 2B: 110k → 130k steps (LR 5e-7, 4-6h)")
    print(f"  📦 Dataset: conversations_mega_train_mistral ({dataset_size_mb:.0f}MB)")
    print(f"  🏗️  Architecture: Medium (260M params)")
    print(f"  ⚡ Optimizations: AMP FP16, torch.compile, Flash Attention")
    print(f"  ⏱️  Durée totale estimée: 6-10h GPU")
    print(f"  📊 Dashboard: http://localhost:5174")
    
    print(f"\n✅ Lancement automatique du training Phase 2 complète (2A + 2B)")
    print(f"⏱️  Durée: 6-10h GPU")
    
    print("\n" + "="*70)
    print("🚀 DÉMARRAGE PHASE 2A (100k → 110k)")
    print("="*70)
    
    # PHASE 2A
    run_name_2a = "french_v2_phase2a_mega"
    cmd_2a = [
        str(venv_python),
        "scripts/train_subtitles_transformer.py",
        "--arch-preset", "medium",
        "--resume-from", str(checkpoint_start),
        "--pretokenized-path", str(dataset),
        "--tokenizer-path", str(tokenizer),
        "--run-name", run_name_2a,
        "--max-steps", "110000",
        "--lr", "1e-6",
        "--batch-size", "10",
        "--checkpoint-interval", "1000",
        "--eval-interval", "500",
        "--sample-interval", "500",
        "--device", "cuda",
        "--metrics-log-fraction", "0.05"
    ]
    
    print(f"📁 Run: {run_name_2a}")
    print(f"📊 Metrics: trained_models/runs/{run_name_2a}/metrics.jsonl")
    print(f"📝 Samples: trained_models/runs/{run_name_2a}/samples.txt")
    print(f"💡 Dashboard: http://localhost:5174")
    print()
    
    start_time = time.time()
    
    try:
        print("⏳ Phase 2A en cours... (2-3h)\n")
        result_2a = subprocess.run(cmd_2a, cwd=Path(__file__).parent)
        
        if result_2a.returncode != 0:
            print(f"\n❌ Phase 2A échouée avec code {result_2a.returncode}")
            return result_2a.returncode
        
        elapsed_2a = (time.time() - start_time) / 3600
        print("\n" + "="*70)
        print(f"✅ PHASE 2A TERMINÉE! ({elapsed_2a:.1f}h)")
        print("="*70)
        
        checkpoint_2a = Path(f"trained_models/runs/{run_name_2a}/checkpoint_step_110000.pt")
        if not checkpoint_2a.exists():
            print(f"❌ Checkpoint 110k non trouvé: {checkpoint_2a}")
            return 1
        
        print(f"✅ Checkpoint: {checkpoint_2a}")
        
        # Pause avant Phase 2B
        print(f"\n⏸️  Pause de 5 secondes avant Phase 2B...")
        time.sleep(5)
        
        print("\n" + "="*70)
        print("🚀 DÉMARRAGE PHASE 2B (110k → 130k)")
        print("="*70)
        
        # PHASE 2B
        run_name_2b = "french_v2_phase2b_mega"
        cmd_2b = [
            str(venv_python),
            "scripts/train_subtitles_transformer.py",
            "--arch-preset", "medium",
            "--resume-from", str(checkpoint_2a),
            "--pretokenized-path", str(dataset),
            "--tokenizer-path", str(tokenizer),
            "--run-name", run_name_2b,
            "--max-steps", "130000",
            "--lr", "5e-7",  # LR encore plus conservateur
            "--batch-size", "10",
            "--checkpoint-interval", "2500",
            "--eval-interval", "1000",
            "--sample-interval", "1000",
            "--device", "cuda",
            "--metrics-log-fraction", "0.05"
        ]
        
        print(f"📁 Run: {run_name_2b}")
        print(f"📊 Metrics: trained_models/runs/{run_name_2b}/metrics.jsonl")
        print(f"🎯 LR réduit: 5e-7 (ultra-conservateur)")
        print()
        
        print("⏳ Phase 2B en cours... (4-6h)\n")
        result_2b = subprocess.run(cmd_2b, cwd=Path(__file__).parent)
        
        if result_2b.returncode != 0:
            print(f"\n❌ Phase 2B échouée avec code {result_2b.returncode}")
            return result_2b.returncode
        
        total_time = (time.time() - start_time) / 3600
        
        print("\n" + "="*70)
        print("✅✅✅ PHASE 2 COMPLÈTE TERMINÉE! ✅✅✅")
        print("="*70)
        print(f"\n⏱️  Durée totale: {total_time:.1f}h")
        print(f"\n📦 Checkpoints créés:")
        print(f"  • Phase 2A: trained_models/runs/{run_name_2a}/checkpoint_step_110000.pt")
        print(f"  • Phase 2B: trained_models/runs/{run_name_2b}/checkpoint_step_130000.pt")
        
        print(f"\n🎯 Prochaines étapes:")
        print(f"  1. Tester génération:")
        print(f"     python3 scripts/test_model_samples.py \\")
        print(f"       --checkpoint trained_models/runs/{run_name_2b}/checkpoint_step_130000.pt \\")
        print(f"       --tokenizer-path {tokenizer}")
        print(f"\n  2. Export HuggingFace:")
        print(f"     python3 scripts/export_to_huggingface.py \\")
        print(f"       --checkpoint trained_models/runs/{run_name_2b}/checkpoint_step_130000.pt \\")
        print(f"       --output-dir trained_models/french_llm_v2_hf")
        print(f"\n  3. Conversion GGUF:")
        print(f"     python3 scripts/convert_to_gguf.py \\")
        print(f"       --model-dir trained_models/french_llm_v2_hf \\")
        print(f"       --output-dir trained_models/french_llm_v2_gguf")
        print(f"\n🚀 Prêt pour publication! 🇫🇷")
        
        return 0
        
    except KeyboardInterrupt:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n\n⚠️  Training interrompu après {elapsed:.1f}h (Ctrl+C)")
        print(f"   Les checkpoints sont sauvegardés:")
        print(f"   - trained_models/runs/{run_name_2a}/")
        if 'run_name_2b' in locals():
            print(f"   - trained_models/runs/{run_name_2b}/")
        return 130
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
