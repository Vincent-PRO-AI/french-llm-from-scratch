#!/usr/bin/env python3
"""
Optimized training launcher for RTX 5080 + 96 GB RAM
- Batch size: 24 (instead of 4)
- Gradient accumulation: 1 (larger batches possible)
- Model: large (24 layers, 20 heads, 1280 embed, 5120 FF)
- Training: 300k → 500k steps for better convergence
"""
import subprocess
import sys
import os
from pathlib import Path

def launch_optimized_training():
    print("🚀 LANCEMENT ENTRAÎNEMENT OPTIMISÉ RTX 5080 + 96 GB RAM")
    print("=" * 70)
    print("Configuration:")
    print("  GPU: NVIDIA RTX 5080 (16 GB VRAM)")
    print("  RAM: 96 GB DDR5 5600MT/s")
    print("  Batch size: 24 (vs 4 précédemment)")
    print("  Gradient accumulation: 1")
    print("  Model size: Large (520M paramètres)")
    print("  Steps: 300k → 500k (200k steps supplémentaires)")
    print("=" * 70)

    # Paths
    script_dir = Path(__file__).parent
    train_script = script_dir / "scripts" / "train_subtitles_transformer.py"
    checkpoint_path = Path("trained_models/runs/french_v3_finetune_resample_1/checkpoint_step_160000.pt")
    mistral_tokenizer = Path("data_clean/mistral_tokenizer")
    train_data = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")
    test_data = Path("data_clean/conversations_mega_test_mistral_tokenized.pt")

    # Verify required paths
    print("\n✓ Vérification des fichiers requis...")
    required = {
        "Script": train_script,
        "Tokenizer": mistral_tokenizer,
        "Train data": train_data,
        "Test data": test_data,
    }
    
    missing = []
    for name, path in required.items():
        if path.exists():
            if path.is_file() and path.stat().st_size > 0:
                size_gb = path.stat().st_size / (1024**3)
                print(f"  ✅ {name}: {path} ({size_gb:.2f} GB)")
            else:
                print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} - MANQUANT")
            missing.append(name)
    
    # Check checkpoint existence (optional, can start from scratch)
    if checkpoint_path.exists():
        print(f"  ✅ Checkpoint 160k: {checkpoint_path} ({checkpoint_path.stat().st_size / (1024**3):.2f} GB)")
        has_checkpoint = True
    else:
        print(f"  ⚠️  Checkpoint 160k: {checkpoint_path} - ABSENT")
        has_checkpoint = False
    
    if missing:
        print(f"\n❌ Fichiers manquants: {', '.join(missing)}")
        print("❌ Impossible de lancer l'entraînement")
        return False
    
    # Set environment variables for optimal GPU utilization
    print("\n⚙️  Configuration des variables d'environnement...")
    os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:2048"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_TF32"] = "1"  # TensorFloat32 pour meilleure perf
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "16"
    
    print("  ✅ PYTORCH_ALLOC_CONF=max_split_size_mb:2048")
    print("  ✅ TORCH_TF32=1 (TensorFloat32 enabled)")
    print("  ✅ CUDA_LAUNCH_BLOCKING=0")
    
    # Build training command
    cmd = [
        sys.executable,
        str(train_script),
        "--arch-preset", "large",  # NEW: Large model
        "--run-name", "french_large_rtx5080_500k",
        "--batch-size", "24",  # 6x larger than previous
        "--lr", "1.5e-4",  # Slightly lower LR for stability
        "--max-steps", "500000",  # 300k → 500k
        "--eval-interval", "250",
        "--sample-interval", "500",
        "--checkpoint-interval", "5000",  # Checkpoint every 5k steps
        "--metrics-log-fraction", "0.02",
        "--sample-prompt", "Utilisateur: Comment peux-tu m'aider ? Assistant:",
        "--sample-max-new-tokens", "200",
        "--sample-temperature", "0.9",
        "--sample-top-k", "50",
        "--pretokenized-path", str(train_data),
        "--tokenizer-path", str(mistral_tokenizer),
        "--use-amp",  # Mixed precision training
        "--gradient-accumulation-steps", "1",  # Direct large batches
        "--num-workers", "8",
        "--pin-memory",
        "--prefetch-factor", "4",
        "--persistent-workers",
        "--auto-resource-adapt",
    ]
    
    # Build training command
    cmd = [
        sys.executable,
        str(train_script),
        "--arch-preset", "medium",  # 18 layers - architecture compatible
        "--run-name", "french_medium_rtx5080_extended",
        "--batch-size", "16",  # Augmenté de 4 à 16 (4x plus)
        "--lr", "1.5e-4",  # Learning rate optimisé
        "--max-steps", "500000",  # 160k → 500k (340k steps supplémentaires)
        "--eval-interval", "250",
        "--sample-interval", "500",
        "--checkpoint-interval", "5000",
        "--metrics-log-fraction", "0.02",
        "--sample-prompt", "Utilisateur: Comment peux-tu m'aider ? Assistant:",
        "--sample-max-new-tokens", "200",
        "--sample-temperature", "0.9",
        "--sample-top-k", "50",
        "--pretokenized-path", str(train_data),
        "--tokenizer-path", str(mistral_tokenizer),
        "--gradient-accumulation-steps", "1",
        "--num-workers", "8",
        "--prefetch-factor", "4",
        "--auto-resource-adapt",
    ]
    
    # Add checkpoint resumption if available
    if has_checkpoint:
        cmd.extend(["--resume-from", str(checkpoint_path)])
        print(f"\n▶️  Reprenant depuis: {checkpoint_path}")
    else:
        print(f"\n▶️  Démarrage de zéro")
    
    print(f"\n📊 Paramètres d'entraînement:")
    print(f"  • Architecture: MEDIUM (18 layers, 16 heads, 1024 embed, 4096 FF)")
    print(f"  • Batch size: 16 (4x plus grand que avant)")
    print(f"  • Learning rate: 1.5e-4")
    print(f"  • Max steps: 500,000 (160k → 500k = +340k steps)")
    print(f"  • Eval interval: 250 steps")
    print(f"  • Checkpoint interval: 5,000 steps")
    print(f"  • Mixed precision: Enabled (AMP)")
    print(f"  • Tokenizer: Mistral-7B (32k vocab)")
    print(f"  • Durée estimée: ~15-20h RTX 5080 (340k steps supplémentaires)")
    
    print(f"\n📊 Paramètres d'entraînement:")
    print(f"  • Architecture: LARGE (24 layers, 20 heads, 1280 embed, 5120 FF)")
    print(f"  • Batch size: 24")
    print(f"  • Learning rate: 1.5e-4")
    print(f"  • Max steps: 500,000")
    print(f"  • Eval interval: 250 steps")
    print(f"  • Checkpoint interval: 5,000 steps")
    print(f"  • Mixed precision: Enabled (AMP)")
    print(f"  • Tokenizer: Mistral-7B (32k vocab)")
    print(f"  • Durée estimée: ~30h RTX 5080 (vs ~100h RTX 3090)")
    
    print(f"\n🚀 Lancement de l'entraînement...")
    print("-" * 70)
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("✅ Entraînement terminé avec succès!")
            print("=" * 70)
            return True
        else:
            print("\n" + "=" * 70)
            print(f"❌ Erreur lors de l'entraînement (code: {result.returncode})")
            print("=" * 70)
            return False
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu par l'utilisateur")
        print("💾 Le dernier checkpoint a été sauvegardé")
        return False
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    success = launch_optimized_training()
    sys.exit(0 if success else 1)
