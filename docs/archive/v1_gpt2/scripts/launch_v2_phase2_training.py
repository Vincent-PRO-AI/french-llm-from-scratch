#!/usr/bin/env python3
"""
🎯 LANCEMENT TRAINING V2 - 100% FRANÇAIS
Utilise checkpoint_step_100000.pt + conversations_mega_train_mistral
Phase 2A: 100k → 110k steps (LR 1e-6, ultra-conservateur)
"""
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

def main():
    print("\n" + "="*70)
    print("🚀 FRENCH LLM V2 - PHASE 2A TRAINING")
    print("="*70)
    print(f"📅 Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    venv_python = Path(".venv/bin/python3")
    checkpoint = Path("trained_models/runs/checkpoint_step_100000.pt")
    dataset = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")
    tokenizer = Path("trained_models/tokenizers/vincent_tokenizer_FR_v2")
    
    # Vérifications
    print("📋 Vérifications pré-training:")
    checks = {
        "Python venv": venv_python.exists(),
        "Checkpoint 100k": checkpoint.exists(),
        "Dataset mega_train": dataset.exists(),
        "Tokenizer V2": tokenizer.exists()
    }
    
    for name, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
        if not status:
            print(f"\n❌ ERREUR: {name} manquant!")
            if name == "Checkpoint 100k":
                print(f"   Attendu: {checkpoint}")
            elif name == "Dataset mega_train":
                print(f"   Attendu: {dataset}")
                print(f"   Disponible: {list(Path('data_clean').glob('*.pt'))[:5]}")
            return 1
    
    print()
    print("⚙️  Configuration Training Phase 2A:")
    config = {
        "Architecture": "Medium (260M params)",
        "Checkpoint départ": "100,000 steps",
        "Checkpoint cible": "110,000 steps",
        "Dataset": f"conversations_mega_train_mistral ({dataset.stat().st_size / 1024**2:.0f}MB)",
        "Learning Rate": "1e-6 (ultra-conservateur)",
        "Batch Size": "10",
        "Gradient Accumulation": "4",
        "Mixed Precision": "AMP FP16",
        "Durée estimée": "2-3h GPU (RTX 5080)"
    }
    for key, val in config.items():
        print(f"  • {key}: {val}")
    
    print()

    if not tokenizer_path.exists():
        print(f"❌ Tokenizer non trouvé: {tokenizer_path}")
        return

    if not partitioned_dir.exists():
        print(f"❌ Données partitionnées non trouvées: {partitioned_dir}")
        return

    print("✅ Fichiers vérifiés:")
    print(f"   - Checkpoint: {checkpoint_path}")
    print(f"   - Tokenizer: {tokenizer_path}")
    print(f"   - Données: {partitioned_dir}")

    # Charger les métadonnées des partitions
    metadata_path = partitioned_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        total_tokens = metadata.get('total_tokens', 0)
        num_partitions = metadata.get('num_partitions', 0)
        print(f"   - Total tokens: {total_tokens:,}")
        print(f"   - Partitions: {num_partitions}")
    else:
        print("⚠️  Métadonnées non trouvées, utilisation par défaut")

    # Créer un script temporaire pour charger les partitions
    loader_script = Path("scripts/load_v2_partitions.py")
    if not loader_script.exists():
        print("\n🔄 Création du script de chargement des partitions...")
        loader_content = '''#!/usr/bin/env python3
"""
Script pour charger et combiner les partitions V2.
"""
import torch
import json
from pathlib import Path

def load_v2_partitions():
    partitioned_dir = Path("data_clean/v2_tokenized/phase2_partitioned")
    metadata_path = partitioned_dir / "metadata.json"

    if not metadata_path.exists():
        print("❌ metadata.json non trouvé")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    partitions = metadata['partitions']
    all_tokens = []

    print(f"🔄 Chargement de {len(partitions)} partitions...")

    for partition in partitions:
        path = Path(partition['path'])
        if path.exists():
            print(f"   Chargement {path.name}...")
            tokens = torch.load(path, map_location='cpu')
            all_tokens.append(tokens)
        else:
            print(f"⚠️  Partition manquante: {path}")

    if all_tokens:
        combined = torch.cat(all_tokens)
        output_path = Path("data_clean/v2_tokenized/combined_phase2.pt")
        torch.save(combined, output_path)
        print(f"✅ Données combinées sauvegardées: {output_path}")
        print(f"   Taille: {len(combined):,} tokens")
        return output_path
    else:
        print("❌ Aucune partition chargée")
        return None

if __name__ == "__main__":
    load_v2_partitions()
'''
        loader_script.parent.mkdir(exist_ok=True)
        with open(loader_script, 'w') as f:
            f.write(loader_content)
        loader_script.chmod(0o755)

    # Combiner les partitions
    if not run_command([str(venv_python), "scripts/load_v2_partitions.py"],
                      "Étape 1: Combinaison des partitions V2"):
        return

    # Vérifier que le fichier combiné existe
    combined_path = Path("data_clean/v2_tokenized/combined_phase2.pt")
    if not combined_path.exists():
        print(f"❌ Fichier combiné non créé: {combined_path}")
        return

    print("\n🔄 Étape 2: Lancement entraînement V2 Phase 2")
    print("   Configuration:")
    print("   - Tokenizer: Vincent SentencePiece 32k")
    print("   - Données: combined_phase2.pt (~103M tokens)")
    print("   - Run: french_v2_phase2")
    print("   - Checkpoint initial: checkpoint_step_17500.pt")
    print("   - Steps: 17500 → 25000 (7500 steps)")
    print("   - Batch size: 8")
    print("   - LR: 1e-6 (conservateur)")
    print("   - Optimisations: torch.compile + FusedAdam + AMP")

    cmd = [
        str(venv_python), "scripts/train_subtitles_transformer.py",
        "--arch-preset", "medium",
        "--pretokenized-path", str(combined_path),
        "--tokenizer-path", str(tokenizer_path / "tokenizer.model"),
        "--run-name", "french_v2_phase2",
        "--resume-from", str(checkpoint_path),
        "--max-steps", "25000",
        "--batch-size", "8",
        "--checkpoint-interval", "500",
        "--sample-interval", "200",
        "--eval-interval", "250",
        "--metrics-log-fraction", "0.05",
        "--lr", "1e-6",
        "--weight-decay", "0.01",
        "--vocab-size", "32000"
    ]

    print(f"   Commande: {' '.join(cmd)}")
    print("\n⏳ Lancement de l'entraînement... (ça va prendre ~2-3h)")
    print("   Tu peux suivre la progression avec:")
    print("   tail -f trained_models/runs/french_v2_phase2/metrics.jsonl")
    print("   Ou utiliser le dashboard: docker-compose up -d && open http://localhost:5174")

    try:
        # Lancer en arrière-plan pour ne pas bloquer
        process = subprocess.Popen(cmd, cwd=Path(__file__).parent)
        print(f"   PID du processus: {process.pid}")
        print("   ✅ Entraînement lancé en arrière-plan!")
        print("   Tu peux fermer ce terminal, l'entraînement continue.")
        print("\n📊 Suivi:")
        print("   - Logs: trained_models/runs/french_v2_phase2/")
        print("   - Métriques: tail -f metrics.jsonl")
        print("   - Samples: tail -f samples.txt")
        print("   - Checkpoints: ls -la checkpoint_step_*.pt")

    except Exception as e:
        print(f"   ❌ Impossible de lancer l'entraînement: {e}")

if __name__ == "__main__":
    main()