#!/usr/bin/env python3
"""
Combine TOUS les datasets tokenisés pour créer le dataset final massif
"""
import torch
from pathlib import Path

DATA_DIR = Path("data_clean")

# Tous les datasets disponibles
DATASETS = [
    # Datasets existants (15M tokens)
    "conversations_all_tokenized.pt",
    "conversations_extra_tokenized.pt",
    
    # Nouveaux datasets massifs (~100M tokens)
    "dolly_fr_tokenized.pt",
    "oasst2_fr_tokenized.pt",
    "ultrachat_fr_tokenized.pt",
    "ultrachat_extended_fr_tokenized.pt",
    "sharegpt_fr_tokenized.pt",
    "openchat_fr_tokenized.pt",
    "vigogne_instruct_fr_tokenized.pt",
]

OUTPUT_FILE = DATA_DIR / "conversations_mega_tokenized.pt"

def main():
    print("=" * 80)
    print("🔗 COMBINAISON DE TOUS LES DATASETS TOKENISÉS")
    print("=" * 80)
    
    all_tokens = []
    total_tokens = 0
    datasets_loaded = 0
    
    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / dataset_name
        
        if not dataset_path.exists():
            print(f"⚠️  Ignoré (non trouvé): {dataset_name}")
            continue
        
        print(f"\n📥 Chargement: {dataset_name}")
        try:
            tokens = torch.load(dataset_path)
            num_tokens = len(tokens)
            all_tokens.append(tokens)
            total_tokens += num_tokens
            datasets_loaded += 1
            
            print(f"   ✅ {num_tokens:,} tokens ({num_tokens/1e6:.1f}M)")
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    if not all_tokens:
        print("\n❌ Aucun dataset chargé!")
        return
    
    print("\n" + "=" * 80)
    print(f"📊 Datasets chargés: {datasets_loaded}/{len(DATASETS)}")
    print(f"📈 Total: {total_tokens:,} tokens ({total_tokens/1e6:.1f}M)")
    print("=" * 80)
    
    # Concaténer tous les tenseurs
    print("\n🔗 Concaténation des tenseurs...")
    combined = torch.cat(all_tokens, dim=0)
    
    # Sauvegarder
    print(f"💾 Sauvegarde: {OUTPUT_FILE}")
    torch.save(combined, OUTPUT_FILE)
    
    # Vérifier
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    
    print("\n" + "=" * 80)
    print("✅ COMBINAISON TERMINÉE")
    print("=" * 80)
    print(f"📂 Fichier: {OUTPUT_FILE}")
    print(f"📊 Tokens: {len(combined):,} ({len(combined)/1e6:.1f}M)")
    print(f"💾 Taille: {file_size_mb:.1f} MB")
    print(f"🎯 Ratio tokens/params: {len(combined)/260e6:.3f} ({len(combined)/260e6 * 100:.1f}% du modèle)")
    print("\n💡 Prochaine étape:")
    print("   Fine-tuner avec ce dataset massif:")
    print("   python scripts/train_subtitles_transformer.py \\")
    print("     --arch-preset medium \\")
    print("     --resume-from trained_models/runs/french_medium_finetune_conversations/checkpoint_step_115000.pt \\")
    print("     --pretokenized-path data_clean/conversations_mega_tokenized.pt \\")
    print("     --max-steps 180000 \\")
    print("     --lr 3e-6 \\")
    print("     --batch-size 4 \\")
    print("     --checkpoint-interval 2500 \\")
    print("     --run-name french_medium_mega_finetune")
    print("=" * 80)

if __name__ == "__main__":
    main()
