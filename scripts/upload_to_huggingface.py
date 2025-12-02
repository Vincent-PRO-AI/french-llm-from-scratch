#!/usr/bin/env python3
"""
Upload le modèle vers Hugging Face Hub
Requiert: huggingface-cli login (ou HF_TOKEN env var)
"""
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path
import argparse
import os


def upload_model(
    local_dir: str,
    repo_id: str,
    private: bool = False,
    token: str = None
):
    """
    Upload le modèle vers Hugging Face Hub
    
    Args:
        local_dir: Dossier local contenant model.safetensors, config.json, etc.
        repo_id: ID du repo HF (format: username/repo-name)
        private: Si True, repo privé
        token: Token HF (optionnel si déjà connecté)
    """
    local_dir = Path(local_dir)
    
    print("=" * 60)
    print("🚀 UPLOAD VERS HUGGING FACE HUB")
    print("=" * 60)
    
    # Vérifier que les fichiers requis existent
    required_files = [
        "model.safetensors",
        "config.json",
        "README.md",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    
    print("\n📋 Vérification des fichiers:")
    for file in required_files:
        file_path = local_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024**2)
            print(f"   ✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {file} MANQUANT")
            if file in ["model.safetensors", "config.json"]:
                raise FileNotFoundError(f"Fichier requis manquant: {file}")
    
    # Initialiser API
    api = HfApi(token=token)
    
    # Créer le repo si nécessaire
    print(f"\n📦 Création/vérification du repo: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✅ Repo créé/existant: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Erreur création repo: {e}")
        raise
    
    # Upload les fichiers
    print(f"\n⬆️  Upload des fichiers...")
    try:
        upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Initial commit: French LLM From Scratch - 260M with Fineweb-32k tokenizer",
            ignore_patterns=["*.pyc", "__pycache__", ".git*"],
        )
        print(f"✅ Upload terminé!")
    except Exception as e:
        print(f"❌ Erreur upload: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("✅ MODÈLE PUBLIÉ SUR HUGGING FACE")
    print("=" * 60)
    print(f"\n🌐 Lien: https://huggingface.co/{repo_id}")
    print(f'\n📥 Pour utiliser le modèle:')
    print(f'```python')
    print(f'from transformers import AutoModelForCausalLM, AutoTokenizer')
    print(f'')
    print(f'model = AutoModelForCausalLM.from_pretrained("{repo_id}")')
    print(f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")')
    print(f'```')
    print(f'\n📦 GitHub: https://github.com/Vincent-PRO-AI/french-llm-from-scratch')


def main():
    parser = argparse.ArgumentParser(description="Upload modèle vers Hugging Face Hub")
    parser.add_argument(
        '--local-dir',
        type=str,
        default='trained_models/huggingface/french-llm-from-scratch',
        help='Dossier local contenant le modèle'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='ID du repo HF (format: username/repo-name)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Créer un repo privé'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Token HF (optionnel si déjà connecté via huggingface-cli login)'
    )
    
    args = parser.parse_args()
    
    # Vérifier token
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("⚠️  Aucun token fourni. Assurez-vous d'être connecté:")
        print("   huggingface-cli login")
        print("   ou définissez HF_TOKEN env var")
    
    upload_model(args.local_dir, args.repo_id, args.private, token)


if __name__ == "__main__":
    main()
