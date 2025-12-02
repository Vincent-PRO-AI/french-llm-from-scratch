#!/usr/bin/env python3
"""
Upload des fichiers GGUF vers Hugging Face Hub dans un repo dédié
"""
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Uploader des fichiers GGUF vers Hugging Face")
    parser.add_argument('--local-dir', type=str, default='trained_models/gguf', help='Dossier contenant les .gguf')
    parser.add_argument('--repo-id', type=str, default='vincent-pro-ai/french-llm-from-scratch-GGUF', help='Repo HF cible')
    parser.add_argument('--private', action='store_true', help='Créer le repo en privé')
    parser.add_argument('--token', type=str, help='Token HF (sinon utilise config locale)')
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder, whoami
    except Exception as e:
        print("❌ huggingface_hub non installé. Installez: pip install huggingface-hub")
        print(e)
        sys.exit(1)

    token = args.token
    if not token:
        try:
            info = whoami()
            print(f"👤 Connecté en tant que: {info.get('name') or info.get('id')}")
        except Exception:
            print("⚠️  Aucun token fourni. Assurez-vous d'être connecté:")
            print("   hf auth login")

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        print(f"❌ Dossier introuvable: {local_dir}")
        sys.exit(1)

    gguf_files = list(local_dir.glob('*.gguf'))
    if not gguf_files:
        print(f"❌ Aucun fichier GGUF trouvé dans {local_dir}")
        sys.exit(1)

    print("============================================================")
    print("🚀 UPLOAD GGUF VERS HUGGING FACE HUB")
    print("============================================================")
    print("Fichiers:")
    for f in gguf_files:
        print(f"   - {f.name} ({f.stat().st_size / (1024**2):.1f} MB)")

    # Créer le repo s'il n'existe pas
    try:
        create_repo(args.repo_id, exist_ok=True, private=args.private, token=token)
        print(f"✅ Repo prêt: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"❌ Erreur création repo: {e}")
        sys.exit(1)

    # Uploader le dossier complet
    try:
        upload_folder(
            repo_id=args.repo_id,
            folder_path=str(local_dir),
            commit_message="Add GGUF quantized models (Q4_K_M, Q5_K_M, Q8_0) + FP32",
            token=token,
        )
        print("✅ Upload terminé!")
        print(f"🔗 https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"❌ Erreur upload: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
