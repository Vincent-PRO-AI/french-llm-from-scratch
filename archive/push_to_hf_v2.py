#!/usr/bin/env python3
"""
🤗 Upload model to new HuggingFace repository (vincent-pro-ai/french-tiny-conversations-300k)
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

# Configuration
REPO_ID = "vincent-pro-ai/french-tiny-conversations-300k"
MODEL_PATH = Path("trained_models/tiny_subtitles_transformer.pt")
TOKENIZER_DIR = Path("data_clean/mistral_tokenizer")
EVAL_RESULTS = Path("EVAL_RESULTS.json")
SAMPLES = Path("EVAL_SAMPLES.txt")

def main():
    print(f"🚀 Uploading to {REPO_ID}...")
    
    api = HfApi()
    
    # 1. Create Repo (if not exists)
    try:
        create_repo(repo_id=REPO_ID, exist_ok=True, private=False)
        print("✓ Repo ready.")
    except Exception as e:
        print(f"⚠️  Repo creation warning: {e}")

    # 2. Upload Model
    if MODEL_PATH.exists():
        print(f"📤 Uploading model {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e9:.2f} GB)...")
        try:
            api.upload_file(
                path_or_fileobj=MODEL_PATH,
                path_in_repo="pytorch_model.pt",
                repo_id=REPO_ID,
                repo_type="model"
            )
            print("✓ Model uploaded.")
        except Exception as e:
            print(f"❌ Model upload failed: {e}")
            sys.exit(1)
    else:
        print(f"❌ Model file not found: {MODEL_PATH}")
        sys.exit(1)

    # 3. Upload Tokenizer
    print("📤 Uploading tokenizer...")
    api.upload_folder(
        folder_path=TOKENIZER_DIR,
        repo_id=REPO_ID,
        repo_type="model"
    )
    print("✓ Tokenizer uploaded.")

    # 4. Upload Eval Results
    if EVAL_RESULTS.exists():
        api.upload_file(
            path_or_fileobj=EVAL_RESULTS,
            path_in_repo="eval_results.json",
            repo_id=REPO_ID
        )
        print("✓ Eval results uploaded.")

    if SAMPLES.exists():
        api.upload_file(
            path_or_fileobj=SAMPLES,
            path_in_repo="eval_samples.txt",
            repo_id=REPO_ID
        )
        print("✓ Samples uploaded.")
        
    print("\n✅ Upload Complete! -> https://huggingface.co/" + REPO_ID)

if __name__ == "__main__":
    main()
