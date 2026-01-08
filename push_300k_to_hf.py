#!/usr/bin/env python3
"""
🤗 Push verified 300k model to Hugging Face
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

# Configuration
REPO_ID = "vincent-pro-ai/french-llm-from-scratch"
MODEL_PATH = Path("/home/vincent/code/repo/french-llm-from-scratch/trained_models/tiny_subtitles_transformer.pt")
TOKENIZER_PATH = Path("/home/vincent/code/repo/french-llm-from-scratch/data_clean/mistral_tokenizer")

print(f"🚀 Pushing model to {REPO_ID}...")

# 1. Create Repo if needed
try:
    url = create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
    print(f"✅ Repository ready: {url}")
except Exception as e:
    print(f"⚠️  Repo check failed (might exist): {e}")

api = HfApi()

# 2. Upload Model Checkpoint
print(f"📤 Uploading model ({MODEL_PATH.stat().st_size / 1e9:.2f} GB)...")
try:
    api.upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo="tiny_subtitles_transformer.pt",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload 300k steps checkpoint (best model)"
    )
    print("✅ Model uploaded successfully")
except Exception as e:
    print(f"❌ Model upload failed: {e}")

# 3. Upload Tokenizer
print("📤 Uploading tokenizer...")
try:
    api.upload_folder(
        folder_path=str(TOKENIZER_PATH),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload tokenizer artifacts"
    )
    print("✅ Tokenizer uploaded successfully")
except Exception as e:
    print(f"❌ Tokenizer upload failed: {e}")

print("\n🎉 DONE! Model available at:")
print(f"https://huggingface.co/{REPO_ID}")
