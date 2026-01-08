#!/bin/bash
# Simple push script - just needs HF_TOKEN environment variable

if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "❌ Environment variable HF_TOKEN not set"
    echo ""
    echo "Set it with:"
    echo "   export HF_TOKEN=hf_xxxxx"
    echo ""
    echo "Then run:"
    echo "   bash push_hf_simple.sh"
    echo ""
    exit 1
fi

echo ""
echo "=================================="
echo "🚀 PUSHING MODEL TO HUGGINGFACE"
echo "=================================="
echo ""

cd /home/vincent/code/repo/french-llm-from-scratch

python3 << 'PYTHON_EOF'
import os
from pathlib import Path
from huggingface_hub import HfApi, login

# Login
token = os.environ.get("HF_TOKEN")
if not token:
    print("❌ HF_TOKEN not found")
    exit(1)

api = HfApi(token=token)

# Get user
user_info = api.get_user_info()
username = user_info.user_id
print(f"👤 User: {username}\n")

repo_id = f"{username}/french-gpt2-conversations"
model_dir = "trained_models/outputs/french_gpt2_pytorch"

print(f"📍 Repo: {repo_id}")
print(f"📦 Model: {model_dir}\n")

# Create repo
print("🔧 Creating repository...")
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=True,
    exist_ok=True
)
print("✅ Repository ready\n")

# Upload
print("📤 Uploading files...")
api.upload_folder(
    folder_path=model_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Add French GPT-2 (124M, 80k steps, loss 3.10)"
)

print("\n✅ DONE!")
print(f"\n🌐 https://huggingface.co/{repo_id}")

PYTHON_EOF
