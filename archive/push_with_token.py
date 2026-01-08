#!/usr/bin/env python3
"""
🤗 Push to HuggingFace - Simple version
"""

import sys
import os
import subprocess

if len(sys.argv) < 2:
    print("\n❌ Usage: python3 push_with_token.py <token>")
    print("\nExample:")
    print("   python3 push_with_token.py hf_xxxxxxxxxxxxx\n")
    sys.exit(1)

token = sys.argv[1]

print("\n" + "="*70)
print("🚀 PUSHING TO HUGGINGFACE")
print("="*70 + "\n")

# Set token
os.environ["HF_TOKEN"] = token

# Verify token
print("🔐 Validating token...")
result = subprocess.run(["hf", "auth", "whoami"], capture_output=True, text=True)

if result.returncode != 0:
    print("❌ Invalid token!")
    print(result.stderr)
    print("\nGet a new token:")
    print("   1. Go to: https://huggingface.co/settings/tokens")
    print("   2. Create NEW token (Write access)")
    print("   3. Copy it")
    sys.exit(1)

print(f"✅ {result.stdout.strip()}\n")

# Upload
model_dir = "trained_models/outputs/french_gpt2_pytorch"

print(f"📦 Model: {model_dir}")
print(f"📤 Uploading...\n")

# Use huggingface-cli
cmd = [
    "huggingface-cli", "upload",
    "vincent-pro-ai/french-gpt2-conversations",
    f"{model_dir}/*",
    "--repo-type", "model",
    "--private"
]

result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n" + "="*70)
    print("✅ UPLOAD SUCCESSFUL!")
    print("="*70 + "\n")
    print("🌐 Model: https://huggingface.co/vincent-pro-ai/french-gpt2-conversations\n")
else:
    print("\n❌ Upload failed!")
    sys.exit(1)
