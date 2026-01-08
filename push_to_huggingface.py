#!/usr/bin/env python3
"""
🤗 Push French GPT-2 to HuggingFace Hub
"""

import os
import sys
import subprocess
from pathlib import Path

def check_auth():
    """Check HuggingFace authentication"""
    result = subprocess.run(
        ["hf", "auth", "whoami"],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def login():
    """Prompt user to login"""
    print("\n" + "="*70)
    print("🔐 HUGGINGFACE AUTHENTICATION REQUIRED")
    print("="*70 + "\n")
    
    print("1️⃣  Get your token from:")
    print("   https://huggingface.co/settings/tokens\n")
    
    print("2️⃣  Create a NEW token with:")
    print("   • Name: French GPT-2")
    print("   • Role: Write\n")
    
    input("Press ENTER when you have your token ready...")
    
    print("\n3️⃣  Login with:")
    subprocess.run(["hf", "auth", "login"])
    
    print("\n✅ Authentication complete!\n")

def push_model():
    """Push model to HuggingFace"""
    print("\n" + "="*70)
    print("🚀 PUSHING MODEL TO HUGGINGFACE")
    print("="*70 + "\n")
    
    # Get username
    result = subprocess.run(
        ["hf", "auth", "whoami"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Authentication failed")
        return False
    
    username = result.stdout.strip()
    print(f"👤 Username: {username}\n")
    
    # Model path
    model_dir = "trained_models/outputs/french_gpt2_pytorch"
    repo_name = "french-gpt2-conversations"
    full_repo = f"{username}/{repo_name}"
    
    if not Path(model_dir).exists():
        print(f"❌ Model not found: {model_dir}")
        return False
    
    print(f"📦 Model: {model_dir}")
    print(f"📍 Repository: https://huggingface.co/{full_repo}\n")
    
    # Create model card
    model_card_path = Path(model_dir) / "README.md"
    if not model_card_path.exists():
        print("📝 Creating model card...")
        model_card = """---
language: fr
tags:
  - french
  - gpt2
  - conversation
  - language-model
license: mit
---

# French GPT-2 (Conversations)

🇫🇷 Trained on 85.5M French conversation tokens

## Model
- Parameters: 124M
- Training: 80,000 steps
- Loss: 3.10
- Context: 1024 tokens

## Usage

```python
from transformers import pipeline
p = pipeline("text-generation", model="{full_repo}")
print(p("Bonjour, "))
```

## Data
- 100% French conversations
- Cleaned & deduplicated

## License
MIT
"""
        with open(model_card_path, 'w') as f:
            f.write(model_card)
        print("✅ Model card created\n")
    
    # Upload
    print("⏳ Uploading to HuggingFace...\n")
    
    cmd = [
        "huggingface-cli", "upload",
        full_repo,
        f"{model_dir}/*",
        "--repo-type", "model",
        "--private",
        "--commit-message", "Add French GPT-2 (124M, 80k steps, loss 3.10)"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print("✅ UPLOAD SUCCESSFUL!")
        print("="*70 + "\n")
        print(f"🌐 Model: https://huggingface.co/{full_repo}\n")
        print("💡 To use:")
        print(f"   from transformers import pipeline")
        print(f"   p = pipeline('text-generation', model='{full_repo}')\n")
        return True
    else:
        print("\n❌ Upload failed!")
        return False

def main():
    print("\n" + "="*70)
    print("🤗 FRENCH GPT-2 → HUGGINGFACE HUB")
    print("="*70)
    
    # Check auth
    if not check_auth():
        print("\n⚠️  Not authenticated")
        login()
    
    # Push
    if check_auth():
        success = push_model()
        sys.exit(0 if success else 1)
    else:
        print("\n❌ Authentication failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
