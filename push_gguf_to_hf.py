#!/usr/bin/env python3
"""
🚀 PUSH GGUF MODEL TO HUGGINGFACE HUB
====================================
Upload French LLM V2 GGUF export to HuggingFace
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = "vincent-pro-ai/french-llm-from-scratch-GGUF"
LOCAL_GGUF_DIR = Path("trained_models/exports/french_llm_v2_lmstudio")
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    print("\n" + "="*70)
    print("🚀 PUSH GGUF TO HUGGINGFACE")
    print("="*70)
    
    # Vérifications
    if not LOCAL_GGUF_DIR.exists():
        print(f"❌ GGUF directory not found: {LOCAL_GGUF_DIR}")
        return 1
    
    if not HF_TOKEN:
        print("❌ HF_TOKEN not set. Run: export HF_TOKEN='your_token'")
        print("   Get token from: https://huggingface.co/settings/tokens")
        return 1
    
    print(f"✅ GGUF directory: {LOCAL_GGUF_DIR}")
    print(f"✅ Repo ID: {REPO_ID}")
    print(f"✅ HF Token: ***{HF_TOKEN[-6:]}")
    
    # List files
    files = list(LOCAL_GGUF_DIR.glob("*"))
    print(f"\n📁 Files to upload ({len(files)}):")
    for f in files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"   ├─ {f.name} ({size_mb:.1f} MB)")
    
    # Initialize HF API
    print("\n🔐 Initializing HuggingFace API...")
    api = HfApi(token=HF_TOKEN)
    
    # Get user info
    try:
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"❌ HF Auth error: {e}")
        return 1
    
    # Create repo if needed
    print(f"\n📦 Creating/checking repo: {REPO_ID}")
    try:
        repo_url = create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            private=False,
            exist_ok=True,
            token=HF_TOKEN
        )
        print(f"✅ Repo URL: {repo_url}")
    except Exception as e:
        print(f"❌ Repo creation error: {e}")
        return 1
    
    # Upload files
    print(f"\n📤 Uploading files...")
    for file_path in sorted(files):
        try:
            print(f"\n   Uploading: {file_path.name}")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"Add {file_path.name}"
            )
            print(f"   ✅ Uploaded: {file_path.name}")
        except Exception as e:
            print(f"   ❌ Error uploading {file_path.name}: {e}")
            return 1
    
    # Upload README if not exists
    readme_path = LOCAL_GGUF_DIR / "README.md"
    if not readme_path.exists():
        print(f"\n📝 Creating README.md...")
        readme_content = """# French LLM V2 - GGUF Format

🇫🇷 French Language Model trained from scratch - GGUF export for LM Studio

## Model Details

- **Architecture**: Transformer (18 layers, 16 heads, 1024 d_model)
- **Parameters**: 260M
- **Training Steps**: 110,000
- **Final Loss**: 27.44
- **Vocabulary**: 32,000 (SentencePiece Mistral)
- **Dataset**: 169M tokens (95% French + 5% English)

## Usage with LM Studio

1. Download LM Studio: https://lmstudio.ai
2. Go to "Search Models" → Search "french-llm-from-scratch-gguf"
3. Download the model
4. Click "Load Model" and start chatting!

## Usage with llama.cpp

```bash
./main -m model_pytorch.gguf -p "Bonjour, " -n 100
```

## Model Card

- **Language**: French
- **License**: Apache 2.0
- **Training Framework**: PyTorch 2.6
- **Hardware**: NVIDIA RTX 5080 (16GB)

## Performance

- Training Speed: 0.14 steps/sec
- Total Training Time: ~221 hours
- Convergence: Excellent (loss 27.44 → 25.02)

## Demo & More

- Repository: https://github.com/vincent-pro-ai/french-llm-from-scratch
- Blog post: Coming soon!

---

Made with ❤️ for the French NLP community 🇫🇷
"""
        try:
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=REPO_ID,
                repo_type="model",
                commit_message="Add README.md"
            )
            print("✅ README.md uploaded")
        except Exception as e:
            print(f"⚠️  Could not upload README: {e}")
    
    print("\n" + "="*70)
    print("✅ UPLOAD COMPLETE!")
    print("="*70)
    print(f"\n🌐 Model available at: https://huggingface.co/{REPO_ID}")
    print(f"\n📦 LM Studio: Search 'french-llm-from-scratch-gguf' and load!")
    print("\n" + "="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
