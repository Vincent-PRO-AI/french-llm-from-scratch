#!/usr/bin/env python3
"""
🤗 Direct push to HuggingFace using API
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, login, get_token
from huggingface_hub.utils import RepositoryNotFoundError

def get_hf_token():
    """Get HuggingFace token"""
    # Try to get from environment or config
    token = get_token()
    
    if not token:
        print("\n❌ No HuggingFace token found")
        print("\nOptions:")
        print("1. Set environment variable: export HF_TOKEN=your_token")
        print("2. Run: hf auth login")
        print("3. Or paste token here:")
        token = input("Enter your HuggingFace token: ").strip()
        
        if token:
            # Save it
            login(token=token)
            print("✅ Token saved!")
    
    return token

def main():
    print("\n" + "="*70)
    print("🚀 PUSHING TO HUGGINGFACE HUB")
    print("="*70 + "\n")
    
    # Get token
    token = get_hf_token()
    if not token:
        print("❌ No token provided")
        sys.exit(1)
    
    # Init API
    api = HfApi(token=token)
    
    # Get user info
    user_info = api.get_user_info()
    username = user_info.user_id
    print(f"👤 User: {username}\n")
    
    # Model info
    model_dir = "trained_models/outputs/french_gpt2_pytorch"
    repo_id = f"{username}/french-gpt2-conversations"
    
    if not Path(model_dir).exists():
        print(f"❌ Model not found: {model_dir}")
        sys.exit(1)
    
    print(f"📦 Model: {model_dir}")
    print(f"📍 Repository: {repo_id}\n")
    
    # Create model card
    model_card_path = Path(model_dir) / "README.md"
    if not model_card_path.exists():
        print("📝 Creating model card...")
        model_card = f"""---
language: fr
tags:
  - french
  - gpt2
  - conversation
  - language-model
license: mit
---

# French GPT-2 (Conversations)

🇫🇷 GPT-2 model trained exclusively on French conversation data.

## Model Details

- **Architecture**: GPT-2
- **Parameters**: 124M
- **Training Steps**: 80,000
- **Final Loss**: 3.10
- **Training Data**: 85.5M French conversation tokens
- **Context Length**: 1024 tokens
- **Vocabulary**: 50,257 tokens

## Training Configuration

- **Batch Size**: 4
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Precision**: Float32
- **Training Time**: 3.25 hours
- **Hardware**: RTX 5080 (16GB VRAM)

## Usage

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{repo_id}")
output = generator("Bonjour, ", max_length=100)
print(output[0]["generated_text"])
```

## Dataset

- **Source**: French conversations (100%)
- **Size**: 85.5M tokens
- **Quality**: Cleaned & deduplicated
- **Languages**: French only

## Performance Tips

- **Temperature**: 0.7 (creative) or 0.3 (factual)
- **Max Tokens**: 256-512 recommended
- **Top P**: 0.9
- **Repetition Penalty**: 1.1

## Limitations

- Conversational bias (trained on conversations)
- Not instruction-tuned
- French-specific training (lower English capability)
- No safety fine-tuning

## License

MIT

## Citation

```bibtex
@model{{french_gpt2_conversations_2025,
  author={{Vincent PRO AI}},
  title={{French GPT-2 Conversations}},
  year={{2025}}
}}
```

## Links

- GitHub: https://github.com/Vincent-PRO-AI/french-llm-from-scratch
- Project: French Language Model Training from Scratch
"""
        with open(model_card_path, 'w') as f:
            f.write(model_card)
        print("✅ Model card created\n")
    
    # Create or get repo
    print("🔧 Creating/accessing repository...")
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            exist_ok=True
        )
        print(f"✅ Repository: {repo_url}\n")
    except Exception as e:
        print(f"⚠️  {e}\n")
    
    # Upload files
    print("📤 Uploading model files...\n")
    
    files_to_upload = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "README.md"
    ]
    
    model_path = Path(model_dir)
    
    for file_name in files_to_upload:
        file_path = model_path / file_name
        if file_path.exists():
            print(f"  📤 Uploading {file_name}...", end=" ", flush=True)
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_name,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add {file_name}"
                )
                print("✅")
            except Exception as e:
                print(f"❌ {e}")
        else:
            print(f"  ⚠️  {file_name} not found")
    
    print("\n" + "="*70)
    print("✅ UPLOAD COMPLETE!")
    print("="*70 + "\n")
    print(f"🌐 Model available at:")
    print(f"   https://huggingface.co/{repo_id}\n")
    print("💡 Load with:")
    print(f"   from transformers import pipeline")
    print(f"   p = pipeline('text-generation', model='{repo_id}')\n")

if __name__ == "__main__":
    main()
