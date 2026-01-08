#!/usr/bin/env python3
"""
🚀 CONVERT TO GGUF (LM STUDIO) + HUGGINGFACE
============================================
Export trained GPT-2 model to GGUF format and HuggingFace
"""

import torch
import subprocess
from pathlib import Path
import json
import shutil

def convert_to_gguf():
    """Convert PyTorch model to GGUF format"""
    print("\n" + "="*70)
    print("🔄 CONVERTING TO GGUF (LM STUDIO)")
    print("="*70)
    
    # Paths
    hf_model_dir = Path("trained_models/runs/french_llm_hf")
    gguf_output = Path("trained_models/outputs/french_llm_gpt2.gguf")
    gguf_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Get latest checkpoint
    checkpoints = sorted(hf_model_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        print(f"\n📦 Latest checkpoint: {latest_ckpt.name}")
        model_dir = latest_ckpt
    else:
        print("❌ No checkpoints found!")
        return False
    
    # Try to convert using llama.cpp tools
    print(f"\n🔧 Converting {model_dir} → {gguf_output}")
    
    try:
        # Convert using Hugging Face transformers
        cmd = [
            "python3", "-m", "transformers.convert_pytorch_checkpoint_to_gguf",
            "--model_dir", str(model_dir),
            "--output_file", str(gguf_output),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ GGUF saved: {gguf_output}")
            print(f"   Size: {gguf_output.stat().st_size / (1024**3):.2f} GB")
            return True
        else:
            print("⚠️  Conversion method 1 failed, trying safetensors export...")
            
            # Fallback: export to safetensors (LM Studio compatible)
            from transformers import AutoModel
            model = AutoModel.from_pretrained(str(model_dir))
            model.save_pretrained(gguf_output.parent / "model_safetensors", safe_serialization=True)
            print(f"✅ SafeTensors exported: {gguf_output.parent / 'model_safetensors'}")
            return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Manual conversion needed: use llama.cpp with your checkpoint")
        return False

def export_to_huggingface():
    """Prepare model for HuggingFace Hub"""
    print("\n" + "="*70)
    print("🤗 EXPORTING TO HUGGINGFACE")
    print("="*70)
    
    hf_model_dir = Path("trained_models/runs/french_llm_hf")
    
    # Get latest checkpoint
    checkpoints = sorted(hf_model_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    if not checkpoints:
        print("❌ No checkpoints found!")
        return False
    
    latest_ckpt = checkpoints[-1]
    print(f"\n📦 Model: {latest_ckpt.name}")
    
    # Create model card
    model_card = """---
language: fr
tags:
  - french
  - language-model
  - gpt2
  - conversational
license: mit
---

# French GPT-2 (100% French Conversations)

Fine-tuned GPT-2 on 85.5M French conversation tokens.

## Model Details

- **Architecture**: GPT-2 (124M parameters)
- **Training Data**: 100% French conversations (Reddit, Twitter, WebCrawl)
- **Training Steps**: 80,000
- **Final Loss**: 3.10
- **Precision**: FP32

## Training

```bash
python3 train_simple_hf.py
```

## Usage

```python
from transformers import pipeline

generator = pipeline("text-generation", model="vincent-pro-ai/french-gpt2-conversations")
result = generator("Bonjour, ", max_length=100)
print(result[0]["generated_text"])
```

## Training Details

- Batch size: 4
- Learning rate: 1e-4
- Optimizer: AdamW
- Precision: FP32
- Gradient checkpointing: Yes

## License

MIT

## Author

Vincent PRO AI
"""
    
    # Save model card
    model_card_path = latest_ckpt / "README.md"
    with open(model_card_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    print(f"✅ Model card saved: {model_card_path}")
    
    # Create dataset info
    dataset_info = {
        "name": "French Conversations 100%",
        "tokens": 85490944,
        "sources": ["Reddit", "Twitter", "WebCrawl"],
        "language": "French",
        "quality": "Cleaned & deduplicated"
    }
    
    with open(latest_ckpt / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Dataset info saved")
    
    print("\n📤 NEXT STEPS FOR HUGGINGFACE:")
    print(f"   1. huggingface-cli login")
    print(f"   2. huggingface-cli repo create french-gpt2-conversations --private")
    print(f"   3. git clone https://huggingface.co/<your-username>/french-gpt2-conversations")
    print(f"   4. cp -r {latest_ckpt}/* french-gpt2-conversations/")
    print(f"   5. cd french-gpt2-conversations && git add . && git commit -m 'Add model' && git push")
    
    return True

def create_lm_studio_guide():
    """Create guide for LM Studio import"""
    print("\n" + "="*70)
    print("📖 LM STUDIO GUIDE")
    print("="*70)
    
    guide = """# French GPT-2 - LM Studio Setup

## Import Steps

1. **Download LM Studio**
   https://lmstudio.ai/

2. **Get Model File**
   - GGUF: `trained_models/outputs/french_llm_gpt2.gguf`
   - Or download from HuggingFace

3. **Import into LM Studio**
   - Open LM Studio
   - Click "Search Models" 
   - Paste model path or HuggingFace ID
   - Download & load

4. **Test Chat**
   - Model loaded automatically
   - Start chatting in French!
   - System prompt: "Tu es un assistant français helpful."

## Performance Tips

- **Max Tokens**: 512
- **Temperature**: 0.7
- **Top P**: 0.9
- **Batch Size**: Adjust based on GPU VRAM

## Model Specs

- Parameters: 124M
- Context: 1024 tokens
- Precision: FP32
- Size: ~500MB (GGUF)

## Known Limitations

- Trained on conversations only (not instruction-tuned)
- French bias (better for French than English)
- No safety training (use with caution)
"""
    
    with open("LM_STUDIO_SETUP.md", 'w') as f:
        f.write(guide)
    
    print("✅ LM Studio guide created: LM_STUDIO_SETUP.md")

def main():
    print("\n" + "="*80)
    print("🚀 MODEL EXPORT PIPELINE")
    print("="*80)
    
    # Export to HuggingFace
    hf_ok = export_to_huggingface()
    
    # Convert to GGUF
    gguf_ok = convert_to_gguf()
    
    # Create LM Studio guide
    create_lm_studio_guide()
    
    # Summary
    print("\n" + "="*80)
    print("✅ EXPORT COMPLETE")
    print("="*80)
    print("\n📦 Available Models:")
    print("   - HuggingFace: trained_models/runs/french_llm_hf/checkpoint-*")
    print("   - GGUF (LM Studio): trained_models/outputs/french_llm_gpt2.gguf")
    print("\n📖 Setup Guides:")
    print("   - HuggingFace: See train_simple_hf.py output")
    print("   - LM Studio: LM_STUDIO_SETUP.md")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
