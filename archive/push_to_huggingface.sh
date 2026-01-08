#!/bin/bash
# 🚀 Push model to HuggingFace Hub

set -e

echo ""
echo "=================================="
echo "📤 PUSHING TO HUGGINGFACE HUB"
echo "=================================="
echo ""

# Check if logged in
echo "🔐 Checking HuggingFace authentication..."
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "❌ Not logged in. Run: huggingface-cli login"
    exit 1
fi

echo "✅ Authenticated"
echo ""

# Model paths
MODEL_DIR="trained_models/outputs/french_gpt2_pytorch"
HF_REPO="vincent-pro-ai/french-gpt2-conversations"

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model not found: $MODEL_DIR"
    echo "   Run: python3 export_model.py"
    exit 1
fi

echo "📦 Model: $MODEL_DIR"
echo "📍 Repository: $HF_REPO"
echo ""

# Add model card if not exists
if [ ! -f "$MODEL_DIR/README.md" ]; then
    echo "📝 Creating model card..."
    cat > "$MODEL_DIR/README.md" << 'EOF'
---
language: fr
tags:
  - french
  - gpt2
  - conversation
  - language-model
license: mit
---

# French GPT-2 (Conversations)

Trained GPT-2 model (124M parameters) on 85.5M French conversation tokens.

## Usage

```python
from transformers import pipeline

generator = pipeline("text-generation", model="vincent-pro-ai/french-gpt2-conversations")
output = generator("Bonjour, ", max_length=100)
print(output[0]["generated_text"])
```

## Model Details
- Parameters: 124M
- Training: 80,000 steps
- Loss: 3.10
- Language: French
- Data: 100% conversations
EOF
    echo "✅ Model card created"
fi

echo ""
echo "📤 Uploading to HuggingFace..."
echo ""

huggingface-cli upload "$HF_REPO" "$MODEL_DIR"/* --repo-type model

echo ""
echo "✅ Upload complete!"
echo "🌐 Model available at: https://huggingface.co/$HF_REPO"
echo ""
