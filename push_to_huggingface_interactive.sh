#!/bin/bash
# Push French GPT-2 model to HuggingFace Hub

echo ""
echo "=================================="
echo "🤗 PUSHING TO HUGGINGFACE HUB"
echo "=================================="
echo ""

# Check authentication
echo "🔐 Checking HuggingFace authentication..."
if ! hf auth whoami > /dev/null 2>&1; then
    echo "❌ Not logged in"
    echo ""
    echo "📝 AUTHENTICATION REQUIRED"
    echo "=================================="
    echo ""
    echo "1. Go to: https://huggingface.co/settings/tokens"
    echo "2. Create a new token (write access)"
    echo "3. Run: hf auth login"
    echo "4. Paste your token when prompted"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Authenticated!"
echo ""

# Get username
USERNAME=$(hf auth whoami 2>&1 | grep -oP '(?<=logged in as )\S+')
echo "👤 User: $USERNAME"
echo ""

# Model path
MODEL_DIR="trained_models/outputs/french_gpt2_pytorch"
REPO_NAME="french-gpt2-conversations"
FULL_REPO="$USERNAME/$REPO_NAME"

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model not found: $MODEL_DIR"
    exit 1
fi

echo "📦 Model: $MODEL_DIR"
echo "📍 Repository: https://huggingface.co/$FULL_REPO"
echo ""

# Create model card if missing
if [ ! -f "$MODEL_DIR/README.md" ]; then
    echo "📝 Creating model card..."
    cat > "$MODEL_DIR/README.md" << 'CARD'
---
language: fr
tags:
  - french
  - gpt2
  - conversation
  - language-model
  - pytorch
license: mit
---

# French GPT-2 (Conversations)

🇫🇷 GPT-2 model trained exclusively on French conversation data.

## Model Details

- **Architecture**: GPT-2 (124M parameters)
- **Training**: 80,000 steps
- **Final Loss**: 3.10
- **Data**: 85.5M French conversation tokens
- **Precision**: Float32
- **Context**: 1024 tokens

## Training Details

- **Batch size**: 4
- **Learning rate**: 1e-4
- **Optimizer**: AdamW
- **Training time**: 3.25 hours
- **Hardware**: RTX 5080 (16GB VRAM)

## Usage

```python
from transformers import pipeline

generator = pipeline("text-generation", 
    model="<username>/french-gpt2-conversations")

output = generator("Bonjour, ", max_length=100)
print(output[0]["generated_text"])
```

## Data

- 100% French conversations
- Sources: Reddit, Twitter, WebCrawl
- Quality: Cleaned & deduplicated
- Tokens: 85,490,944

## Performance Tips

- Temperature: 0.7 (creative), 0.3 (factual)
- Max tokens: 256-512 recommended
- Top P: 0.9
- Repetition penalty: 1.1

## Limitations

- Conversational bias (trained on conversations)
- Not instruction-tuned
- French-specific (lower English capability)

## License

MIT

## Citation

If you use this model, please cite:
```bibtex
@model{french_gpt2_conversations_2025,
  author={Vincent PRO AI},
  title={French GPT-2 Conversations},
  year={2025}
}
```

## Contact

GitHub: https://github.com/Vincent-PRO-AI/french-llm-from-scratch
CARD
    echo "✅ Model card created"
fi

echo ""
echo "⏳ Uploading to HuggingFace..."
echo ""

# Upload model
huggingface-cli upload "$FULL_REPO" "$MODEL_DIR"/* \
    --repo-type model \
    --private \
    --commit-message "Add French GPT-2 model (124M, trained on 85.5M French tokens)"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ UPLOAD SUCCESSFUL!"
    echo ""
    echo "🌐 Model available at:"
    echo "   https://huggingface.co/$FULL_REPO"
    echo ""
    echo "📦 Use in code:"
    echo "   from transformers import pipeline"
    echo "   p = pipeline('text-generation', model='$FULL_REPO')"
    echo ""
else
    echo ""
    echo "❌ Upload failed!"
    exit 1
fi

echo ""
