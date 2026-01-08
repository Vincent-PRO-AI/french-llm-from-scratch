# French GPT-2 - LM Studio Setup

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
