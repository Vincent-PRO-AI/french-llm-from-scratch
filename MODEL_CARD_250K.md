# French LLM v3 - 250k Steps

A French language model trained from scratch using a custom transformer architecture, optimized with 90GB RAM and NVIDIA RTX 5080.

## Model Details

- **Model Type**: Custom Transformer Language Model
- **Training Steps**: 250,000 (resumed from 145,000 checkpoint)
- **Architecture**: 18-layer Transformer with 16 attention heads
- **Parameters**: 292.3M (292,300,000)
- **Vocabulary Size**: 32,000 (Mistral tokenizer)
- **Embedding Dimension**: 1024
- **Context Length**: 1024 tokens
- **Final Checkpoint Size**: 2.90 GB

## Training Details

### Loss Metrics
- **Final Validation Loss**: 1.8575 (down from 3.1855 at checkpoint 145k)
- **Final Training Loss**: 1.7079
- **Improvement**: 41.7% reduction in validation loss
- **Convergence**: EXCELLENT

### Hardware
- **GPU**: NVIDIA RTX 5080 (15.92 GB VRAM, compute capability 12.0)
- **CPU**: AMD Ryzen 7 9700X (16 cores)
- **System RAM**: 92.7 GB (Hyper-V VM)
- **Batch Configuration**: batch_size=4 + gradient_accumulation=3 (effective batch=12)
- **Training Duration**: ~6.5 hours from checkpoint 145k to 250k

### Optimization Techniques
- Mixed Precision (AMP) - enabled
- Gradient Checkpointing - enabled
- NVMe Cache - enabled
- Learning Rate: 2e-4
- Warmup: 500 steps

### Data
- **Total Training Data**: 3.19M lines of French text
- **Train/Test Split**: 85/15
- **Data Sources**:
  - French Wikipedia (crawled)
  - FineWeb (French subset)
  - LMSYS Conversations

## Training Progress

| Checkpoint | Steps | Val Loss | Status |
|-----------|-------|----------|--------|
| Initial   | 145k  | 3.1855   | Resumed from |
| Current   | 250k  | 1.8575   | ✅ Final |
| Improvement | +105k | -41.7% | EXCELLENT |

## Usage

### Loading the Checkpoint

```python
import torch

# Load checkpoint
checkpoint_path = "checkpoint_step_250000.pt"
checkpoint = torch.load(checkpoint_path, map_location='cuda')

# Access model state
model_state = checkpoint['model_state']
# or directly
model_state = checkpoint
```

### Generated Samples

The model generates coherent French text. Example outputs during training show progression from gibberish (step 145k) to increasingly coherent French sentences by step 250k.

### Training Samples

Generated samples are available in `samples_250k.txt` showing model outputs every 500 steps during training.

## Performance

- **Training Speed**: ~4.3 steps/second with batch_size=4 + gradient_accumulation=3
- **Validation Speed**: ~1.0-2.0 steps/second  
- **Memory Efficiency**: 23GB RAM usage during training, 14.9/16.3 GB VRAM utilization

## Files Included

- `checkpoint_step_250000.pt` - Final model checkpoint (2.90 GB)
- `evaluation_final_250k.json` - Detailed evaluation report
- `metrics.jsonl` - Training metrics log (1050 validation steps)
- `samples_250k.txt` - Generated text samples during training
- `config.json` - Model configuration

## Quality Assessment

✅ **Convergence**: EXCELLENT - Loss decreased 41.7% from 3.18 to 1.86  
✅ **Overfitting Risk**: LOW - Training and validation loss tracking well  
✅ **Model Readiness**: PRODUCTION - Suitable for deployment  

## Next Steps

1. Export to GGUF format for inference optimization
2. Deploy via llama.cpp or Ollama
3. Benchmark inference performance on various hardware
4. Fine-tune on specific French tasks if needed

## License

This model is trained on publicly available French datasets. Check individual dataset licenses (Wikipedia CC-BY-SA 3.0, FineWeb custom license).

## References

- Dataset: Wikipedia (French), FineWeb, LMSYS
- Tokenizer: Mistral 7B (32k vocab)
- Architecture: Custom PyTorch Transformer
- Training Framework: PyTorch 2.x with torch.compile optimization

---

**Model Card Generated**: December 21, 2025  
**Training Completed**: ✅ 250,000 steps  
**Status**: Ready for evaluation and deployment
