# Setup Instructions for Training on New PC

## Prerequisites
- CUDA 12.1+ (check with `nvidia-smi`)
- Python 3.11+
- Git
- WSL2 (Windows) or Linux

## Step 1: Clone the Repository
```bash
git clone https://github.com/Vincent-PRO-AI/french-llm-from-scratch.git
cd french-llm-from-scratch
git checkout feat/french-llm-training
```

## Step 2: Setup Python Environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or use conda
# conda create -n defendgpt python=3.11
# conda activate defendgpt
```

## Step 3: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Step 4: Copy Data & Checkpoints
Copy the backed-up data from your external drive:

**From Windows Explorer or WSL:**
```bash
# Copy from D:\backupllmfromscratch\ to your new workspace:
# - data_clean/        → /path/to/defendGPT/data_clean/
# - trained_models/    → /path/to/defendGPT/trained_models/
# - tokenizers/        → /path/to/defendGPT/tokenizers/ (if needed)
```

**Or via terminal (WSL/Linux):**
```bash
# If on WSL, assuming D: is mounted at /mnt/d
rsync -avh /mnt/d/backupllmfromscratch/data_clean/ ./data_clean/
rsync -avh /mnt/d/backupllmfromscratch/trained_models/ ./trained_models/
rsync -avh /mnt/d/backupllmfromscratch/tokenizers/ ./tokenizers/
```

## Step 5: Verify Setup
```bash
# Check GPU is available
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Check datasets exist
ls -lah data_clean/conversations_train_20M_tokenized.pt
ls -lah trained_models/checkpoint_step_17500.pt
```

## Step 6: Launch Training

### Option A: Direct Python (Recommended)
```bash
# Phase 2 fine-tuning on 20M tokens
python scripts/v2/train_phase2_conversations.py

# Parameters: batch_size=10, grad_accum_steps=5, warmup=1000 steps
# Expected speed: ~800-1000 ms/step with optimizations (TF32, BF16, Flash Attention 2, FusedAdamW)
```

### Option B: Docker
```bash
# Build image
docker build -f Dockerfile -t defendgpt:latest .

# Run training with GPU
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  defendgpt:latest \
  python scripts/v2/train_phase2_conversations.py
```

### Option C: Docker Compose
```bash
docker compose up training
```

## Step 7: Monitor Training

### TensorBoard (if enabled in script)
```bash
tensorboard --logdir runs/
# Open: http://localhost:6006
```

### Check GPU Status
```bash
watch nvidia-smi
```

### View Training Loss
```bash
tail -f training_output.log  # If script logs to file
```

## Available Training Scripts

| Script | Purpose | Data | Speed | Status |
|--------|---------|------|-------|--------|
| `scripts/v2/train_phase2_conversations.py` | Phase 2 fine-tuning | 20M tokens (conversations) | ~800ms/step | ✅ Recommended |
| `scripts/v2/train_phase2_optimized.py` | Optimized with DataLoader | 103M tokens | ~1500ms/step | ⚠️ DataLoader overhead |
| `scripts/v2/train_phase2_fast.py` | Direct batching experiments | Large | Variable | 🔧 Development |

## Checkpoint & Resume

### Latest Checkpoint
- Location: `trained_models/checkpoint_step_17500.pt` (Phase 2)
- Automatically loaded at start if available

### Continue Training
Script automatically detects latest checkpoint and resumes:
```bash
python scripts/v2/train_phase2_conversations.py
# Script will load from checkpoint_step_*.pt and continue
```

### Reset Training
Remove checkpoint and logs:
```bash
rm trained_models/checkpoint_step_*.pt
rm -rf runs/
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in script (default: 10)
- Reduce `context_length` (default: 2048)
- Enable gradient checkpointing (already enabled)

### Slow Speed
- Ensure CUDA is active: `nvidia-smi`
- Check GPU utilization: `nvidia-smi -lms 100` (watch mode)
- Verify Flash Attention 2 is installed: `python -c "import flash_attn"`

### Dataset Not Found
- Verify paths in `data_clean/` and checkpoint location
- Run: `ls -lah data_clean/conversations_train_20M_tokenized.pt`

## File Structure
```
french-llm-from-scratch/
├── data_clean/                           # Datasets (70 GB)
│   ├── conversations_train_20M_tokenized.pt
│   ├── conversations_mega_tokenized.pt
│   ├── corpus_10g.bin
│   └── ...
├── trained_models/                       # Checkpoints & Models (58 GB)
│   ├── checkpoint_step_17500.pt
│   ├── gguf/                             # GGUF quantized models
│   ├── huggingface/                      # HuggingFace format
│   └── ...
├── tokenizers/
│   └── vincent_tokenizer_FR_v2/          # SentencePiece 32k Unigram
├── scripts/
│   ├── v2/
│   │   ├── train_phase2_conversations.py
│   │   ├── train_phase2_optimized.py
│   │   ├── train_phase2_fast.py
│   │   └── create_partitioned_dataset.py
│   └── ...
├── training_configs/
│   └── v2_from_scratch.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Performance Tips

### Optimization Flags (Already Enabled)
- ✅ TF32 for faster matrix ops
- ✅ BF16 mixed precision
- ✅ Flash Attention 2
- ✅ xFormers memory-efficient attention
- ✅ FusedAdamW optimizer
- ✅ Gradient checkpointing
- ✅ Cosine annealing + warmup

### Expected Performance
- **GPU**: RTX 5080 / A100 / H100
- **Batch Size**: 10 (conversations)
- **Tokens/sec**: ~2000-2500 (optimized)
- **Time per 1K steps**: ~10-15 minutes

## Questions?
Check logs in `runs/` or refer to inline comments in training scripts.
