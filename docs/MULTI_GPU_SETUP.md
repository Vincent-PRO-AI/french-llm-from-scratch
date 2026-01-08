# 🚀 Configuration Multi-GPU avec Docker

## Hardware Setup
- **CPU**: Hyper-V VM (16 cores, 92.7 GB RAM)
- **GPU 0 (Worker)**: RTX 5080 (12 GB VRAM)
- **GPU 1 (Master)**: RTX 4090 (24 GB VRAM)
- **Total VRAM**: 36 GB
- **Total RAM**: 96 GB

## Architecture
```
┌─────────────────────────────────────────┐
│    Docker Compose Multi-GPU Network     │
├─────────────────────────────────────────┤
│                                         │
│  training-master (RTX 4090)             │
│  ├─ RANK=0, WORLD_SIZE=2               │
│  ├─ Batch=12, Grad_Accum=2             │
│  └─ Effective Batch=48 (12*2*2)        │
│                                         │
│  training-worker (RTX 5080)             │
│  ├─ RANK=1, WORLD_SIZE=2               │
│  ├─ Batch=12, Grad_Accum=2             │
│  └─ Synced with master via NCCL        │
│                                         │
│  backend (monitoring)                   │
│  └─ Watches both GPUs, metrics          │
│                                         │
│  frontend (React dashboard)             │
│  └─ Real-time training visualization    │
│                                         │
└─────────────────────────────────────────┘
```

## Quick Start

### 1️⃣ Vérifier les GPUs
```bash
nvidia-smi -L
# Should show: GPU 0 RTX 5080, GPU 1 RTX 4090
```

### 2️⃣ Lancer le training multi-GPU
```bash
# Terminal 1: Start Docker Compose
cd /home/vincent/code/repo/french-llm-from-scratch
docker-compose -f docker-compose.multi-gpu.yml up --profile multi-gpu

# This starts:
# - training-master (RTX 4090) - RANK=0
# - training-worker (RTX 5080) - RANK=1
# - backend (monitoring)
# - frontend (dashboard)
```

### 3️⃣ Ouvrir le dashboard
```
Dashboard: http://localhost:5174
API: http://localhost:8000/api/metrics
```

### 4️⃣ Arrêter cleanly
```bash
docker-compose -f docker-compose.multi-gpu.yml down
```

## Performance Expectations

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Batch size per GPU | 12 |
| Gradient accumulation | 2 |
| Effective batch size | 48 (12 × 2 × 2 GPUs) |
| Model size | 292M parameters |
| Sequence length | 1024 tokens |

### Estimated Throughput
- **RTX 4090**: ~120 tokens/sec
- **RTX 5080**: ~90 tokens/sec
- **Combined (DDP)**: ~180-200 tokens/sec (near-linear scaling)
- **Training time for 100k steps**: ~12-15 hours

### Memory Usage
- **RTX 4090**: 22-23 GB (with batch=12, grad_accum=2)
- **RTX 5080**: 11-12 GB (optimized)
- **System RAM**: ~60 GB used, 36 GB available

## Optimizations Applied

### 1. GPU Assignment
```yaml
training-master:
  device_ids: ["1"]  # RTX 4090 (more VRAM = master)

training-worker:
  device_ids: ["0"]  # RTX 5080 (less VRAM = worker)
```

### 2. Memory Management
```yaml
training-master:
  mem_limit: 48gb          # More RAM for RTX 4090
  shm_size: 32gb           # Shared memory for data loading

training-worker:
  mem_limit: 32gb          # Less RAM for RTX 5080
  shm_size: 16gb           # Reduced shared memory
```

### 3. NCCL Communication
```python
# Fast GPU-to-GPU communication
import torch.distributed as dist
dist.init_process_group("nccl")

# All-reduce operations during backward pass
```

### 4. Gradient Checkpointing (if needed)
```python
model.gradient_checkpointing_enable()
```

## Monitoring

### During Training
```bash
# In another terminal - Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor Docker stats
docker stats --no-stream
```

### Logs Location
- Training logs: `trained_models/runs/french_medium_multi_gpu/training_log.jsonl`
- Checkpoints: `trained_models/runs/french_medium_multi_gpu/checkpoint_step_*.pt`
- Docker logs:
  ```bash
  docker logs -f french-llm-training-master
  docker logs -f french-llm-training-worker
  ```

## Common Issues

### 1. NCCL Connection Error
```
RuntimeError: NCCL operation failed
```
**Solution**: Ensure network bridge works
```bash
docker network ls
docker network inspect llm-network
```

### 2. Out of Memory Error
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or enable gradient checkpointing
```python
model.gradient_checkpointing_enable()
```

### 3. Slow Training on RTX 5080
```
Solution: Enable AMP (mixed precision)
```python
from torch.cuda.amp import autocast
with autocast():
    outputs = model(input_ids, labels=labels)
```

## Advanced Configuration

### 3+ GPUs (if available)
Modify `docker-compose.multi-gpu.yml`:
```yaml
training-gpu2:
  extends: training-worker
  environment:
    - RANK=2
    - WORLD_SIZE=3
  deploy:
    device_ids:
      - "2"  # Third GPU
```

### Different Batch Sizes per GPU
```yaml
# RTX 4090: higher batch (more VRAM)
training-master:
  command: >
    python -m torch.distributed.launch
    ... --batch-size 16

# RTX 5080: lower batch (less VRAM)
training-worker:
  command: >
    python -m torch.distributed.launch
    ... --batch-size 8
```

### Custom NCCL Settings
```yaml
environment:
  - NCCL_DEBUG=INFO          # Debug NCCL issues
  - NCCL_SOCKET_IFNAME=eth0  # Specify network interface
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

## Resuming from Checkpoint

The training script supports resuming:
```bash
docker-compose -f docker-compose.multi-gpu.yml up \
  --profile multi-gpu \
  -e RESUME_FROM="trained_models/runs/french_medium_multi_gpu/checkpoint_step_50000.pt"
```

Or modify `docker-compose.multi-gpu.yml`:
```yaml
training-master:
  command: >
    python -m torch.distributed.launch
    ... --resume-from trained_models/runs/french_medium_multi_gpu/checkpoint_step_50000.pt
```

## Data Preparation

Before training, ensure data is ready:
```bash
# Prepare tokenized data
python scripts/prepare_training_data.py \
  --output-dir data_clean \
  --train-split 0.85 \
  --val-split 0.15

# This creates:
# - data_clean/train_tokens.bin
# - data_clean/val_tokens.bin
```

## Expected Output

```
================================================================================
  Multi-GPU Training Started
================================================================================
Rank: 0/2 | GPU: 1 (RTX 4090)
Rank: 1/2 | GPU: 0 (RTX 5080)
Max Steps: 250000 | Batch Size: 12
Gradient Accumulation: 2
Effective Batch: 48 (12 × 2 × 2 GPUs)
================================================================================

Step 10 | Loss: 4.2134 | LR: 9.00e-05 | Time: 0.23s | Throughput: 186.5 samples/s | GPU Mem: 23421MB
Step 20 | Loss: 4.1256 | LR: 8.99e-05 | Time: 0.22s | Throughput: 192.1 samples/s | GPU Mem: 23421MB
...
Saved checkpoint: trained_models/runs/french_medium_multi_gpu/checkpoint_step_2500.pt
...
```

## Troubleshooting

### Check if GPUs are detected
```bash
docker exec french-llm-training-master python -c "import torch; print(torch.cuda.device_count())"
```

### Verify distributed initialization
```bash
docker logs french-llm-training-master | grep "init_process_group"
```

### Manual multi-GPU test (non-Docker)
```bash
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  scripts/train_subtitles_transformer_ddp.py \
  --max-steps 100
```

## Performance Tips

1. **Use CUDA 12.1+** for best RTX 4090/5080 performance
2. **Enable NCCL fast path**: `NCCL_FAST_ALLREDUCE=1`
3. **Use DistributedSampler** to avoid duplicate samples
4. **Pin memory** in DataLoader: `pin_memory=True`
5. **Monitor thermal throttling**: RTX 5080 may thermal throttle if not cooled properly

## Next Steps

1. ✅ Start training: `docker-compose -f docker-compose.multi-gpu.yml up --profile multi-gpu`
2. ✅ Monitor via dashboard: http://localhost:5174
3. ✅ Check logs: `tail -f trained_models/runs/french_medium_multi_gpu/training_log.jsonl`
4. ✅ Adjust batch size if OOM: Reduce `--batch-size` in docker-compose.multi-gpu.yml

