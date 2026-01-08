# 🔧 Multi-GPU Training - Optimization & Debugging

## 📊 Performance Analysis

### Théorique vs. Pratique
```
┌──────────────────────────────────────────────────────────┐
│ Speedup Expected (Loi d'Amdahl)                          │
├──────────────────────────────────────────────────────────┤
│ 2 GPUs, ~95% parallelizable:                             │
│   Speedup = 1 / (0.05 + 0.95/2) = 1.90x                  │
│ 2 GPUs, ~90% parallelizable (avec overhead):             │
│   Speedup = 1 / (0.10 + 0.90/2) = 1.82x                  │
│                                                          │
│ Expected: 1.7x - 1.9x speedup (RTX 4090 + RTX 5080)     │
└──────────────────────────────────────────────────────────┘
```

### Baseline (Single GPU RTX 4090)
- **Throughput**: ~120 tokens/sec
- **Batch size**: 12
- **Gradient accumulation**: 2
- **Memory usage**: 22 GB

### Multi-GPU (RTX 4090 + RTX 5080)
- **Expected throughput**: ~200-220 tokens/sec
- **Effective batch size**: 48 (12 per GPU × 2 accum × 2 GPUs)
- **Memory usage**: 23 GB + 12 GB = 35 GB (combined)
- **Expected speedup**: 1.7x - 1.9x

## 🚀 Optimization Techniques

### 1. Batch Size Tuning
```python
# GPU Memory Available
# RTX 4090: 24 GB → batch_size = 12-16
# RTX 5080: 12 GB → batch_size = 6-8

# Effective batch size = batch_size × grad_accum × num_gpus
# Current: 12 × 2 × 2 = 48
# This maintains good gradient updates while fitting VRAM
```

### 2. Gradient Accumulation Strategy
```python
# Accumulate 2 steps before updating
# Pros: Larger effective batch size, stable training
# Cons: Slower per-step throughput
# Trade-off: Best for this setup

gradient_accumulation_steps = 2
# If OOM: increase to 3-4
# If memory abundant: reduce to 1
```

### 3. Mixed Precision (AMP)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Expected: 30-40% speedup, same accuracy
# Memory reduction: 30-50%
```

### 4. Gradient Checkpointing
```python
# Trades computation for memory
model.gradient_checkpointing_enable()

# Memory savings: 50-70%
# Computation overhead: 20-30%
# Use if RTX 5080 shows OOM errors
```

### 5. Flash Attention (if available)
```python
# For RTX 4090 & 5080 with compute capability 8.0+
from flash_attn import flash_attn_func

# 2-4x faster attention, 70% less memory
# Install: pip install flash-attn
```

## 🐛 Debugging Common Issues

### Issue 1: NCCL Errors
```
RuntimeError: NCCL error (unhandled cuda error)
```

**Debugging**:
```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Check network connectivity
docker exec french-llm-training-master ping training-worker

# Verify hostname resolution
docker exec french-llm-training-master getent hosts training-worker
```

**Solutions**:
1. Ensure both containers can ping each other
2. Check Docker network: `docker network inspect llm-network`
3. Verify `MASTER_ADDR=training-master` in environment
4. Use explicit IP instead: `MASTER_ADDR=172.17.0.2`

### Issue 2: Out of Memory (OOM)
```
RuntimeError: CUDA out of memory. Tried to allocate 2.43 GiB
```

**Diagnosis**:
```bash
# Check memory allocation
docker exec french-llm-training-worker nvidia-smi
# Should show used memory < 12 GB for RTX 5080

# Monitor in real-time
docker exec -it french-llm-training-worker nvidia-smi -l
```

**Solutions** (priority order):
1. Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
2. Reduce batch size: `--batch-size 8`
3. Increase gradient accumulation: `--gradient-accumulation-steps 4`
4. Enable mixed precision (AMP)
5. Reduce sequence length: `--seq-length 512`

### Issue 3: Slow Training on RTX 5080
```
Expected: 190+ tokens/sec
Actual: 80 tokens/sec
```

**Diagnosis**:
```bash
# Check GPU utilization
docker exec french-llm-training-worker watch -n 1 nvidia-smi

# Check memory bandwidth
docker exec french-llm-training-worker nvidia-smi dmon
```

**Likely causes**:
1. ✋ **Thermal throttling** → Improve cooling
2. ✋ **Memory bottleneck** → Reduce batch size
3. ✋ **Communication overhead** → RTX 5080 slower, expected

**Solutions**:
1. Reduce batch size to 6-8 (RTX 5080 constraint)
2. Enable AMP for 30% speedup
3. Limit master to batch=16 (it has more VRAM)

### Issue 4: Uneven Training Progress
```
Master makes 100 steps, Worker still at 50
```

**Causes**:
1. Different batch processing speeds
2. Gradient synchronization delays
3. Memory pressure on slower GPU

**Fix**:
```yaml
# Reduce batch size for slower GPU (RTX 5080)
training-worker:
  command: >
    ... --batch-size 8

training-master:
  command: >
    ... --batch-size 12
```

## 📈 Performance Monitoring

### Real-time Dashboard
```bash
# Terminal 1: Logs
tail -f trained_models/runs/french_medium_multi_gpu/training_log.jsonl

# Terminal 2: GPU monitoring
watch -n 1 'nvidia-smi && echo "---" && docker stats --no-stream'

# Terminal 3: Network traffic
docker exec french-llm-training-master iftop -n

# Terminal 4: Web dashboard
open http://localhost:5174
```

### Parse Training Logs
```bash
# Extract key metrics
python << 'EOF'
import json
with open("trained_models/runs/french_medium_multi_gpu/training_log.jsonl") as f:
    for line in f:
        log = json.loads(line)
        print(f"Step {log['step']:6d}: Loss={log['loss']:.4f}, LR={log['lr']:.2e}, GPU={log.get('gpu_memory_mb', 0):.0f}MB")
EOF
```

### Monitor Training Stability
```python
# Check for:
# 1. Loss divergence
# 2. GPU memory leaks
# 3. Training speed degradation

import json
losses = []
with open("training_log.jsonl") as f:
    for line in f:
        log = json.loads(line)
        losses.append(log['loss'])

# Check for NaN or Inf
if any(x != x or abs(x) > 1e6 for x in losses):
    print("⚠️ Loss divergence detected!")

# Check for increasing loss
if losses[-10] > losses[0]:
    print("⚠️ Loss is increasing!")

# Check GPU memory growth
mems = [json.loads(line)['gpu_memory_mb'] for line in open("training_log.jsonl")]
if max(mems[-100:]) > max(mems[:100]) + 1000:
    print("⚠️ Possible memory leak!")
```

## 🎯 Optimization Checklist

### Before Training
- [ ] Verify both GPUs detected: `nvidia-smi`
- [ ] Check Docker network: `docker network ls`
- [ ] Check disk space: `df -h .`
- [ ] Verify data exists: `ls -lh data_clean/`
- [ ] Clear Docker cache: `docker system prune -a` (optional)

### During Training (every 1-2 hours)
- [ ] Monitor GPU memory: `nvidia-smi`
- [ ] Check loss trend: `tail training_log.jsonl`
- [ ] Verify throughput: `tail -20 training_log.jsonl | grep throughput`
- [ ] Monitor network: `docker exec master iftop -n`

### Performance Targets
| Metric | Target | Current |
|--------|--------|---------|
| Throughput | 200+ tokens/sec | TBD |
| GPU Util (4090) | 95%+ | TBD |
| GPU Util (5080) | 85%+ | TBD |
| Memory (4090) | <24 GB | TBD |
| Memory (5080) | <12 GB | TBD |
| Loss trend | Decreasing | TBD |
| Steps/hour | 15000+ | TBD |

## 🔌 Hardware-Specific Tuning

### RTX 4090 (RANK=0, Master)
- **Architecture**: Ada Lovelace, 16384 CUDA cores
- **Memory**: 24 GB GDDR6X, 960 GB/s bandwidth
- **Best for**: Model, optimizer state, accumulation
- **Recommended batch**: 14-16
- **Priority**: Maximize throughput

### RTX 5080 (RANK=1, Worker)
- **Architecture**: AD102, 14080 CUDA cores
- **Memory**: 12 GB GDDR6X, 480 GB/s bandwidth
- **Best for**: Model replica (lighter), gradient sync
- **Recommended batch**: 6-10
- **Priority**: Avoid OOM, reduce strain

## 📊 Expected Metrics

### Healthy Training Session
```json
Step   1: Loss=4.8234, LR=1.00e-05, Time=0.45s, Throughput=214.3 samples/s
Step  10: Loss=4.2156, LR=9.99e-05, Time=0.22s, Throughput=218.2 samples/s
Step 100: Loss=3.1234, LR=9.99e-04, Time=0.22s, Throughput=218.7 samples/s
```

### Warning Signs
```
⚠️ Loss not decreasing (stuck at 4.8)
⚠️ Throughput dropping: 218 → 180 samples/s
⚠️ GPU memory increasing: 23 GB → 24 GB over 10 steps
⚠️ NaN loss detected
```

## 🚨 Emergency Procedures

### Stop Immediately (kill switch)
```bash
docker-compose -f docker-compose.multi-gpu.yml down --remove-orphans
pkill -9 python  # If containers stuck
```

### Recover from Crash
```bash
# 1. Check latest checkpoint
ls -lrt trained_models/runs/french_medium_multi_gpu/checkpoint_*.pt | tail -1

# 2. Resume from checkpoint
docker-compose -f docker-compose.multi-gpu.yml up --profile multi-gpu \
  -e RESUME_FROM="trained_models/runs/french_medium_multi_gpu/checkpoint_step_50000.pt"

# 3. If issue persists, reduce batch size and retry
```

### Clear Cache & Restart
```bash
# Remove all Docker containers
docker container prune -f

# Remove dangling images
docker image prune -a -f

# Restart training
./launch_multi_gpu_training.sh 250000 10
```

## 📝 Next Steps

1. **Run first training**: `./launch_multi_gpu_training.sh`
2. **Monitor for 30 mins**: Check GPU util, memory, throughput
3. **Adjust if needed**: Tune batch size based on actual performance
4. **Validate scaling**: Compare single-GPU vs multi-GPU throughput
5. **Fine-tune hyperparams**: Adjust learning rate if loss unstable

