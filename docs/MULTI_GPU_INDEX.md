# 📚 Multi-GPU Infrastructure - Complete Index

## 🎯 START HERE

**New to multi-GPU training?** → `QUICK_START_MULTI_GPU.md`  
**Need guidance?** → `python multi_gpu_decision_tree.py`  
**Want to validate?** → `python validate_multi_gpu_setup.py`  

---

## 📁 File Structure

### 🚀 Quick References
- **`QUICK_START_MULTI_GPU.md`** - 3 actions quick reference
- **`multi_gpu_decision_tree.py`** - Interactive decision helper
- **`MULTI_GPU_SUMMARY.md`** - Executive summary

### 🔧 Configuration Files
- **`docker-compose.multi-gpu.yml`** - Docker multi-GPU setup (Master + Worker)
- **`scripts/train_subtitles_transformer_ddp.py`** - DDP training script
- **`launch_multi_gpu_training.sh`** - Training launcher script
- **`validate_multi_gpu_setup.py`** - Setup validator

### 📖 Documentation
- **`MULTI_GPU_SETUP.md`** - Complete setup guide (7 sections)
- **`MULTI_GPU_OPTIMIZATION.md`** - Optimization & debugging (7 sections)
- **`MULTI_GPU_DEPLOYMENT_PHASES.md`** - Deployment phases (3 phases)

---

## 📊 Decision Matrix

| Situation | Action | File |
|-----------|--------|------|
| **Just validating** | `python validate_multi_gpu_setup.py` | - |
| **1 GPU (RTX 5080)** | `docker-compose up --profile training` | `docker-compose.yml` |
| **2 GPUs ready** | `./launch_multi_gpu_training.sh` | `docker-compose.multi-gpu.yml` |
| **Need guidance** | `python multi_gpu_decision_tree.py` | - |
| **Need debugging** | `MULTI_GPU_OPTIMIZATION.md` | - |
| **Want details** | `MULTI_GPU_SETUP.md` | - |

---

## 🎓 Learning Path

### Level 1: Understanding (15 min)
1. Read: `QUICK_START_MULTI_GPU.md` (section "Key Concepts")
2. Read: `MULTI_GPU_SUMMARY.md` (sections "DDP", "NCCL", "Effective Batch Size")

### Level 2: Getting Started (30 min)
1. Run: `python validate_multi_gpu_setup.py`
2. Read: `MULTI_GPU_DEPLOYMENT_PHASES.md` (understand phases)
3. Choose your path: single GPU or wait for multi-GPU

### Level 3: Running Training (1 hour)
1. Launch: `./launch_multi_gpu_training.sh` (or docker-compose)
2. Monitor: Dashboard + logs
3. Troubleshoot if needed: `MULTI_GPU_OPTIMIZATION.md`

### Level 4: Optimization (2+ hours)
1. Profile: `nvidia-smi dmon`
2. Study: `MULTI_GPU_OPTIMIZATION.md` (all sections)
3. Tune hyperparameters based on metrics

---

## 🚀 Quick Commands

### Validation
```bash
python validate_multi_gpu_setup.py
```

### Single GPU Training
```bash
docker-compose up --profile training
```

### Multi-GPU Training (when 2 GPUs available)
```bash
./launch_multi_gpu_training.sh 250000 12
```

### Interactive Guide
```bash
python multi_gpu_decision_tree.py
```

### Monitoring
```bash
# Terminal 1: Training logs
tail -f trained_models/runs/french_medium_multi_gpu/training_log.jsonl

# Terminal 2: GPU monitoring
watch -n 2 nvidia-smi

# Terminal 3: Dashboard
open http://localhost:5174
```

---

## 📋 File Details

### QUICK_START_MULTI_GPU.md
- 3 quick actions
- Performance comparison table
- Pre-flight checklist
- Monitoring commands
- Quick troubleshooting

### multi_gpu_decision_tree.py
- Interactive CLI guide
- GPU detection
- Action recommendations
- Debugging helpers
- Solution suggestions

### MULTI_GPU_SUMMARY.md
- 5 files overview
- Performance expectations
- System requirements
- File reference table
- Integration points

### docker-compose.multi-gpu.yml
- Backend service (monitoring)
- Training-master service (RTX 4090, RANK=0)
- Training-worker service (RTX 5080, RANK=1)
- Frontend service (dashboard)
- Network configuration

### train_subtitles_transformer_ddp.py
- DDP distributed training
- Multi-node support
- Checkpoint management
- Performance logging
- Gradient accumulation support

### launch_multi_gpu_training.sh
- GPU detection
- Docker image building
- Parameter validation
- Startup instructions
- Cleanup on exit

### validate_multi_gpu_setup.py
- GPU availability check
- Docker configuration check
- System resources check
- Project structure check
- Network connectivity check

### MULTI_GPU_SETUP.md
1. Hardware Setup (current + target)
2. Architecture diagram
3. Quick Start (5 steps)
4. Performance Expectations table
5. Optimizations Applied
6. Monitoring & Logs
7. Common Issues + Solutions

### MULTI_GPU_OPTIMIZATION.md
1. Performance Analysis (theory vs practice)
2. 5 Optimization Techniques
3. 4 Common Issues Debugging
4. Performance Monitoring methods
5. Hardware-Specific Tuning
6. Expected Metrics
7. Emergency Procedures

### MULTI_GPU_DEPLOYMENT_PHASES.md
1. Phase 1: Single GPU Optimized (current)
2. Phase 2: Dual GPU Hyper-V (next)
3. Phase 3: Multi-Node (future)
4. Checklist & Workflow

---

## 🎯 Common Use Cases

### "I want to start training now"
1. `python validate_multi_gpu_setup.py`
2. `docker-compose up --profile training`
3. Monitor via dashboard

### "I want multi-GPU when ready"
1. `python validate_multi_gpu_setup.py` (check resources)
2. `docker-compose -f docker-compose.multi-gpu.yml build` (prepare)
3. Wait for RTX 4090 to be visible
4. `./launch_multi_gpu_training.sh` (launch)

### "I'm getting errors"
1. `python multi_gpu_decision_tree.py` (interactive help)
2. or `MULTI_GPU_OPTIMIZATION.md` (debugging section)
3. Check GPU logs: `docker logs french-llm-training-master`

### "I want to understand multi-GPU"
1. `MULTI_GPU_SUMMARY.md` (quick concepts)
2. `MULTI_GPU_SETUP.md` (detailed guide)
3. `MULTI_GPU_OPTIMIZATION.md` (deep dive)

---

## 🔄 Workflow

### Setup Phase
```
validate_multi_gpu_setup.py
    ↓
Choose: Single GPU or Multi-GPU?
    ├─→ Single GPU: docker-compose up
    └─→ Multi-GPU: ./launch_multi_gpu_training.sh
```

### Training Phase
```
Monitor:
  ├─→ Dashboard: http://localhost:5174
  ├─→ Logs: tail -f training_log.jsonl
  └─→ GPU: nvidia-smi
```

### Optimization Phase
```
If issues:
  ├─→ OOM: Reduce batch size
  ├─→ Slow: Check GPU util, enable AMP
  ├─→ NCCL: Restart Docker, rebuild
  └─→ Other: Check MULTI_GPU_OPTIMIZATION.md
```

---

## 📊 Key Metrics

### Performance Targets
| Metric | Single GPU | Dual GPU | Target |
|--------|-----------|----------|--------|
| Throughput | ~90 tok/s | ~200 tok/s | 200+ |
| GPU Util | 85%+ | 90%+ | 90%+ |
| Memory | ~12 GB | ~35 GB | <36 GB |
| Steps/Hour | 5000+ | 11000+ | 10000+ |

### System Resources
| Resource | Current | Required |
|----------|---------|----------|
| GPU VRAM | 15 GB | 36 GB (2 GPUs) |
| System RAM | 86 GB | 96 GB |
| Disk | 243 GB | 50 GB min |
| CPU cores | 16 | 8+ |

---

## 🔗 Integration

### Already Integrated
- ✅ Tracing (OpenTelemetry - see tracing_*.py)
- ✅ Dashboard (Flask/React - port 5174)
- ✅ Checkpoint system (resume support)

### Ready to Integrate
- ⏳ Pre-flight checklist (multi-GPU aware)
- ⏳ Auto-scaling based on GPU detect
- ⏳ Energy monitoring

---

## 💡 Pro Tips

1. **Faster Docker builds**: `docker-compose -f docker-compose.multi-gpu.yml build --parallel`
2. **Debug NCCL**: `export NCCL_DEBUG=INFO`
3. **Profile training**: `nvidia-smi dmon -s pucm -n 100`
4. **Memory leak detection**: Monitor `nvidia-smi` every hour
5. **Quick benchmark**: `python -m torch.distributed.launch --nproc_per_node=2 --help`

---

## 📞 Getting Help

| Problem | Solution |
|---------|----------|
| "Which file should I read?" | Start with `QUICK_START_MULTI_GPU.md` |
| "How do I choose my path?" | Run `python multi_gpu_decision_tree.py` |
| "Is my setup ready?" | Run `python validate_multi_gpu_setup.py` |
| "I'm stuck on an error" | Check `MULTI_GPU_OPTIMIZATION.md` |
| "I want all details" | Read `MULTI_GPU_SETUP.md` |

---

## ✅ Completion Checklist

- [ ] Read `QUICK_START_MULTI_GPU.md`
- [ ] Run `validate_multi_gpu_setup.py`
- [ ] Choose your path (single or multi-GPU)
- [ ] Run training script
- [ ] Monitor first 30 minutes
- [ ] Adjust hyperparameters if needed
- [ ] Set up automated monitoring

---

Last Updated: January 3, 2026
Files Version: 1.0
Status: Ready for production

