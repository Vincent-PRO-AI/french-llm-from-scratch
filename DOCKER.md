# 🐳 Docker Setup for French LLM Training

## Prerequisites

- **Docker** 20.10+ with Compose V2
- **NVIDIA Docker Runtime** (for GPU support)
- **GPU:** NVIDIA GPU with CUDA support (RTX 5080, RTX 5090, A100, etc.)
- **Driver:** NVIDIA Driver 525+ (for CUDA 12.1)

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 Quick Start

### 1. Launch Dashboard Only

Monitor training progress with the web interface:

```bash
# Start backend + frontend
docker-compose up -d backend frontend

# Access dashboard at http://localhost:5174
# API accessible at http://localhost:8000
```

### 2. Run Training

```bash
# Base training (60k steps)
docker-compose run --rm training python scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --tokenizer-path trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --pretokenized-path data_clean/mixed_tokenized.pt \
  --max-steps 60000 \
  --batch-size 4 \
  --lr 0.0001 \
  --checkpoint-interval 5000

# With dashboard monitoring (in separate terminal)
docker-compose up -d backend frontend
docker-compose run --rm training python scripts/train_subtitles_transformer.py [options]
```

### 3. Fine-Tuning

```bash
# Fine-tune on conversations
docker-compose run --rm training python finetune_conversations.py
```

### 4. Testing

```bash
# Generate samples
docker-compose run --rm training python test_model_samples.py

# Test chat capabilities
docker-compose run --rm training python test_chat_model.py
```

---

## 📋 Common Commands

### Build Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build backend
docker-compose build frontend
```

### Data Preparation

```bash
# Download and prepare Wikipedia
docker-compose run --rm training python scripts/prepare_wikipedia.py --max-articles 3000
docker-compose run --rm training python scripts/clean_wikipedia.py

# Tokenize corpus
docker-compose run --rm training python scripts/pretokenize_corpus.py \
  --input-dir data_clean/wikipedia \
  --tokenizer trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --output data_clean/wikipedia_tokenized.pt

# Download conversations
docker-compose run --rm training python scripts/download_conversations.py --max-conversations 5000
```

### View Logs

```bash
# Backend logs
docker-compose logs -f backend

# Frontend logs
docker-compose logs -f frontend

# Training logs (if running as service)
docker-compose logs -f training
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Interactive Shell

```bash
# Enter training container
docker-compose run --rm training bash

# Enter backend container
docker exec -it french-llm-backend bash
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root:

```env
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Training Parameters
MAX_STEPS=60000
BATCH_SIZE=4
LEARNING_RATE=0.0001
CHECKPOINT_INTERVAL=5000

# Dashboard
VITE_API_URL=http://localhost:8000
```

### Custom Training Script

Create `docker-train.sh`:

```bash
#!/bin/bash
docker-compose run --rm training python scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --tokenizer-path trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --pretokenized-path data_clean/mixed_tokenized.pt \
  --max-steps ${MAX_STEPS:-60000} \
  --batch-size ${BATCH_SIZE:-4} \
  --lr ${LEARNING_RATE:-0.0001} \
  --checkpoint-interval ${CHECKPOINT_INTERVAL:-5000}
```

Make executable and run:
```bash
chmod +x docker-train.sh
./docker-train.sh
```

---

## 📦 Volume Persistence

All important data is persisted via volumes:

```yaml
volumes:
  - ./data_clean:/workspace/data_clean          # Tokenized datasets
  - ./trained_models:/workspace/trained_models  # Checkpoints & tokenizers
  - ./scripts:/workspace/scripts                # Training scripts (live reload)
  - ./dashboard:/workspace/dashboard            # Dashboard code (live reload)
```

**Benefits:**
- ✅ Checkpoints survive container restarts
- ✅ Data preparation happens once
- ✅ Code changes reflected immediately
- ✅ Easy backup/transfer

---

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, check Docker daemon config
cat /etc/docker/daemon.json
# Should contain:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}

# Restart Docker
sudo systemctl restart docker
```

### Out of Memory (OOM)

Reduce batch size in training command:
```bash
docker-compose run --rm training python scripts/train_subtitles_transformer.py \
  --batch-size 2  # Instead of 4
```

Or set memory limits in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 32G
```

### Port Already in Use

```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Backend: 8001 instead of 8000
  - "5175:5174"  # Frontend: 5175 instead of 5174
```

### Slow Build

Use BuildKit for faster builds:
```bash
DOCKER_BUILDKIT=1 docker-compose build
```

### Permission Issues

If volumes have permission errors:
```bash
# Fix ownership
sudo chown -R $USER:$USER data_clean/ trained_models/

# Or run container as current user
docker-compose run --rm --user $(id -u):$(id -g) training bash
```

---

## 🚢 Production Deployment

### Build for Production

```bash
# Build optimized images
docker-compose -f docker-compose.prod.yml build

# Push to registry
docker tag french-llm-training:latest your-registry/french-llm-training:latest
docker push your-registry/french-llm-training:latest
```

### Cloud Deployment (Azure/AWS)

**Azure Container Instances:**
```bash
az container create \
  --resource-group llm-training \
  --name french-llm \
  --image your-registry/french-llm-training:latest \
  --gpu-count 1 \
  --gpu-sku V100 \
  --cpu 8 \
  --memory 32
```

**AWS ECS with GPU:**
```bash
# Create task definition with GPU requirements
# Launch on p3.2xlarge instance with Tesla V100
```

---

## 📊 Resource Usage

**Small Training (medium preset, 60k steps):**
- CPU: 4-8 cores
- RAM: 16-32GB
- GPU: RTX 5080 (16GB) or better
- Storage: ~50GB (data + checkpoints)

**Large Training (scaling up):**
- CPU: 16+ cores
- RAM: 64GB+
- GPU: RTX 5090 (24GB), A100 (40/80GB)
- Storage: 100GB+

---

## 🔗 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [Main README](README.md)

---

**✨ Docker simplifie le partage et la reproductibilité du projet !**
