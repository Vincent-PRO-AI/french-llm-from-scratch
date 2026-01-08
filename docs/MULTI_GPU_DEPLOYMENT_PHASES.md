# 🚀 Configuration Multi-GPU - Guide Étapes

## 🔧 Situation Actuelle
- **GPU disponible maintenant**: RTX 5080 (12 GB accesible, 15 GB total)
- **GPUs en préparation**: RTX 4090 (à connecter), second RTX 5080 / autre
- **RAM système**: 86 GB (cible: 96 GB)
- **Disque**: 243 GB libre ✅

## 🎯 Phases de Déploiement

### Phase 1: Single GPU Optimized (Actuellement) ✅
**Objectif**: Optimiser l'entraînement sur RTX 5080 seule avec la 5080 visible

```yaml
# Utiliser: docker-compose.yml (original)
# Configuration:
GPU: 0 (RTX 5080)
Batch size: 12
Gradient accum: 2
Max batch: 24 effective
Throughput: ~90 tokens/sec
```

**Lancer**:
```bash
docker-compose up --profile training
```

### Phase 2: Dual GPU Hyper-V (Quand RTX 4090 connectée) ⏳
**Prérequis**:
1. Connecter RTX 4090 au Hyper-V
2. Vérifier avec `nvidia-smi` (doit voir 2 GPUs)
3. Relancer validation

**Configuration**:
```bash
nvidia-smi -L
# GPU 0: RTX 5080 (12 GB)
# GPU 1: RTX 4090 (24 GB)
```

**Lancer**:
```bash
docker-compose -f docker-compose.multi-gpu.yml up --profile multi-gpu
```

### Phase 3: Multi-Node (Optionnel - Futur)
Quand vous aurez 3+ GPUs, utiliser Kubernetes ou multi-host DDP.

---

## 📊 Configuration Actuelle Recommandée

### Option A: Single RTX 5080 (Immédiat)
```bash
# Utiliser la config existante
docker-compose up --profile training
```

**Performance**:
- Batch: 12
- Grad accum: 2
- Throughput: ~90 tokens/sec
- 250k steps: ~77 heures

### Option B: Préparer pour Multi-GPU
**Si vous avez accès à la RTX 4090 maintenant**:

```bash
# Vérifier les deux GPUs
nvidia-smi -L

# Si vus:
./launch_multi_gpu_training.sh 250000 12

# Sinon, attendre la connexion et revenir à Phase 1
```

---

## 🔗 Intégration Multi-GPU (Quand 2 GPUs disponibles)

### Étape 1: Vérifier les 2 GPUs
```bash
nvidia-smi -L
# GPU 0: NVIDIA GeForce RTX 5080
# GPU 1: NVIDIA GeForce RTX 4090

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 0, GeForce RTX 5080, 12288 MiB
# 1, GeForce RTX 4090, 24576 MiB
```

### Étape 2: Valider la Setup Multi-GPU
```bash
python validate_multi_gpu_setup.py
# Doit montrer ✅ pour tous les checks
```

### Étape 3: Lancer Training Multi-GPU
```bash
# Option 1: Via script helper (recommandé)
chmod +x launch_multi_gpu_training.sh
./launch_multi_gpu_training.sh 250000 12

# Option 2: Via docker-compose directement
docker-compose -f docker-compose.multi-gpu.yml up --profile multi-gpu
```

### Étape 4: Monitorage
```bash
# Terminal 1: Logs training
tail -f trained_models/runs/french_medium_multi_gpu/training_log.jsonl

# Terminal 2: GPU monitoring
watch -n 2 'nvidia-smi'

# Terminal 3: Dashboard (port 5174)
# Terminal 4: API metrics (port 8000)
```

---

## 🛠️ Actions pour Préparer Multi-GPU Maintenant

Même si vous n'avez pas les 2 GPUs visibles, vous pouvez préparer:

1. **Test single GPU setup**:
   ```bash
   python validate_multi_gpu_setup.py
   # Vérifier les ressources CPU/RAM/Disque
   ```

2. **Pré-valider la config multi-GPU**:
   ```bash
   docker-compose -f docker-compose.multi-gpu.yml config
   # Doit montrer pas d'erreurs
   ```

3. **Pré-build les images Docker**:
   ```bash
   docker-compose -f docker-compose.multi-gpu.yml build
   # Prépare les images pour lancement rapide
   ```

4. **Préparer les données**:
   ```bash
   # Vérifier data_clean/train_tokens.bin existe
   ls -lh data_clean/
   ```

---

## 📋 Checklist Multi-GPU Launch

- [ ] Vérifier 2 GPUs: `nvidia-smi -L` (montre 2 GPUs)
- [ ] Valider setup: `python validate_multi_gpu_setup.py`
- [ ] Vérifier Docker: `docker compose version`
- [ ] Vérifier données: `ls -lh data_clean/train_tokens.bin`
- [ ] Vérifier espace disque: `df -h .` (>50 GB libre)
- [ ] Pré-build images: `docker-compose -f docker-compose.multi-gpu.yml build`
- [ ] Tester NCCL: `docker-compose -f docker-compose.multi-gpu.yml run training-master python -c "import torch.distributed as dist; print('NCCL OK')"`
- [ ] Lancer: `./launch_multi_gpu_training.sh`

---

## 🚀 Quand Vous Voyez 2 GPUs

### Immédiatement:
```bash
cd /home/vincent/code/repo/french-llm-from-scratch
python validate_multi_gpu_setup.py  # Doit passer tous les checks
./launch_multi_gpu_training.sh 250000 12
```

### Résultat attendu:
```
✅ GPU count (2): Found 2 GPU(s)
✅ Total VRAM (36 GB): Total 36 GB
✅ Docker network connectivity

Training started:
  Master (RTX 4090): RANK=0
  Worker (RTX 5080): RANK=1
  Effective batch: 48 (12 × 2 × 2 GPUs)
  Expected throughput: 200+ tokens/sec
```

---

## 💾 Fichiers Créés Pour Support Multi-GPU

| Fichier | Objectif |
|---------|----------|
| `docker-compose.multi-gpu.yml` | Configuration Docker multi-GPU |
| `scripts/train_subtitles_transformer_ddp.py` | Training script DDP |
| `launch_multi_gpu_training.sh` | Launcher avec validation |
| `validate_multi_gpu_setup.py` | Validateur complet |
| `MULTI_GPU_SETUP.md` | Guide détaillé |
| `MULTI_GPU_OPTIMIZATION.md` | Optimization & debugging |

---

## 🔄 Workflow Recommandé

### Aujourd'hui (Avec RTX 5080)
```bash
# Option 1: Continuer avec single GPU
docker-compose up --profile training

# Ou

# Option 2: Tester la config prête pour multi-GPU
python validate_multi_gpu_setup.py
# Vérifier les ressources CPU/RAM/Disque

# Ou

# Option 3: Préparer Docker images
docker-compose -f docker-compose.multi-gpu.yml build
```

### Dès que 2 GPUs visibles
```bash
python validate_multi_gpu_setup.py  # Valider
./launch_multi_gpu_training.sh      # Lancer
tail -f trained_models/runs/french_medium_multi_gpu/training_log.jsonl
```

---

## 🎯 Résumé Next Steps

1. **Vérifiez GPU actuel**:
   ```bash
   nvidia-smi -L
   ```

2. **Exécutez validation**:
   ```bash
   python validate_multi_gpu_setup.py
   ```

3. **Choisissez action**:
   - **A)** Si 1 GPU visible → Utilisez `docker-compose up --profile training`
   - **B)** Si 2 GPUs visibles → Utilisez `./launch_multi_gpu_training.sh`
   - **C)** Si préparation seulement → Exécutez `docker-compose -f docker-compose.multi-gpu.yml build`

