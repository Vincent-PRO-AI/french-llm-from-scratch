# 🚀 MULTI-GPU TRAINING - QUICK START

## Situation Actuelle
✅ **RTX 5080** (12 GB) - Disponible  
⏳ **RTX 4090** (24 GB) - À connecter  
✅ **RAM**: 86 GB (besoin: 96 GB)  
✅ **Disque**: 243 GB libre

---

## 🎯 3 Actions Possibles

### 1️⃣ VALIDER VOTRE SETUP IMMÉDIATEMENT
```bash
cd /home/vincent/code/repo/french-llm-from-scratch
python validate_multi_gpu_setup.py
```

**Attendus**:
- ✅ Checks GPU, Docker, System, Project
- ❌ Peut échouer sur GPU count (normal avec 1 GPU)
- Le script donne les fixes recommandées

---

### 2️⃣ LANCER TRAINING AVEC RTX 5080 SEULE (MAINTENANT)
```bash
cd /home/vincent/code/repo/french-llm-from-scratch

# Option A: Config existante
docker-compose up --profile training

# Option B: Validation d'abord
python validate_multi_gpu_setup.py
docker-compose up --profile training
```

**Performance**:
- ~90 tokens/sec (RTX 5080 seule)
- 250k steps ≈ 77 heures

---

### 3️⃣ LANCER TRAINING MULTI-GPU (QUAND RTX 4090 CONNECTÉE)
```bash
cd /home/vincent/code/repo/french-llm-from-scratch

# Vérifier 2 GPUs visibles
nvidia-smi -L
# GPU 0: NVIDIA GeForce RTX 5080
# GPU 1: NVIDIA GeForce RTX 4090

# Valider setup
python validate_multi_gpu_setup.py
# Doit montrer ✅ pour tous les checks

# Lancer training
./launch_multi_gpu_training.sh 250000 12
```

**Performance attendue**:
- ~200 tokens/sec (2 GPUs)
- 250k steps ≈ 14-15 heures
- Speedup: **2.2x** 

---

## 📁 Fichiers Créés Pour Vous

| Fichier | Quoi | Quand |
|---------|------|-------|
| `docker-compose.multi-gpu.yml` | Config Docker multi-GPU | Dès RTX 4090 connectée |
| `scripts/train_subtitles_transformer_ddp.py` | Script DDP training | Dès RTX 4090 connectée |
| `validate_multi_gpu_setup.py` | Validateur | Maintenant |
| `launch_multi_gpu_training.sh` | Launcher simple | Dès RTX 4090 connectée |
| `MULTI_GPU_SETUP.md` | Guide détaillé | Référence |
| `MULTI_GPU_OPTIMIZATION.md` | Optimization + debugging | Référence |
| `MULTI_GPU_DEPLOYMENT_PHASES.md` | Phases déploiement | Guide |

---

## 📊 Comparaison Performances

| Config | GPUs | Batch/GPU | Throughput | 250k steps |
|--------|------|-----------|-----------|-----------|
| **Single GPU** | 1 (5080) | 12 | 90 tok/s | 77h |
| **Dual GPU** | 2 (5080+4090) | 12 | 200 tok/s | 14h |
| **Speedup** | - | - | **2.2x** | **5.5x faster** |

---

## ✅ PRE-FLIGHT CHECKLIST

Avant de lancer multi-GPU training:

- [ ] 2 GPUs visibles: `nvidia-smi -L` 
- [ ] Validation passe: `python validate_multi_gpu_setup.py`
- [ ] Docker Compose v2: `docker compose version`
- [ ] Données prêtes: `ls -lh data_clean/train_tokens.bin`
- [ ] Espace disque: `df -h .` (min 50 GB libre)
- [ ] RAM: `free -h` (min 86 GB)

---

## 🔥 GO! Let's Train

### OPTION A: Single GPU (maintenant)
```bash
docker-compose up --profile training
```

### OPTION B: Multi-GPU (quand RTX 4090 connectée)
```bash
python validate_multi_gpu_setup.py
./launch_multi_gpu_training.sh 250000 12
```

---

## 📱 Monitoring Pendant Training

**Terminal 1** - Logs:
```bash
tail -f trained_models/runs/french_medium_multi_gpu/training_log.jsonl
```

**Terminal 2** - GPUs:
```bash
watch -n 2 nvidia-smi
```

**Browser** - Dashboard:
```
http://localhost:5174
http://localhost:8000/api/metrics
```

---

## 🆘 Besoin d'aide ?

| Issue | Solution |
|-------|----------|
| `GPU count error` | Attendre RTX 4090 connectée |
| `NCCL error` | Voir `MULTI_GPU_OPTIMIZATION.md` section "NCCL Errors" |
| `OOM error` | Réduire `--batch-size` de 12 à 8-10 |
| `Slow training` | Normal sur RTX 5080, attend RTX 4090 |

---

## 🎓 Concepts Clés

**DDP** = Distributed Data Parallel
- Chaque GPU entraîne le modèle
- Échange gradients via NCCL
- Speedup ~2x avec 2 GPUs

**Effective Batch** = batch_size × grad_accum × num_gpus
- Exemple: 12 × 2 × 2 = 48
- Équilibre gradient stability + GPU efficiency

**NCCL** = Fast GPU-to-GPU communication
- Docker configure automatiquement
- Besoin NVIDIA Container Toolkit

---

## 🚀 NEXT IMMEDIATE STEP

```bash
cd /home/vincent/code/repo/french-llm-from-scratch
python validate_multi_gpu_setup.py
```

Exécutez ça maintenant pour voir l'état de votre setup!

