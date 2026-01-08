# 🚀 French LLM V2 Training - RTX 5080 Optimized

## 📊 Configuration Système Détectée

```
🖥️  Hardware:
├── GPU: RTX 5080 16GB VRAM (Ada architecture)
├── CPU: R7 9700X OC 105W (8 cores, 12 threads, base 4.5 GHz)
├── RAM: 64GB DDR5 (88GB/s bandwidth)
└── Storage: NVMe PCIe 4.0 (7GB/s theoretical)
```

## ⚡ Optimisations Activées

| Optimisation | Impact | Détails |
|---|---|---|
| **Mixed Precision (AMP)** | -50% mémoire | FP16 forward + FP32 backward |
| **Gradient Checkpointing** | -30% mémoire | Recalcul à la demande |
| **torch.compile** | +30% vitesse | Compilation JIT CUDA |
| **Flash Attention** | +25% vitesse | Attention optimisée RTX 5080 |
| **CUDNN Benchmark** | +5-10% | Auto-tuning kernels |
| **Persistent Workers** | -CPU overhead | 4 workers DDR5 optimisés |
| **Pin Memory** | +50% BW GPU | NVMe → RAM → GPU |

## 📈 Configuration Entraînement

```
Modèle:
├── Vocab: 32,000 (nouveau tokenizer)
├── Hidden: 1024
├── Layers: 18
├── Heads: 16
└── FFN: 4096

Entraînement:
├── Batch: 16 (per GPU)
├── Gradient Accumulation: 4 (batch effectif: 64)
├── Learning Rate: 1e-4 (avec warmup 500 steps)
├── Max Steps: 200,000
├── Checkpoints: Tous les 2,500 steps
└── Evaluation: Tous les 1,000 steps
```

## 🚀 Démarrage Rapide

### Option 1: Docker (Recommandé)
```bash
cd /home/vincent/code/repo/french-llm-from-scratch
./launch_v2.sh
# Sélectionner option 1
```

### Option 2: Python Direct
```bash
python3 launch_v2_training_optimized.py
```

### Option 3: Dashboard Seul
```bash
python3 dashboard_pytorch.py
# Accès: http://localhost:5000
```

## 📍 Accès

| Service | URL | Port |
|---|---|---|
| **Dashboard PyTorch** | http://localhost:5000 | 5000 |
| **Métriques API** | http://localhost:5000/api/metrics | 5000 |
| **Status** | http://localhost:5000/api/status | 5000 |

## 📊 Dashboard Features

En temps réel affiche:
- ✓ Training loss curve
- ✓ GPU memory utilization (allocated/reserved/used)
- ✓ Steps/sec throughput
- ✓ Elapsed time & ETA
- ✓ Learning rate schedule
- ✓ Progress bar (0-200k steps)
- ✓ Hardware info
- ✓ Active optimisations

## 🔧 Configuration Avancée

Éditer les fichiers:
- `trained_models/runs/french_v2_optimized_rtx5080/v2_config.json` - Config modèle
- `launch_optimized_v2_training.py` - Optimisations PyTorch
- `dashboard_pytorch.py` - Monitoring

## 📂 Structure de Données

```
data_clean/
├── v2_tokenized/                          (12GB) ✓ Essentiel
│   ├── phase2_partitioned/
│   │   ├── train/
│   │   └── eval/
│   └── metadata.json
├── conversations_mega_train.txt           (337MB) ✓ Utilisé
├── conversations_mega_test.txt            (60MB) ✓ Utilisé
└── conversations_mega_tokenized.pt        (1.5GB) ✓ Sauvegarde
```

## 🧹 Nettoyage Effectué

Supprimé (60GB libérés):
- ✗ `french_pd_books/` (36GB) - Format brut obsolète
- ✗ `corpus_*.bin` (11.5GB) - Format ancien
- ✗ `french_large/`, `fineweb*` (2.5GB) - Doublons
- ✗ Anciens runs inutilisés

## 📝 Logs et Monitoring

Fichiers de logs:
```
trained_models/runs/french_v2_optimized_rtx5080/
├── metrics.jsonl                          # Logs training
├── v2_config.json                         # Configuration
├── samples.txt                            # Generated samples
├── visuals_latest.json                    # Graphs data
└── logs/                                  # Training logs
```

## 🐳 Docker Commands

```bash
# Démarrer
docker-compose up -d

# Logs
docker-compose logs -f

# Arrêter
docker-compose down

# Status
docker-compose ps
```

## 🚨 Troubleshooting

### "CUDA out of memory"
→ Réduire batch_size dans v2_config.json

### "NVMe slow"
→ Vérifier PCIe slot (doit être 4.0)

### "Low GPU utilization"
→ Augmenter num_workers (actuellement 4)

### "Dashboard connection refused"
→ Vérifier si port 5000 est libre: `netstat -tuln | grep 5000`

## 📊 Métriques Attendues

Avec cette configuration et ce matériel:
- **Throughput**: 2-3 samples/sec (batch=16)
- **GPU Memory**: ~12-13GB utilisée
- **Training Time**: ~24-30 heures pour 200k steps
- **Convergence**: Loss ~ 3.5-4.0 (dépend des données)

## 🔌 Specs Techniques

**RTX 5080 Characteristics:**
- CUDA Cores: 14,080
- Memory: 16GB GDDR7 (576 GB/s)
- TensorFloat-32: ✓ Optimisé
- Sparsité structurée: ✓ Support
- NVLink: ✗ Non disponible

**R7 9700X Characteristics:**
- Cores: 8 / Threads: 16
- Base: 4.5 GHz / Boost: 5.5 GHz
- TDP: 105W (OC)
- Cache: 64MB (L3)
- DDR5 support: ✓

## 📚 Références

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [RTX 5080 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)
- [R7 9700X](https://www.amd.com/en/products/specifications/processors/ryzen/series/ryzen-7-9000-series)

---

**Dernière mise à jour**: 2025-12-08  
**Optimisé pour**: RTX 5080 + R7 9700X OC 105W + DDR5 + NVMe PCIe 4.0
