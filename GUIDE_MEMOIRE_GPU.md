# 🎓 Guide MLOps : Comprendre l'utilisation mémoire GPU pour le training

## 📊 Concepts fondamentaux

### 1. Capacité vs Utilisation

**Vos GPUs :**
- **RTX 5080** : 16 GB de VRAM totale
- **RTX 4090** : 24 GB de VRAM totale

**⚠️ IMPORTANT** : La capacité totale ≠ mémoire utilisée par le training

### 2. Composants qui occupent la mémoire GPU

```
┌─────────────────────────────────────────────┐
│         MÉMOIRE GPU PENDANT TRAINING        │
├─────────────────────────────────────────────┤
│ 1. Poids du modèle (FP16)      ~0.55 GB    │
│ 2. Gradients (FP16)             ~0.55 GB    │
│ 3. Optimizer states (Adam)      ~2.20 GB    │
│    - Momentum (FP32)            ~1.10 GB    │
│    - Variance (FP32)            ~1.10 GB    │
│ 4. Activations (forward pass)   ~3-4 GB    │
│    - Dépend du batch size                   │
│    - Dépend de la séquence length           │
├─────────────────────────────────────────────┤
│ TOTAL (batch_size=8)           ~6-7 GB      │
│ TOTAL (batch_size=12)          ~8-9 GB      │
│ TOTAL (batch_size=16)          ~10-11 GB    │
└─────────────────────────────────────────────┘
```

## 🔬 Détail des calculs

### Votre modèle : 292M paramètres

```python
# 1. Poids du modèle
Paramètres = 292,270,080
FP32 (4 bytes) = 292M × 4 = 1.17 GB
FP16 (2 bytes) = 292M × 2 = 0.58 GB  ✅ Mixed precision

# 2. Gradients (même taille que les poids)
Gradients FP16 = 0.58 GB

# 3. Optimizer Adam (2 states en FP32)
Momentum FP32 = 292M × 4 = 1.17 GB
Variance FP32 = 292M × 4 = 1.17 GB
Total optimizer = 2.34 GB

# 4. Activations (dépend du batch)
batch_size=8  → ~3 GB
batch_size=12 → ~4 GB
batch_size=16 → ~5 GB

# TOTAL
batch_size=8  : 0.58 + 0.58 + 2.34 + 3 = 6.5 GB
batch_size=12 : 0.58 + 0.58 + 2.34 + 4 = 7.5 GB
batch_size=16 : 0.58 + 0.58 + 2.34 + 5 = 8.5 GB
```

## 🎮 Mode DDP (Distributed Data Parallel)

### Principe clé : Réplication du modèle

```
┌─────────────────────────────────────────────────┐
│              Mode Single GPU                    │
├─────────────────────────────────────────────────┤
│ GPU 0 (RTX 4090 - 24GB)                         │
│ ├─ Modèle complet (292M params)                 │
│ ├─ Gradients complets                           │
│ ├─ Optimizer states complets                    │
│ └─ Activations (batch_size=16)                  │
│                                                  │
│ Utilisation: ~8.5 GB / 24 GB (35%)              │
│ GPU 1 : INUTILISÉ                               │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              Mode DDP (2 GPUs)                  │
├─────────────────────────────────────────────────┤
│ GPU 0 (RTX 5080 - 16GB)                         │
│ ├─ Modèle COMPLET (292M params)                 │
│ ├─ Gradients COMPLETS                           │
│ ├─ Optimizer states COMPLETS                    │
│ └─ Activations (batch_size=8)                   │
│ Utilisation: ~6.5 GB / 16 GB                    │
│                                                  │
│ GPU 1 (RTX 4090 - 24GB)                         │
│ ├─ Modèle COMPLET (292M params)                 │
│ ├─ Gradients COMPLETS                           │
│ ├─ Optimizer states COMPLETS                    │
│ └─ Activations (batch_size=10)                  │
│ Utilisation: ~7.5 GB / 24 GB                    │
│                                                  │
│ Batch effectif total: 8 + 10 = 18               │
│ Vitesse: 1.8× plus rapide qu'un seul GPU       │
└─────────────────────────────────────────────────┘
```

### ⚠️ Erreur commune à éviter

```python
# ❌ FAUX : "DDP divise le modèle entre GPUs"
# Chaque GPU n'a PAS que 146M paramètres

# ✅ VRAI : "DDP réplique le modèle sur chaque GPU"
# Chaque GPU a les 292M paramètres complets

# La parallélisation se fait sur les DONNÉES, pas le MODÈLE
# - GPU 0 traite batch[0:8]
# - GPU 1 traite batch[8:18]
# - Les gradients sont moyennés entre GPUs
```

## 📈 Comprendre la répartition mémoire

### Pourquoi "8GB" dans mon tableau précédent ?

C'était un **exemple simplifié** du pic d'utilisation pour un batch_size donné, pas la capacité totale du GPU.

### Allocation dynamique réelle

```python
# La mémoire GPU s'alloue progressivement :

Démarrage :          ~0.5 GB (CUDA context)
Chargement modèle :  +0.6 GB → 1.1 GB
Optimizer init :     +2.3 GB → 3.4 GB
Premier forward :    +3.0 GB → 6.4 GB ✅ Stable

# La mémoire reste à ~6-7 GB pendant tout le training
# Elle ne remplit PAS tout le GPU (16GB ou 24GB)
```

## 🎯 Configuration optimale pour vos GPUs

### Single GPU (baseline)

```bash
# RTX 4090 seul
python train_subtitles_transformer.py \
    --batch-size 16 \
    --gradient-accumulation-steps 2
    
# Utilisation: ~8.5 GB / 24 GB
# Batch effectif: 16 × 2 = 32
# Vitesse: 1.0× (référence)
```

### DDP 2 GPUs (recommandé)

```bash
# DDP automatique
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train_subtitles_transformer_ddp.py \
    --batch-size 10 \  # Par GPU
    --gradient-accumulation-steps 2
    
# GPU 0 (RTX 5080): batch=8  → ~6.5 GB / 16 GB
# GPU 1 (RTX 4090): batch=10 → ~7.5 GB / 24 GB
# Batch effectif total: (8+10) × 2 = 36
# Vitesse: 1.7-1.9× plus rapide
```

## 💡 Concepts MLOps avancés

### 1. Gradient Accumulation

```python
# Simule un grand batch sans utiliser plus de mémoire
batch_size = 8              # 8 échantillons en mémoire
gradient_accumulation = 4   # Accumule sur 4 itérations
# Batch effectif = 8 × 4 = 32 (équivalent batch_size=32)

# Utilisation mémoire: celle de batch_size=8
# Comportement training: celui de batch_size=32
```

### 2. Mixed Precision (FP16)

```python
# Stocke les poids en FP16 (2 bytes) au lieu de FP32 (4 bytes)
# → Divise la mémoire par 2
# → Accélère les calculs GPU (Tensor Cores)

Modèle FP32: 1.17 GB
Modèle FP16: 0.58 GB ✅ Économie 50%
```

### 3. Gradient Checkpointing

```python
# Trade-off : mémoire vs temps
# Recalcule les activations au backward au lieu de les stocker

Sans checkpointing: 4 GB activations → rapide
Avec checkpointing: 1 GB activations → 20% plus lent

# Utile si vous manquez de VRAM
```

## 🔍 Monitoring en temps réel

### Commandes essentielles

```bash
# Voir utilisation GPU en direct
nvidia-smi dmon -s pucvmet

# Voir allocations mémoire détaillées
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv -l 1

# Dans Python pendant training
import torch
print(f"Alloué: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Réservé: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Exemple output pendant training

```
GPU 0 (RTX 5080):
├─ Capacité totale:  16.0 GB
├─ Mémoire allouée:   6.8 GB (43%)  ← Training actif
├─ Mémoire libre:     9.2 GB (57%)  ← Marge disponible
└─ Utilisation GPU:   95%           ← GPU compute utilisé

GPU 1 (RTX 4090):
├─ Capacité totale:  24.0 GB
├─ Mémoire allouée:   7.3 GB (30%)  ← Training actif
├─ Mémoire libre:    16.7 GB (70%)  ← Beaucoup de marge
└─ Utilisation GPU:   98%           ← GPU compute utilisé
```

## 📚 Résumé pour MLOps

### Points clés à retenir

1. **Capacité ≠ Utilisation**
   - RTX 5080 : 16GB total, mais training utilise ~6-7GB
   - RTX 4090 : 24GB total, mais training utilise ~7-9GB

2. **Composants mémoire**
   - Modèle : ~0.6 GB (FP16)
   - Gradients : ~0.6 GB (FP16)
   - Optimizer : ~2.3 GB (FP32)
   - Activations : ~3-4 GB (batch dependent)

3. **DDP = Réplication, pas division**
   - Chaque GPU a une copie complète du modèle
   - Parallélisation sur les données (batches différents)
   - Synchronisation des gradients après chaque step

4. **Optimisations disponibles**
   - Mixed Precision (FP16) → -50% mémoire
   - Gradient Accumulation → simule grand batch
   - Gradient Checkpointing → -70% activations

5. **Monitoring essentiel**
   - `nvidia-smi` pour voir l'utilisation réelle
   - `torch.cuda.memory_allocated()` pour debug
   - TensorBoard pour métriques training

## 🚀 Configuration Docker Multi-GPU prête

Votre fichier `docker-compose.multi-gpu.yml` est maintenant configuré avec :

- **RTX 5080** : batch_size=8 → ~6.5 GB utilisés
- **RTX 4090** : batch_size=10 → ~7.5 GB utilisés
- **Batch effectif** : (8+10) × 2 accumulation = 36
- **Vitesse estimée** : 1.8× plus rapide que single GPU

### Lancement

```bash
cd /home/vincent/code/repo/french-llm-from-scratch
docker-compose --profile multi-gpu up
```

Voilà ! Vous comprenez maintenant comment la mémoire GPU est réellement utilisée pendant le training. 🎓
