# 🚀 Optimisations Performance Training

## État Actuel
- **GPU**: 1x RTX 5080 (16GB VRAM)
- **Batch size**: 4 (effectif 12 avec accumulation)
- **Précision**: Mixed Precision FP16/FP32 (AMP)
- **Workers**: Défaut (probablement 0-2)

## ⚡ Optimisations Disponibles

### 1. **Augmenter l'Utilisation VRAM**

**Actuellement**: ~2-4GB utilisés sur 16GB disponibles

**Actions**:
```python
--batch-size 8        # Au lieu de 4 (double la VRAM)
--gradient-accumulation-steps 2  # Batch effectif = 16
```

**Gain estimé**: 1.5-2x vitesse d'entraînement

### 2. **Augmenter l'Utilisation RAM pour DataLoader**

**Actuellement**: Probablement 0-2 workers

**Actions**:
```python
--num-workers 8       # Utilise 8 cores CPU pour charger données
--pin-memory          # Accélère transferts CPU→GPU
--prefetch-factor 4   # Précharge 4 batches en RAM
```

**Gain estimé**: Élimine les attentes I/O, GPU toujours occupé

### 3. **Multi-GPU (si 2ème GPU disponible)**

**Méthode 1: DataParallel** (simple mais moins efficace)
```bash
python train.py --data-parallel
```

**Méthode 2: DistributedDataParallel** (recommandé)
```bash
torchrun --nproc_per_node=2 train.py
```

**Gain estimé**: 1.7x vitesse avec 2 GPUs (0.85x par GPU)

### 4. **Précision FP8 (Hopper/Ada GPUs)**

**Requis**: 
- GPU RTX 40xx/50xx ou H100
- `transformer_engine` de NVIDIA

**Installation**:
```bash
pip install transformer-engine[pytorch]
```

**Gain estimé**: 
- 2x vitesse calcul
- 50% réduction VRAM
- **ATTENTION**: Peut affecter stabilité training

### 5. **Optimisations Supplémentaires**

```python
# Dans train_subtitles_transformer.py
torch.backends.cudnn.benchmark = True  # Optimise convolutions
torch.set_float32_matmul_precision('medium')  # TensorCores TF32
```

## 📊 Configuration Recommandée (1 GPU)

```bash
python scripts/train_subtitles_transformer.py \
    --resume-from checkpoint_step_160000.pt \
    --max-steps 220000 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 3e-5 \
    --num-workers 8 \
    --pin-memory \
    --prefetch-factor 4 \
    --gradient-checkpointing \
    --use-amp \
    --dataset-path data_clean/french_large_filtered_mistral_tokenized.pt
```

**Batch effectif**: 8 × 2 = 16 (au lieu de 12)
**VRAM**: ~8-10GB (reste de la marge)
**Speedup total**: **~2-2.5x** vs config actuelle

## 📊 Configuration Multi-GPU (si disponible)

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_subtitles_transformer.py \
    --resume-from checkpoint_step_160000.pt \
    --max-steps 220000 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 3e-5 \
    --num-workers 8 \
    --pin-memory \
    --dataset-path data_clean/french_large_filtered_mistral_tokenized.pt
```

**Batch effectif**: 2 GPUs × 8 × 2 = 32
**Speedup total**: **~3.5-4x** vs config actuelle

## ⚠️ Problème Principal: Dataset

**Les optimisations de vitesse ne règlent pas le problème de qualité!**

Le dataset `french_large_filtered` contient:
- Markup HTML: `&lt;`, `&amp;`, `&gt;`
- Références bibliographiques
- Numéros de catalogues

**Solution requise**: Filtrer davantage le dataset
- Supprimer les lignes avec HTML entities
- Exclure les fichiers avec >5% de nombres
- Privilégier texte narratif pur

## 🎯 Recommandation

1. **Court terme**: Utiliser `phase2_mixed_fr_mistral_tokenized.pt` (14M tokens, 100% propre)
2. **Moyen terme**: Créer `french_large_narrative_only.pt` (filtrage avancé)
3. **Optimiser ensuite**: Appliquer optimisations VRAM/RAM/Multi-GPU
