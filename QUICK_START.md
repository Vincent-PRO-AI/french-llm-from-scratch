# 🚀 French LLM Training - Status & Quick Start

> Un modèle de langage français entraîné 100% localement sur RTX 5080

## ✅ STATUS ACTUEL

**Entraînement en cours**: 160k → 500k steps (340k steps restants)
- **GPU**: RTX 5080 (16 GB) ✅
- **RAM**: 96 GB DDR5 ✅
- **Durée estimée**: 15-20 heures
- **Date fin**: 21 Décembre 2025

```
Progression: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 32%
```

---

## 🎯 Commandes Utiles

### 1. Suivre la progression en temps réel
```bash
# Option A: Dashboard simple
python3 dashboard.py

# Option B: Voir les logs d'entraînement
tail -f training_log.txt

# Option C: Vérifier les métriques JSON
tail trained_models/runs/french_medium_rtx5080_extended/metrics.jsonl
```

### 2. Exporter en GGUF (après entraînement)
```bash
# Export pour LM Studio (quantisé Q4_K_M)
python3 quick_export_gguf.py
# Output: trained_models/exports/french_medium_500k.gguf (~2 GB)
```

### 3. Publier sur HuggingFace
```bash
# Configuration (une seule fois)
huggingface-cli login

# Publication
python3 publish_huggingface.py
# Repo: Vincent-PRO-AI/french-llm-500k
```

### 4. Pipeline complet (automatique)
```bash
bash full_pipeline.sh
# Lance: entraînement → export → publication
```

---

## 📊 Configuration Détaillée

### Architecture Modèle
| Aspect | Valeur |
|--------|--------|
| **Type** | Transformer |
| **Layers** | 18 |
| **Heads** | 16 |
| **Embed** | 1024 |
| **FF** | 4096 |
| **Paramètres** | ~260M |
| **Vocab** | 32k (Mistral) |

### Optimisations Appliquées
- ✅ Mixed Precision (AMP)
- ✅ Gradient Checkpointing (économise 30% VRAM)
- ✅ NVMe Cache (spillover)
- ✅ Batch size auto-adapted (12)
- ✅ Learning rate: 1.5e-4

---

## 💻 Utilisation Après Entraînement

### Via LM Studio (GUI simple)
1. Télécharger: `french_medium_500k.gguf`
2. Ouvrir LM Studio
3. Load Model → Charger le GGUF
4. Chat!

### Via Python (Transformers)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Vincent-PRO-AI/french-llm-500k")
model = AutoModelForCausalLM.from_pretrained(
    "Vincent-PRO-AI/french-llm-500k",
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Bonjour, comment peux-tu m'aider ? "
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.9)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Via llama.cpp (CPU optimisé)
```bash
# Convertir GGUF en ggml si nécessaire
./llama-cpp-bin/main -m french_medium_500k.gguf \
  -n 100 \
  -t 8 \
  -p "Bonjour, comment peux-tu m'aider ? "
```

---

## 📁 Fichiers Importants

```
french-llm-from-scratch/
├── start_training.sh                    # Relancer l'entraînement
├── dashboard.py                         # Monitoring en temps réel
├── quick_export_gguf.py                 # Export GGUF
├── publish_huggingface.py               # Publication HF
├── full_pipeline.sh                     # Automation complète
├── TRAINING_STATUS.txt                  # Ce rapport
│
├── trained_models/runs/
│   └── french_medium_rtx5080_extended/
│       ├── checkpoint_step_5000.pt
│       ├── checkpoint_step_10000.pt
│       └── ...checkpoint_step_500000.pt (final)
│
├── trained_models/exports/
│   └── french_medium_500k.gguf          (généré après)
│
├── data_clean/
│   ├── mistral_tokenizer/
│   ├── conversations_mega_train_mistral_tokenized.pt
│   └── conversations_mega_test_mistral_tokenized.pt
│
└── scripts/
    ├── train_subtitles_transformer.py   (script principal)
    ├── export_to_gguf.py                (export)
    └── ...
```

---

## 🔄 Si l'entraînement s'arrête

```bash
# Relancer depuis le dernier checkpoint
bash start_training.sh
# OU
python3 scripts/train_subtitles_transformer.py \
  --resume-from trained_models/runs/french_medium_rtx5080_extended/checkpoint_step_XXXXX.pt \
  --arch-preset medium \
  --max-steps 500000 \
  ... (autres paramètres)
```

---

## 📈 Performance Attendue

### Entraînement
- **Vitesse**: ~3.5 steps/sec
- **Temps/step**: ~285ms
- **Throughput**: ~8.5K tokens/sec

### Inférence (RTX 5080)
- **Prefill**: 50-100 ms/token
- **Decode**: 100-150 ms/token
- **Memory**: ~8 GB

### Quantisé GGUF (Q4_K_M)
- **Taille**: ~2 GB
- **Memory**: ~2.5 GB
- **Vitesse CPU**: 20-50 ms/token

---

## 🆘 Troubleshooting

### Entraînement très lent?
```bash
# Vérifier GPU
nvidia-smi

# Vérifier batch size dans les logs
grep "Batch size" training_log.txt

# Vérifier vitesse steps
grep "step" trained_models/runs/french_medium_rtx5080_extended/metrics.jsonl | tail
```

### Erreur CUDA Memory?
- Entraînement adapte automatiquement batch size
- Vérifier: `grep "Adapted config" training_log.txt`
- Restart si nécessaire

### Export GGUF échoue?
```bash
# Vérifier le checkpoint existe
ls -lh trained_models/runs/french_medium_rtx5080_extended/checkpoint_step_500000.pt

# Essayer export manuel
python3 export_to_gguf.py \
  trained_models/runs/french_medium_rtx5080_extended/checkpoint_step_500000.pt \
  trained_models/exports/french_medium_500k.gguf
```

---

## 📞 Support & Documentation

- **Entraînement**: `scripts/train_subtitles_transformer.py --help`
- **Export**: `python3 quick_export_gguf.py --help`
- **Publication**: `python3 publish_huggingface.py --help`

---

## 🎉 Résumé

✅ **Matériel optimisé** (RTX 5080 + 96GB RAM)
✅ **Entraînement actif** (160k → 500k steps)
✅ **Scripts prêts** (export, publication, monitoring)
✅ **Documentation complète** (ce fichier)
✅ **Prêt pour LM Studio & HuggingFace**

---

**Date**: 20 Décembre 2025  
**Durée estimée**: 15-20 heures  
**Fin estimée**: 21 Décembre 2025 (après-midi/soirée)

**Bon entraînement! 🚀**
