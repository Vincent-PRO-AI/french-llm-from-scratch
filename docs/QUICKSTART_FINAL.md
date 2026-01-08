# 🚀 Quick Start - French LLM V2 Final

## Objectif
Créer un modèle LLM français **performant** sans artefacts BPE, publié sur HuggingFace/GitHub, utilisable dans LM Studio.

## État Actuel (8 déc 2025)
- ✅ Checkpoint 100k steps (base pré-entraînée Wikipedia + FineWeb)
- ✅ Tokenizer V2 SentencePiece 32k (vincent_tokenizer_FR_v2)
- ✅ Datasets français: dolly_fr, oasst2_fr
- ✅ Infrastructure RTX 5080 optimisée

## 3 Options de Démarrage

### Option A: Pipeline Automatique (Recommandé)
```bash
cd /home/vincent/code/repo/french-llm-from-scratch

# Préparation uniquement (pas de training)
python3 launch_full_v2_pipeline.py --skip-training

# OU Pipeline complet avec training (6-8h GPU)
python3 launch_full_v2_pipeline.py
```

### Option B: Étape par Étape
```bash
# 1. Tester tokenizer (5 min)
.venv/bin/python3 test_tokenizer_v2.py

# 2. Créer dataset validé FR (15 min)
.venv/bin/python3 scripts/create_validated_conversations_fr.py \
  --sources oasst2,dolly_fr \
  --output data_clean/conversations_validated_fr.pt

# 3. Phase 2A: Instruction tuning (1-2h)
.venv/bin/python3 scripts/v2/train_phase2_conversations.py \
  --resume-from trained_models/runs/checkpoint_step_100000.pt \
  --dataset-path data_clean/conversations_validated_fr.pt \
  --max-steps 105000 --lr 1e-6

# 4. Phase 2B: Full instruction (3-4h)
.venv/bin/python3 scripts/v2/train_phase2_conversations.py \
  --resume-from trained_models/runs/french_v2_phase2a/checkpoint_step_105000.pt \
  --max-steps 120000 --lr 5e-7

# 5. Export HuggingFace (10 min)
.venv/bin/python3 scripts/export_to_huggingface.py \
  --checkpoint trained_models/runs/french_v2_phase2b/checkpoint_step_120000.pt \
  --output-dir trained_models/french_llm_v2_hf

# 6. Convert GGUF (20 min)
.venv/bin/python3 scripts/convert_to_gguf_mixed.py \
  --model-dir trained_models/french_llm_v2_hf \
  --quantization Q4_K_M,Q5_K_M,Q8_0
```

### Option C: Docker
```bash
docker-compose up -d
# Dashboard: http://localhost:5174 (frontend)
# API: http://localhost:8000 (backend)
```

## Critères de Succès

✅ **Tokenizer**: Pas d'artefacts Ġ, Ċ, ▁ visibles  
✅ **Qualité**: >95% mots français, cohérence, pas de boucles  
✅ **Déploiement**: HF Hub + GitHub + LM Studio ready  

## Timeline

| Étape | Durée | GPU |
|-------|-------|-----|
| Préparation (Steps 1-3) | 1h | Non |
| Phase 2A (100k→105k) | 1-2h | Oui |
| Phase 2B (105k→120k) | 3-4h | Oui |
| Export HF + GGUF | 30min | Non |
| Publication | 1-2h | Non |
| **TOTAL** | **6-10h** | 4-6h GPU |

## Fichiers Générés

```
trained_models/
├── french_llm_v2_hf/          # Format HuggingFace
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── french_llm_v2_gguf/        # Format LM Studio
    ├── french_llm_v2_q4_k_m.gguf  (~2GB)
    ├── french_llm_v2_q5_k_m.gguf  (~2.5GB)
    └── french_llm_v2_q8_0.gguf    (~3.5GB)
```

## Publication

### HuggingFace
- Repo: `Vincent-PRO-AI/french-llm-v2`
- Upload: HF format + GGUF files
- Model card avec benchmarks

### GitHub
- Push sur: `Vincent-PRO-AI/french-llm-from-scratch`
- Release: v2.0.0
- README reproduction exacte

### LinkedIn
- Post inspiré v1
- Hashtags: #NLP #AI #MachineLearning #French #OpenSource

## Références

- Plan complet: `PLAN_ACTION_FINAL_V2.py`
- Rapport historique: `RAPPORT_COMPLET.md`
- Config RTX 5080: `README_V2_OPTIMIZED.md`

---

**Prochaine action**: `python3 launch_full_v2_pipeline.py --skip-training`
