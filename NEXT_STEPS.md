# 🎯 Prochaines Étapes - French LLM V2

## ✅ Ce qui a été fait (8 déc 2025)

1. **Nettoyage**: 60GB libérés (french_pd_books, corpus*.bin, anciens runs)
2. **Optimisations RTX 5080**: Scripts créés, config PyTorch optimisée
3. **Pipeline**: Système automatisé complet
4. **Documentation**: Plans, guides, rapports

## 📊 État Actuel

### Checkpoints Disponibles
- ✅ `checkpoint_step_100000.pt` (3GB) - Base pré-entraînée
- ✅ `checkpoint_step_17500.pt` (3GB) - Backup

### Datasets Disponibles  
- ✅ `conversations_train_20M_tokenized.pt` (20M tokens, 80MB)
- ✅ `conversations_mega_train_mistral_tokenized.pt` (653MB)
- ✅ `dolly_fr_tokenized.pt` (21MB)

### Scripts Disponibles
- ✅ `scripts/train_subtitles_transformer.py` - Script training principal
- ✅ `scripts/export_to_huggingface.py` - Export HF format
- ✅ `scripts/convert_to_gguf.py` - Conversion GGUF

### Tokenizer
- ✅ `trained_models/tokenizers/vincent_tokenizer_FR_v2/` (SentencePiece 32k)

## 🚀 Prochaines Actions

### Option A: Training Immédiat avec Données Existantes (Recommandé)

```bash
cd /home/vincent/code/repo/french-llm-from-scratch

# Phase 2A: 100k → 110k steps avec conversations_train_20M
.venv/bin/python3 scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --resume-from trained_models/runs/checkpoint_step_100000.pt \
  --pretokenized-path data_clean/conversations_train_20M_tokenized.pt \
  --tokenizer-path trained_models/tokenizers/vincent_tokenizer_FR_v2 \
  --run-name french_v2_phase2a_20M \
  --max-steps 110000 \
  --lr 1e-6 \
  --batch-size 10 \
  --checkpoint-interval 1000 \
  --eval-interval 500 \
  --device cuda \
  --use-amp \
  --sample-interval 500
```

**Durée estimée**: 2-3h GPU  
**Objectif**: Adaptation progressive au format conversationnel

### Option B: Training avec Dataset Complet

```bash
# Phase 2B: 110k → 130k steps avec conversations_mega
.venv/bin/python3 scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --resume-from trained_models/runs/french_v2_phase2a_20M/checkpoint_step_110000.pt \
  --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
  --tokenizer-path trained_models/tokenizers/vincent_tokenizer_FR_v2 \
  --run-name french_v2_phase2b_full \
  --max-steps 130000 \
  --lr 5e-7 \
  --batch-size 10 \
  --checkpoint-interval 2500 \
  --eval-interval 1000 \
  --device cuda \
  --use-amp
```

**Durée estimée**: 4-6h GPU  
**Objectif**: Fine-tuning conversationnel complet

### Option C: Test de Génération Avant Training

```bash
# Tester le checkpoint 100k actuel
.venv/bin/python3 scripts/test_model_samples.py \
  --checkpoint trained_models/runs/checkpoint_step_100000.pt \
  --tokenizer-path trained_models/tokenizers/vincent_tokenizer_FR_v2 \
  --prompts "Bonjour, comment vas-tu ?" \
  --max-length 100
```

## 📈 Validation à Chaque Étape

### Prompts de Test
1. **Simple**: "Bonjour, comment vas-tu ?"
   - Attendu: Réponse polie française courte

2. **Factuel**: "Quelle est la capitale de la France ?"
   - Attendu: "Paris" + contexte optionnel

3. **Explication**: "Explique-moi ce qu'est l'intelligence artificielle."
   - Attendu: Définition cohérente 2-3 phrases français

### Critères d'Acceptation
- ✅ >95% mots français
- ✅ Pas d'artefacts (Ġ, Ċ, ▁)
- ✅ Cohérence (pas de boucles)
- ✅ Pertinence au prompt

## 📦 Export Final (Après Training Réussi)

### 1. Export HuggingFace
```bash
.venv/bin/python3 scripts/export_to_huggingface.py \
  --checkpoint trained_models/runs/french_v2_phase2b_full/checkpoint_step_130000.pt \
  --output-dir trained_models/french_llm_v2_hf \
  --tokenizer-path trained_models/tokenizers/vincent_tokenizer_FR_v2
```

### 2. Conversion GGUF
```bash
.venv/bin/python3 scripts/convert_to_gguf.py \
  --model-dir trained_models/french_llm_v2_hf \
  --output-dir trained_models/french_llm_v2_gguf \
  --quant Q4_K_M,Q5_K_M,Q8_0
```

### 3. Publication
- **HuggingFace**: `Vincent-PRO-AI/french-llm-v2`
- **GitHub**: Release v2.0.0
- **LinkedIn**: Post avec démo

## ⚙️ Configuration Training

### Architecture (Medium)
- Layers: 18
- Heads: 16
- Embedding: 1024
- FFN: 4096
- Vocab: 32000
- Total Params: ~260M

### Hyperparamètres Phase 2A
- LR: 1e-6 (ultra-conservateur)
- Batch: 10
- Gradient Accumulation: 4
- Warmup: 500 steps
- Weight Decay: 0.01

### Hyperparamètres Phase 2B
- LR: 5e-7 (encore plus bas)
- Batch: 10
- Checkpoints: Tous les 2500 steps
- Early stopping: Si régression qualité

## 🎯 Objectif Final

Modèle LLM français:
- ✅ Sans artefacts BPE
- ✅ Conversationnel fonctionnel
- ✅ >95% français pur
- ✅ Publié HF + GitHub
- ✅ Utilisable LM Studio (GGUF)
- ✅ Post LinkedIn avec démo

## 📞 Prochaine Commande à Exécuter

**Démarrer Phase 2A immédiatement**:
```bash
.venv/bin/python3 scripts/train_subtitles_transformer.py \
  --resume-from trained_models/runs/checkpoint_step_100000.pt \
  --pretokenized-path data_clean/conversations_train_20M_tokenized.pt \
  --run-name french_v2_phase2a_20M \
  --max-steps 110000 \
  --lr 1e-6
```

---

**Date**: 8 décembre 2025  
**System**: RTX 5080 16GB | R7 9700X OC 105W | 64GB DDR5
