# 📦 Inventaire Datasets - French LLM V2

## 🎯 Datasets Actuellement Tokenisés (data_clean/)

| Fichier | Taille | Tokens | Status |
|---------|--------|--------|--------|
| conversations_mega_train_mistral_tokenized.pt | 653MB | ~169M | ✅ Prêt V2 |
| conversations_mega_tokenized.pt | 1.5GB | ~390M | ⚠️ Ancien tokenizer |
| conversations_train_20M_tokenized.pt | 653MB | 20M | ✅ Prêt V2 |
| conversations_combined_tokenized.pt | 115MB | ~30M | ⚠️ Ancien |
| conversations_all_tokenized.pt | 49MB | ~12M | ⚠️ Ancien |
| conversations_extra_tokenized.pt | 67MB | ~17M | ⚠️ Ancien |
| dolly_fr_tokenized.pt | 21MB | ~5M | ✅ Prêt V2 |
| conversations_mega_test_mistral_tokenized.pt | 116MB | ~30M | ✅ Test V2 |

**Total Tokenisé V2**: ~204M tokens (834MB)

## 📁 Datasets Bruts Disponibles (Backup /mnt/g)

### 1. Conversations Massives (/mnt/g/backupllmfromscratch/data_clean/conversations_massive/)
- **ultrachat_fr.txt**: 167MB - Conversations françaises UltraChat
- **ultrachat_extended_part_*.txt**: 9060 fichiers × 2-3MB = ~20GB - Dataset massif
- **dolly_fr.txt**: Format conversations Dolly français
- **oasst2_fr.txt**: OpenAssistant conversations françaises
- **Total**: ~843MB direct + 20GB partitionné

### 2. Conversations Combinées
- **conversations_mega_train.txt**: 337MB - Dataset principal
- **conversations_mega_test.txt**: 60MB - Split test
- **Total**: 397MB

### 3. Autres Sources
- **wikipedia_fr_combined.txt**: 19MB - Articles Wikipedia français
- **conversations_hf/conversations_hf_fr.txt**: Conversations HuggingFace
- **conversations_all/conversations_all_fr.txt**: Mix conversations
- **conversations_extra/conversations_extra_fr.txt**: Conversations supplémentaires

## 🚀 Plan de Tokenisation

### Phase 1: Tokenisation Datasets Prioritaires (MAINTENANT)

```bash
# 1. UltraChat FR (167MB) - Conversations qualité
python scripts/tokenize_dataset_v2.py \
  --input /mnt/g/backupllmfromscratch/data_clean/conversations_massive/ultrachat_fr.txt \
  --output data_clean/ultrachat_fr_v2_tokenized.pt \
  --tokenizer trained_models/tokenizers/vincent_tokenizer_FR_v2 \
  --format txt

# 2. OASST2 FR - Conversations assistant
python scripts/tokenize_dataset_v2.py \
  --input /mnt/g/backupllmfromscratch/data_clean/conversations_massive/oasst2_fr.txt \
  --output data_clean/oasst2_fr_v2_tokenized.pt \
  --tokenizer trained_models/tokenizers/vincent_tokenizer_FR_v2 \
  --format txt

# 3. Wikipedia FR (19MB) - Connaissances
python scripts/tokenize_dataset_v2.py \
  --input /mnt/g/backupllmfromscratch/data_clean/wikipedia_fr_combined.txt \
  --output data_clean/wikipedia_fr_v2_tokenized.pt \
  --tokenizer trained_models/tokenizers/vincent_tokenizer_FR_v2 \
  --format txt
```

**Estimation**: ~50M tokens supplémentaires = ~200MB tokenisés

### Phase 2: UltraChat Extended (Optionnel - 20GB)

Si besoin de plus de données :
```bash
# Combine tous les parts UltraChat Extended
cat /mnt/g/backupllmfromscratch/data_clean/conversations_massive/ultrachat_extended_part_*.txt > \
    /tmp/ultrachat_extended_full.txt

# Tokenise le dataset complet
python scripts/tokenize_dataset_v2.py \
  --input /tmp/ultrachat_extended_full.txt \
  --output data_clean/ultrachat_extended_v2_tokenized.pt \
  --tokenizer trained_models/tokenizers/vincent_tokenizer_FR_v2 \
  --format txt \
  --chunk-size 100000
```

**Estimation**: ~5-6 milliards tokens = ~20GB tokenisés

## 📊 Stratégie Optimale

### Option A: Training Rapide (Recommandé)
**Datasets**: conversations_mega_train_mistral (653MB) + dolly_fr (21MB)  
**Total**: ~174M tokens  
**Durée Phase 2A**: 2-3h GPU  
**Avantage**: Déjà tokenisés, training immédiat

### Option B: Training avec UltraChat
**Datasets**: mega (653MB) + ultrachat_fr (167MB brut → ~50MB tokenisé) + dolly (21MB)  
**Total**: ~224M tokens  
**Durée Phase 2A**: 3-4h GPU  
**Avantage**: Plus de diversité conversationnelle

### Option C: Training Massif (Si performances insuffisantes)
**Datasets**: Tous les datasets + UltraChat Extended (20GB)  
**Total**: ~6 milliards tokens  
**Durée Phase 2**: 50-100h GPU  
**Avantage**: Modèle ultra-performant mais long

## ✅ Recommandation Immédiate

**Lancer Option A maintenant** avec les datasets déjà tokenisés :
- Phase 2A: 100k → 110k steps (2-3h)
- Validation qualité
- Si OK → Phase 2B: 110k → 130k steps
- Si besoin plus → Tokeniser UltraChat FR (Option B)

**Commande**:
```bash
.venv/bin/python3 scripts/train_subtitles_transformer.py \
  --resume-from trained_models/runs/checkpoint_step_100000.pt \
  --pretokenized-path data_clean/conversations_mega_train_mistral_tokenized.pt \
  --run-name french_v2_phase2a_mega \
  --max-steps 110000 \
  --lr 1e-6 \
  --use-amp
```
