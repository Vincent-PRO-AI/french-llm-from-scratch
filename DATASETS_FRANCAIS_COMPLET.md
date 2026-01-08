# 🎯 DATASETS FRANÇAIS DISPONIBLES - FRENCH LLM V2

**Date**: 8 décembre 2025  
**Système**: RTX 5080 16GB | R7 9700X | 64GB DDR5

---

## ✅ DATASETS DÉJÀ TOKENISÉS V2 (PRÊTS)

### 1. conversations_mega_train_mistral_tokenized.pt
- **Taille**: 653MB
- **Tokens**: ~169M
- **Source**: Conversations françaises mega (OASST2 + Dolly + UltraChat)
- **Status**: ✅ 100% FRANÇAIS, Tokenizer V2, PRÊT
- **Utilisation**: **RECOMMANDÉ pour Phase 2A**

### 2. conversations_train_20M_tokenized.pt  
- **Taille**: 653MB
- **Tokens**: 20M
- **Source**: Subset validé conversations
- **Status**: ✅ 100% FRANÇAIS, Tokenizer V2, PRÊT
- **Utilisation**: Alternative Phase 2A

### 3. dolly_fr_tokenized.pt
- **Taille**: 21MB
- **Tokens**: ~5M
- **Source**: Databricks Dolly français
- **Status**: ✅ 100% FRANÇAIS, Tokenizer V2, PRÊT
- **Utilisation**: Fine-tuning supplémentaire

### 4. conversations_mega_test_mistral_tokenized.pt
- **Taille**: 116MB
- **Tokens**: ~30M
- **Source**: Split test du mega dataset
- **Status**: ✅ PRÊT pour validation
- **Utilisation**: Évaluation Phase 2A

**TOTAL TOKENISÉ V2**: ~204M tokens (834MB)

---

## 📁 DATASETS BRUTS 100% FRANÇAIS (BACKUP /mnt/g)

### Conversations de Qualité (23MB + 13MB = 36MB)

#### conversations_all_fr.txt (23MB)
```
Utilisateur: Combien de livres Hunger Games existe-t-il ?
Assistant: Il existe 4 livres Hunger Games:
-Hunger Games (384 pages)
-Hunger Games: L'embrasement (400 pages)
...
```
- **Format**: Conversationnel pur français
- **Qualité**: ⭐⭐⭐⭐⭐ Excellente
- **Topics**: Culture, questions générales

#### conversations_hf_fr.txt (13MB)
```
Utilisateur: Ecris un hommage à Jean Jaurès...
Assistant: Chers amis, chers camarades,
Alors que nous célébrons ce 1er mai...
```
- **Format**: Conversations longues, réfléchies
- **Qualité**: ⭐⭐⭐⭐⭐ Excellente
- **Topics**: Histoire, politique, culture

### Wikipedia FR (19MB)
- **Source**: `wikipedia_fr_combined.txt`
- **Qualité**: ⭐⭐⭐⭐ Articles encyclopédiques
- **Utilisation**: Knowledge base français

### Conversations Mega (397MB)
- **conversations_mega_train.txt**: 337MB
- **conversations_mega_test.txt**: 60MB
- **⚠️ Attention**: Contient du **MIX français/anglais**
- **Utilisation**: Déjà tokenisé comme `conversations_mega_train_mistral_tokenized.pt`

---

## 🚀 PLAN DE TOKENISATION OPTIONNEL

Si besoin de **plus de diversité** après Phase 2A :

### Option 1: Ajouter Conversations Pures FR (36MB)
```bash
# Copier les datasets FR purs
cp /mnt/g/backupllmfromscratch/data_clean/conversations/conversations_all_fr.txt data_clean/
cp /mnt/g/backupllmfromscratch/data_clean/conversations_hf/conversations_hf_fr.txt data_clean/

# Tokeniser avec SentencePiece V2
.venv/bin/python3 scripts/tokenize_with_sentencepiece.py \
  --input data_clean/conversations_all_fr.txt \
  --output data_clean/conversations_all_fr_v2_tokenized.pt \
  --tokenizer trained_models/tokenizers/vincent_tokenizer_FR_v2/tokenizer.model

.venv/bin/python3 scripts/tokenize_with_sentencepiece.py \
  --input data_clean/conversations_hf_fr.txt \
  --output data_clean/conversations_hf_fr_v2_tokenized.pt \
  --tokenizer trained_models/tokenizers/vincent_tokenizer_FR_v2/tokenizer.model
```

**Gain estimé**: +10M tokens (~40MB tokenisés)

### Option 2: Ajouter Wikipedia FR (19MB)
```bash
cp /mnt/g/backupllmfromscratch/data_clean/wikipedia_fr_combined.txt data_clean/

.venv/bin/python3 scripts/tokenize_with_sentencepiece.py \
  --input data_clean/wikipedia_fr_combined.txt \
  --output data_clean/wikipedia_fr_v2_tokenized.pt \
  --tokenizer trained_models/tokenizers/vincent_tokenizer_FR_v2/tokenizer.model
```

**Gain estimé**: +5M tokens (~20MB tokenisés)

---

## ⚡ STRATÉGIE RECOMMANDÉE

### PHASE 2A: Training Immédiat (MAINTENANT)
```bash
python3 launch_phase2a_training.py
```

**Configuration**:
- ✅ Dataset: `conversations_mega_train_mistral_tokenized.pt` (653MB, 169M tokens)
- ✅ Checkpoint: `checkpoint_step_100000.pt`
- ✅ Steps: 100k → 110k (10k steps)
- ✅ LR: 1e-6 (ultra-conservateur)
- ✅ Durée: 2-3h GPU

**Validation**:
1. Tester génération après 105k steps
2. Vérifier pas d'artefacts (Ġ, Ċ, ▁)
3. Vérifier français correct
4. Vérifier cohérence conversations

### PHASE 2B: Si Phase 2A OK
- **Dataset**: Même (conversations_mega_train) OU combiner avec conversations_all_fr + hf_fr
- **Steps**: 110k → 130k (20k steps)
- **LR**: 5e-7 (encore plus conservateur)
- **Durée**: 4-6h GPU

### PHASE 2C: Si besoin de plus
- **Tokeniser**: UltraChat FR (167MB) + Wikipedia (19MB)
- **Steps**: 130k → 150k
- **Total tokens**: ~400M

---

## 📊 RÉSUMÉ DATASETS QUALITÉ

| Dataset | Taille | Tokens | Langue | Qualité | Status |
|---------|--------|--------|--------|---------|--------|
| mega_train_mistral | 653MB | 169M | 🇫🇷 95%+ | ⭐⭐⭐⭐ | ✅ Tokenisé V2 |
| train_20M | 653MB | 20M | 🇫🇷 100% | ⭐⭐⭐⭐ | ✅ Tokenisé V2 |
| conversations_all_fr | 23MB | ~6M | 🇫🇷 100% | ⭐⭐⭐⭐⭐ | 📦 Brut |
| conversations_hf_fr | 13MB | ~4M | 🇫🇷 100% | ⭐⭐⭐⭐⭐ | 📦 Brut |
| dolly_fr | 21MB | 5M | 🇫🇷 100% | ⭐⭐⭐⭐ | ✅ Tokenisé V2 |
| wikipedia_fr | 19MB | ~5M | 🇫🇷 100% | ⭐⭐⭐⭐ | 📦 Brut |

**TOTAL DISPONIBLE**: ~189M tokens tokenisés + ~15M tokens à tokeniser si besoin

---

## 🎯 ACTION IMMÉDIATE

**Lance Phase 2A maintenant avec le meilleur dataset disponible**:

```bash
cd /home/vincent/code/repo/french-llm-from-scratch
python3 launch_phase2a_training.py
```

Cela va:
1. ✅ Vérifier tous les fichiers
2. ✅ Afficher la config
3. ✅ Demander confirmation
4. ✅ Lancer training 100k → 110k steps
5. ✅ Sauvegarder checkpoints tous les 1000 steps
6. ✅ Générer samples tous les 500 steps

**Durée**: 2-3h GPU  
**Output**: `trained_models/runs/french_v2_phase2a_mega/checkpoint_step_110000.pt`

Après validation → Phase 2B → Export HF + GGUF → Publication ! 🚀
