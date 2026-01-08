# 📦 Legacy V1 - GPT-2 Architecture (Archived)

**Date d'archivage:** 10 Décembre 2025

## ℹ️ À propos

Ce dossier contient tous les scripts et configurations de la **première version** du projet French LLM, basée sur l'architecture GPT-2.

### Raison de l'archivage

La version GPT-2 présentait des incompatibilités fondamentales:
- ❌ Tokenizer mismatch (50,257 vocab vs 117,043 Mistral)
- ❌ Non compatible avec LM Studio
- ❌ Architecture inadaptée pour la communauté LLaMA/Mistral

### Branche Git de sauvegarde

La version complète est disponible dans la branche Git:
```bash
git checkout archive-gpt2-legacy
```

---

## 📂 Structure

```
legacy_v1_gpt2/
├── scripts/          # Tous les scripts Python d'entraînement GPT-2
│   ├── train_simple_hf.py
│   ├── api_server.py
│   ├── launch_mistral_training.py
│   ├── finetune_conversations.py
│   └── ... (32 fichiers au total)
│
├── old_runs/         # Anciens checkpoints et résultats d'entraînement
│   └── trained_models/
│       └── runs/french_llm_hf/
│           └── checkpoint-80000/
│
└── README.md         # Ce fichier
```

---

## 📝 Fichiers clés

### Scripts d'entraînement
- `train_simple_hf.py` - Phase 1 (80K steps, loss 3.10)
- `train_phase2_80k.py` - Phase 2 (préparé mais non exécuté)
- `train_finetune_conversations_60k.py` - Phase 3 (préparé)

### Scripts d'API/déploiement
- `api_server.py` - FastAPI server (était running)
- `dashboard_simple.py` - Interface de monitoring

### Scripts d'expérimentation
- `test_api.py` - Tests d'API
- `test_tokenizer_v2.py` - Tests de tokenizer

---

## 📊 Résultats de Phase 1 (GPT-2)

| Métrique | Valeur |
|----------|--------|
| Architecture | GPT-2 (124M parameters) |
| Steps | 80,000 |
| Loss initial | 6.97 |
| Loss final | 3.10 |
| Runtime | ~3.25 heures |
| Tokenizer | GPT-2 (50,257 vocab) |
| Status | ✅ Training réussi, ❌ Inutilisable |

---

## ⚠️ Pourquoi cette version n'est plus utilisée

1. **Tokenizer Mismatch:** Les données pré-tokenisées utilisent Mistral (117K), le modèle utilise GPT-2 (50K)
2. **LM Studio:** GPT-2 n'est pas supporté par LM Studio
3. **Communauté:** La communauté préfère Mistral/LLaMA family
4. **GGUF:** Conversion en GGUF échouée, format invalide

---

## 🔄 Migration vers V2 (Mistral)

La nouvelle version utilise:
- ✅ Architecture Mistral Small (125M parameters)
- ✅ Tokenizer Mistral (117K vocab - aligné avec données)
- ✅ 100% compatible LM Studio (GGUF support)
- ✅ Données pré-tokenisées prêtes (100.5M tokens)

Voir `PLAN_ACTION.md` pour le plan de migration.

---

## 📚 Documentation

Pour référence historique:
- `BILAN_COMPLET.md` - Analyse complète de tous les problèmes
- `BILAN_COURT.md` - Résumé exécutif
- `ENSEMBLE_TOKENIZERS_COMPLET.md` - Analyse des tokenizers

---

## 🔍 Restauration

Si nécessaire de restaurer la version GPT-2:

```bash
# Option 1: Via Git
git checkout archive-gpt2-legacy

# Option 2: Copier depuis ce dossier
cp -r legacy_v1_gpt2/scripts/*.py .
cp -r legacy_v1_gpt2/old_runs/trained_models .
```

---

**Status:** 🗂️ Archivé, non-utilisé  
**Branche Git:** `archive-gpt2-legacy`  
**Dernière mise à jour:** 10 Décembre 2025
