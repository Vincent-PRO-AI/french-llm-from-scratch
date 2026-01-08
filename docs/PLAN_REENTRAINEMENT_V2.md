# Plan de Réentraînement V2 - LLM Français Optimisé

## 🎯 Objectif
Créer un modèle conversationnel français **fonctionnel et reproductible** en évitant le catastrophic forgetting.

## 📊 Diagnostic des Échecs Précédents

### Checkpoints Testés (Tous Défaillants)
| Checkpoint | Steps | Statut | Problème |
|------------|-------|--------|----------|
| 65k | 60k base + 5k FT | ❌ | Incohérent (post LinkedIn optimiste) |
| 100k | Base | ⚠️ | Français grammatical mais sans sens |
| 195k | Extended FT | ❌ | Mélange FR/EN, dégradé |
| 200k | LinkedIn v2 | ❌ | Gibberish pur |
| 300k | Mistral | ❌ | Boucles infinies "etique" |

### Causes Racines
1. **Dataset contamination** : `conversations_mega_train.txt` était 78% anglais (même si tokenized corrigé)
2. **LR trop élevé** : 3e-4 → 3e-6 encore trop pour fine-tuning
3. **Pas de validation early stopping** : continue training même quand qualité régresse
4. **Format prompt incompatible** : "Utilisateur:\nAssistant:" pas appris en base

## 🏗️ Architecture de Réentraînement

### Phase 1 : Base Pré-entraînée Pure (0-100k steps)
**Dataset** : Wikipedia FR + FineWeb FR (100% français validé)
- ✅ Déjà fait : checkpoint_step_100000.pt
- **Action** : Garder comme base, ne PAS refaire

**Résultat attendu** : Modèle de langue français général (pas conversationnel)

### Phase 2 : Instruction Tuning Progressif (100k-120k steps)

#### Étape 2A : Adaptation Format (100k-105k)
**Dataset** : Mélange 50% base + 50% conversations
- But : Introduire progressivement le format "Utilisateur:/Assistant:"
- LR : **1e-6** (ultra-conservateur)
- Validation : Toutes les 100 steps

**Nouveau dataset** : Créer `conversations_progressive_fr.pt`
```python
# Combiner :
# - 50% Wikipedia/FineWeb (continuité)
# - 50% Conversations FR validées (nouveau format)
```

#### Étape 2B : Full Instruction (105k-120k)
**Dataset** : 100% conversations françaises VALIDÉES
- Sources prioritaires :
  - ✅ OASST2 FR (filtré manuellement)
  - ✅ Dolly FR (2k dialogues vérifiés)
  - ❌ ÉVITER UltraChat (contamination anglaise suspectée)
- LR : **5e-7** (encore plus bas)
- Checkpoints : **Tous les 1000 steps**
- **Validation obligatoire** : Tester génération à chaque checkpoint

### Phase 3 : Refinement & Alignment (120k-130k)
**Dataset** : Conversations high-quality + preference learning
- Instructions courtes : "Bonjour" → "Bonjour ! Comment puis-je vous aider ?"
- Rejeter réponses longues incohérentes
- LR : **1e-7** (stabilisation finale)

## 🛠️ Optimisations Techniques

### 1. Dataset Preparation
```bash
# Créer nouveau dataset conversations 100% FR validé
python scripts/create_validated_conversations_fr.py \
  --sources oasst2_fr,dolly_fr \
  --validate-language \
  --max-tokens 50000000 \
  --output data_clean/conversations_validated_fr.pt
```

### 2. Training avec Early Stopping
```python
# Ajouter dans train_subtitles_transformer.py
class EarlyStoppingValidator:
    def __init__(self, test_prompts, tokenizer, max_degradation=5):
        self.test_prompts = test_prompts
        self.tokenizer = tokenizer
        self.best_score = -float('inf')
        self.degradation_count = 0
        self.max_degradation = max_degradation
    
    def validate(self, model, step):
        """Retourne True si training doit continuer, False si arrêt"""
        score = self.compute_quality_score(model)
        
        if score < self.best_score:
            self.degradation_count += 1
            print(f"⚠️  Step {step}: Qualité régresse ({score:.3f} < {self.best_score:.3f})")
            
            if self.degradation_count >= self.max_degradation:
                print(f"🛑 Early stopping: {self.max_degradation} régressions consécutives")
                return False
        else:
            self.best_score = score
            self.degradation_count = 0
            print(f"✅ Step {step}: Qualité améliore ({score:.3f})")
        
        return True
    
    def compute_quality_score(self, model):
        """Score basé sur : % français, cohérence, pas de boucles"""
        # TODO: Implémenter métriques
        pass
```

### 3. Validation Checkpoints Automatique
```python
# Script test_checkpoint_quality.py
def validate_checkpoint(ckpt_path):
    """Retourne score 0-100 de qualité"""
    model = load_model(ckpt_path)
    
    scores = []
    for prompt in TEST_PROMPTS_FR:
        output = generate(model, prompt)
        
        # Métriques
        french_ratio = count_french_words(output) / total_words(output)
        has_loops = detect_loops(output)  # "etique etique"
        coherence = perplexity(output)
        
        score = (
            french_ratio * 40 +  # 40% poids langue
            (1 - has_loops) * 30 +  # 30% pas de boucles
            (1 / coherence) * 30  # 30% cohérence
        )
        scores.append(score)
    
    return np.mean(scores)
```

### 4. Learning Rate Schedule Adaptatif
```python
# Réduire LR si val_loss stagne
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,  # Diviser par 2
    patience=3,  # Après 3 évals sans amélioration
    min_lr=1e-8
)
```

## 📋 Commandes de Lancement

### Phase 2A : Introduction Progressive (100k→105k)
```bash
python scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --resume-from trained_models/runs/french_medium_base_60k/checkpoint_step_100000.pt \
  --pretokenized-path data_clean/conversations_progressive_fr.pt \
  --tokenizer-path trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --run-name french_v2_instruction_progressive \
  --max-steps 105000 \
  --lr 1e-6 \
  --batch-size 8 \
  --checkpoint-interval 500 \
  --eval-interval 100 \
  --device cuda \
  --use-amp \
  --weight-decay 0.01 \
  --sample-interval 200
```

### Phase 2B : Full Instruction (105k→120k)
```bash
python scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --resume-from trained_models/runs/french_v2_instruction_progressive/checkpoint_step_105000.pt \
  --pretokenized-path data_clean/conversations_validated_fr.pt \
  --tokenizer-path trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --run-name french_v2_instruction_full \
  --max-steps 120000 \
  --lr 5e-7 \
  --batch-size 8 \
  --checkpoint-interval 1000 \
  --eval-interval 100 \
  --device cuda \
  --use-amp \
  --weight-decay 0.01 \
  --enable-early-stopping \
  --early-stopping-patience 5
```

## ✅ Critères de Succès

### Validation à Chaque Checkpoint
**Prompts de test obligatoires** :
1. "Utilisateur: Bonjour, comment vas-tu ?\nAssistant:"
   - ✅ Attendu : Réponse polie en français (≤50 tokens)
   - ❌ Échec : Anglais, boucles, gibberish

2. "Utilisateur: Quelle est la capitale de la France ?\nAssistant:"
   - ✅ Attendu : "Paris" + contexte optionnel
   - ❌ Échec : Réponse incorrecte ou hors-sujet

3. "Utilisateur: Explique-moi ce qu'est l'intelligence artificielle.\nAssistant:"
   - ✅ Attendu : Définition cohérente 2-3 phrases français
   - ❌ Échec : Incohérence, mixing languages

### Métriques Quantitatives
- **French token ratio** : >95%
- **Val loss** : Ne doit PAS augmenter >5% vs checkpoint précédent
- **Perplexity sur test set** : <20 (baseline GPT-2 small ~30)
- **Pas de boucles** : Aucun 3-gram répété >3 fois

## 🎓 Leçons des Échecs

1. **Ne JAMAIS faire confiance à la loss seule** → Toujours valider génération
2. **LR fine-tuning** : Diviser par 100-1000 vs pre-training (1e-6 to 1e-7)
3. **Dataset quality > quantity** : 10M tokens validés > 200M contaminés
4. **Early stopping essentiel** : Catastrophic forgetting arrive subitement
5. **Checkpoints fréquents** : Toutes les 500-1000 steps en fine-tuning

## 📦 Livrables Finaux

Si succès :
1. ✅ Checkpoint validé 100% français conversationnel
2. ✅ GGUF quantized Q4_K_M pour déploiement
3. ✅ HuggingFace model card avec benchmarks honnêtes
4. ✅ README avec reproduction exacte
5. ✅ Demo Gradio/Streamlit avec exemples réels

## ⏱️ Estimation Temps

- Phase 2A (5k steps) : ~1h
- Phase 2B (15k steps) : ~3-4h
- Phase 3 (10k steps) : ~2h
- **Total** : ~6-7h GPU time

## 🚀 Prochaine Action Immédiate

1. **Créer dataset validé** : `conversations_validated_fr.pt` (OASST2 + Dolly seulement)
2. **Ajouter early stopping** dans `train_subtitles_transformer.py`
3. **Lancer Phase 2A** depuis checkpoint 100k
4. **Valider à step 100500, 101000, etc.** → Arrêter dès régression

---

**Question décisive** : Veux-tu que je lance la Phase 2A maintenant avec :
- Dataset : OASST2 FR (validé) + Dolly FR uniquement
- LR : 1e-6
- Validation : Toutes les 500 steps avec tests automatiques
- Durée : ~1h pour 5k steps

Ou préfères-tu d'abord que je prépare le dataset validé et le système d'early stopping ?
