# 🔍 RETOUR D'EXPÉRIENCE - TRAINING FRENCH LLM 250K
## Erreurs, Leçons Apprises et Guide pour Prochains Départs

**Session**: 2025-12-21  
**Durée totale**: ~24 heures  
**Résultat final**: ✅ Training complété (145k→250k steps), ⚠️ Qualité modèle insuffisante  

---

## 📋 ERREURS MAJEURES COMMISES

### ❌ ERREUR #1: Supposition sans validation du hardware
**Problème**:
- Utilisateur a dit "j'ai 96GB RAM" 
- J'ai cru sans vérifier
- Réalité: 45GB seulement (Hyper-V balloon driver)

**Impact**: 
- Perte de temps à créer des plans pour 96GB
- Configurations initiales surdimensionnées
- Comprendre tard les vraies limites

**Leçon**:
```
✅ TOUJOURS vérifier avant de planifier:
  1. free -h
  2. nvidia-smi
  3. lscpu
  4. uname -a
  5. Vérifier les vrais paramètres, pas les suppositions
```

---

### ❌ ERREUR #2: Pas d'évaluation de qualité avant d'optimiser

**Problème**:
- A immédiatement optimisé le training (batch_56, RAM allocation)
- N'a jamais testé la **qualité réelle du modèle**
- Découvert à la fin que le modèle génère du charabia (mélange FR/EN)

**Impact**:
- 6+ heures de training pour un modèle inutilisable
- Optimisations pour rien
- Ressources gaspillées

**Leçon**:
```
✅ AVANT toute optimisation:
  1. Tester la qualité du checkpoint existant (145k)
  2. Évaluer samples générés
  3. Mesurer performance réelle
  4. PUIS décider si optimisation vaut le coup
  
❌ Ne pas supposer que "loss faible = qualité bonne"
```

---

### ❌ ERREUR #3: Mauvaise composition des données d'entraînement

**Problème**:
- Données = FineWeb (50% anglais) + LMSYS (mélange) + Wikipedia
- Aucun filtrage français
- Tokenizer Mistral pas optimisé pour français

**Impact**:
- Modèle apprend du charabia français/anglais
- Impossible de faire répondre aux questions en français
- Loss faible mais qualité output = 0

**Leçon**:
```
✅ AVANT de lancer un training:
  1. Analyser la composition des données
     - % français vs autres langues
     - Qualité des sources
     - Longueur moyenne des textes
  
  2. Nettoyer les données:
     - Retirer contenu non-français de FineWeb
     - Valider LMSYS-fr contient réellement du français
     - Équilibrer les sources
  
  3. Tester le tokenizer:
     - Vérifier qu'il tokenise bien le français
     - Pas de tokens étranges
     - Bonne couverture vocabulaire

❌ Ne pas lancer avec des données "probablement OK"
```

---

### ❌ ERREUR #4: Pas de checkpoint de contrôle intermédiaire

**Problème**:
- Training lancé directement 145k → 250k
- Aucun test intermédiaire (175k, 200k, etc)
- Découvert le problème de qualité seulement à la fin

**Impact**:
- Perdu 6 heures si on avait détecté à 175k
- Impossible de revenir en arrière rapidement
- Aucune validation progressive

**Leçon**:
```
✅ Plan d'évaluation intermédiaire:
  1. Faire des tests à:
     - 50% des steps
     - 75% des steps
     - Avant le final
  
  2. À chaque test:
     - Générer 10+ samples
     - Évaluer qualité grammaticale
     - Checker présence d'anglais
     - Vérifier réponses aux questions
  
  3. Critères de STOP:
     - Si qualité se dégrade → STOP
     - Si loss stagne → STOP
     - Si mélange FR/EN → STOP
```

---

### ❌ ERREUR #5: Pas d'analyse des hyperparamètres

**Problème**:
- Batch_size 4 + gradient_accum 3 = batch_12 effectif
- Pas de justification pourquoi ces valeurs
- Pas de test avec autres configs (batch_8, batch_16, etc)

**Impact**:
- Peut-être pas la meilleure config
- Performance sous-optimale
- Temps de training potentiellement perdu

**Leçon**:
```
✅ AVANT de lancer le training final:
  1. Tester plusieurs configs sur 5k steps:
     - batch_4 + grad_accum_1 (batch_4 eff)
     - batch_8 + grad_accum_1 (batch_8 eff)
     - batch_4 + grad_accum_3 (batch_12 eff)
     - batch_8 + grad_accum_2 (batch_16 eff)
  
  2. Comparer:
     - Speed (steps/sec)
     - VRAM utilisée
     - Loss trajectory
     - Stabilité (OOM?)
  
  3. Choisir le meilleur trade-off
     vitesse vs qualité vs stabilité
```

---

### ❌ ERREUR #6: Pas de documentation pendant le processus

**Problème**:
- Aucun log des décisions prises
- Pas de tracé des problèmes rencontrés
- Découvert trop tard ce qui s'était passé

**Impact**:
- Difficile de reproduire/expliquer
- Impossible d'améliorer le processus
- Leçons non documentées

**Leçon**:
```
✅ Documenter EN TEMPS RÉEL:
  1. Créer un fichier SESSION_LOG.md
  2. Noter chaque décision:
     - Quoi: description
     - Pourquoi: justification
     - Quand: timestamp
     - Résultat: outcome
  
  3. Format:
     [HH:MM] DECISION: Lancer training batch_4
     Raison: VRAM limit, batch_12 safe
     Résultat: OK, running
     
  4. À la fin:
     - Rétrospective
     - Erreurs identifiées
     - Leçons apprises
```

---

## 📊 TABLEAU DES ERREURS

| # | Erreur | Cause | Impact | Prévention |
|---|--------|-------|--------|-----------|
| 1 | Supposer RAM sans vérifier | Confiance mal placée | -3h planification | `free -h` avant tout |
| 2 | Optimiser sans évaluer qual | Priorités mal alignées | -6h training inutile | Test 145k d'abord |
| 3 | Données contaminées (FR/EN) | Pas d'analyse préalable | Output inutilisable | Analyser composition |
| 4 | Pas de checkpoints interméd | Pas de plan d'éval | Découverte tard du pb | Tests à 50%, 75% |
| 5 | Config sans justification | Pas d'expérimentation | Perfs sous-optimales | Benchmark 5 configs |
| 6 | Pas de logs/docs | Oubli des best practices | Impossible à reproduire | LOG tout en temps réel |

---

## ✅ CHECKLIST POUR PROCHAIN DÉPART DE ZÉRO

### Phase 0: Diagnostic Initial (30 min)
```
☐ Vérifier hardware réel:
  ☐ free -h (RAM total et libre)
  ☐ nvidia-smi (VRAM, GPU type)
  ☐ lscpu (nombre cores)
  ☐ Vérifier drivers (CUDA version)
  
☐ Analyser données existantes:
  ☐ Vérifier composition (FR vs autres)
  ☐ Compter lignes par source
  ☐ Valider UTF-8, encodage
  ☐ Estimer taille totale
  
☐ Tester infrastructure:
  ☐ Training simple 100 steps (batch_1)
  ☐ Vérifier CUDA, pas d'erreurs
  ☐ Tester checkpointing
  ☐ Vérifier metrics logging
```

### Phase 1: Baseline Quality (2-4 heures)
```
☐ Évaluer checkpoint existant:
  ☐ Charger 145k ou autre
  ☐ Générer 20+ samples
  ☐ Évaluer qualité grammaticale
  ☐ Mesurer % français vs anglais
  ☐ Tester réponses questions
  
☐ Documenter baseline:
  ☐ Créer rapport de qualité
  ☐ Mesurer loss (train/val)
  ☐ Calculer perplexity
  ☐ Archiver samples
```

### Phase 2: Hyperparameter Search (4-6 heures)
```
☐ Tester configurations:
  ☐ Batch 4: 5k steps
  ☐ Batch 8: 5k steps
  ☐ Batch 12: 5k steps
  ☐ Batch 16: 5k steps (si possible)
  
☐ Comparer:
  ☐ Speed (steps/sec)
  ☐ VRAM max utilisée
  ☐ Loss trajectory
  ☐ Stabilité (OOM?)
  ☐ Qualité samples
  
☐ Décider config:
  ☐ Meilleur speed/quality
  ☐ Stable sans OOM
  ☐ Documenté avec justification
```

### Phase 3: Data Cleaning (4-8 heures)
```
☐ Analyser sources:
  ☐ FineWeb: % français
  ☐ LMSYS: vérifier contenu
  ☐ Wikipedia: valider format
  
☐ Nettoyer:
  ☐ Filtrer uniquement français
  ☐ Retirer doublons
  ☐ Valider encoding
  ☐ Tester tokenizer
  
☐ Valider:
  ☐ Samples de chaque source
  ☐ Vérifier pas d'anglais
  ☐ Vérifier pas de corruption
```

### Phase 4: Training Plan (1 heure)
```
☐ Définir objectif:
  ☐ Nombre de steps (100k? 500k?)
  ☐ Target loss (2.0? 1.5?)
  ☐ Durée estimée
  
☐ Plan évaluation:
  ☐ Test à 25% des steps
  ☐ Test à 50% des steps
  ☐ Test à 75% des steps
  ☐ Test final
  
☐ Critères d'arrêt:
  ☐ Si loss stagne > 2h
  ☐ Si qualité regresse
  ☐ Si OOM récurrent
  
☐ Monitoring plan:
  ☐ Script de monitoring continu
  ☐ Alerts automatiques
  ☐ Logging complet
```

### Phase 5: Training Execution (Variable)
```
☐ Avant de lancer:
  ☐ Vérifier GPU libre
  ☐ Vérifier RAM libre
  ☐ Vérifier stockage libre
  ☐ Tuer processus zombies
  
☐ Lancer:
  ☐ Avec nohup/tmux
  ☐ Redirection logs
  ☐ Monitoring actif
  
☐ Monitoring continu:
  ☐ Vérifier loss toutes les heures
  ☐ Vérifier pas d'OOM
  ☐ Vérifier checkpoints créés
  ☐ Documenter progression
  
☐ À chaque checkpoint:
  ☐ Vérifier intégrité fichier
  ☐ Archiver si important
  ☐ Mettre à jour metrics
```

### Phase 6: Final Evaluation (2-4 heures)
```
☐ Charger checkpoint final:
  ☐ Vérifier intégrité
  ☐ Tester chargement
  
☐ Évaluer qualité:
  ☐ 20+ samples générés
  ☐ Comparaison avec baseline
  ☐ Mesure d'amélioration
  ☐ Évaluation linguistique
  
☐ Tester utilité:
  ☐ Répondre à questions
  ☐ Tester en français
  ☐ Tester edge cases
  
☐ Documenter résultats:
  ☐ Rapport final
  ☐ Métriques quantitatives
  ☐ Exemples samples
  ☐ Conclusion et recommandations
```

---

## 📝 TEMPLATE SESSION LOG

```markdown
# SESSION LOG - [DATE] - [RUN NAME]

## Contexte Initial
- Hardware: [GPU], [RAM], [CPU]
- Checkpoint: [path, steps]
- Objectif: [description]
- Durée estimée: [hours]

## Décisions Prises
### [HH:MM] DECISION: [Quoi]
**Raison**: [Pourquoi]
**Config**: [Paramètres]
**Résultat**: [Outcome - OK/ERREUR]

## Erreurs Rencontrées
### [HH:MM] ERREUR: [Quoi]
**Cause**: [Pourquoi]
**Solution**: [Comment corrigé]
**Impact**: [Temps perdu / ressources]

## Métriques Finales
- Training loss: [X]
- Validation loss: [X]
- Convergence: [Bonne/Mauvaise]
- Qualité output: [Score 1-10]
- Durée réelle: [hours]

## Leçons Apprises
1. [Leçon 1]
2. [Leçon 2]
3. [Leçon 3]

## Recommandations Prochaine Fois
- ☐ [Action 1]
- ☐ [Action 2]
- ☐ [Action 3]

## Files Générés
- [File 1]: [Description]
- [File 2]: [Description]
```

---

## 🎯 RÈGLES D'OR POUR NEXT TIME

### 1. **DIAGNOSE FIRST, ACT SECOND**
```
❌ Plan → Code → Run
✅ Verify → Analyze → Plan → Code → Run
```

### 2. **NEVER TRUST ASSUMPTIONS**
```
❌ "L'utilisateur a dit 96GB, donc c'est 96GB"
✅ Vérifier: free -h, /proc/meminfo, Hyper-V settings
```

### 3. **QUALITY OVER SPEED**
```
❌ Optimiser pour 8x speedup sur un modèle mauvais
✅ D'abord vérifier la qualité, PUIS optimiser
```

### 4. **INTERMEDIATE CHECKPOINTS**
```
❌ Training 145k → 250k sans pause
✅ Testing à 50%, 75% avec évaluation de qualité
```

### 5. **DOCUMENT EVERYTHING**
```
❌ "Je me souviendrai de mes décisions"
✅ Fichier SESSION_LOG.md avec timestamps et raisons
```

### 6. **AUTOMATE MONITORING**
```
❌ Vérifier manuellement les logs régulièrement
✅ Script monitoring continu qui alerte automatiquement
```

### 7. **VALIDATE DATA FIRST**
```
❌ "Les données sont probablement OK"
✅ Analyser: composition, qualité, encoding, filtres
```

### 8. **TEST CONFIGS**
```
❌ "Batch 12 devrait être bon"
✅ Tester 4, 8, 12, 16 sur 5k steps chacun
```

---

## 📚 RESSOURCES À CRÉER

Pour prochaine session, créer ces outils:

```
□ setup_diagnostic.py
  - Vérifier hardware réel
  - Analyser données existantes
  - Tester infrastructure
  
□ data_analyzer.py
  - Composition (FR vs autres)
  - Qualité des sources
  - Tokenizer coverage
  
□ config_benchmarker.py
  - Tester 4-5 configs
  - Comparer speed/quality
  - Recommander meilleure
  
□ quality_evaluator.py
  - Générer samples
  - Mesurer qualité linguistique
  - Tester réponses questions
  
□ session_monitor.py
  - Logging continu
  - Alerts automatiques
  - Métriques en temps réel
  
□ final_report.py
  - Générer rapport complet
  - Comparer baseline vs final
  - Recommandations
```

---

## 🏁 CONCLUSION

### Ce qui a marché ✅
- Infrastructure (Docker, training script)
- Optimisation système (RAM allocation)
- Monitoring final des métriques
- Documentation rétrospective

### Ce qui n'a pas marché ❌
- Pas de validation de assumptions
- Pas d'évaluation de qualité initialement
- Données contaminées (FR/EN mix)
- Pas de checkpoints de contrôle intermédiaires

### Prochaine fois
Suivre **scrupuleusement** la checklist Phase 0-1 avant de toucher au Phase 5 (Training).
Investir 8-12 heures en préparation pour éviter 24+ heures de training sur un mauvais modèle.

**Golden Rule**: Une heure de préparation vaut mieux qu'une journée de training à jeter.

---

**Archivé le**: 2025-12-21  
**Pour**: Sessions futures de training LLM français  
**Priorité**: ⭐⭐⭐⭐⭐ Lire avant toute nouvelle tentative
