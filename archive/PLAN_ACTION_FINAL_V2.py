#!/usr/bin/env python3
"""
PLAN D'ACTION FINAL - LLM FRANÇAIS V2
Objectif: Modèle performant sans artefacts BPE, publié sur HF/GitHub, utilisable LM Studio
"""

# ============================================================================
# ÉTAT ACTUEL (d'après rapport 7 déc 2025)
# ============================================================================

CHECKPOINTS_DISPONIBLES = {
    "checkpoint_step_17500.pt": {
        "steps": 17500,
        "status": "⚠️ Pas encore testé pour qualité",
        "source": "Phase 2A du plan réentraînement",
    },
    "checkpoint_step_100000.pt": {
        "steps": 100000,
        "status": "⚠️ Français grammatical mais sans sens conversationnel",
        "source": "Phase 1 base pré-entraînée (Wikipedia + FineWeb)",
    },
}

DATASETS_VALIDES_FR = {
    "conversations_train_20M_tokenized.pt": {
        "tokens": 20_000_000,
        "size_mb": 80,
        "status": "✅ Utilisable",
        "quality": "⚠️ Non validé pour % français",
    },
    "dolly_fr_tokenized.pt": {
        "status": "✅ Validé français",
        "quality": "✅ Haute qualité",
    },
    "oasst2_fr_tokenized.pt": {
        "status": "✅ Validé français",
        "quality": "✅ Haute qualité",
    },
}

TOKENIZER_ACTUEL = {
    "type": "SentencePiece Unigram",
    "vocab_size": 32000,
    "path": "trained_models/tokenizers/vincent_tokenizer_FR_v2/",
    "artifacts_bpe": "❓ À TESTER (Ġ, Ċ présents ?)",
}

# ============================================================================
# PROBLÈMES IDENTIFIÉS DES VERSIONS PRÉCÉDENTES
# ============================================================================

PROBLEMES_V1 = {
    "1_catastrophic_forgetting": {
        "cause": "Dataset 78% anglais (conversations_mega_train.txt contaminé)",
        "symptom": "Mélange FR/EN, gibberish, boucles infinies",
        "solution": "✅ Créer conversations_validated_fr.pt (100% FR vérifié)",
    },
    "2_artefacts_bpe": {
        "cause": "Tokenizer BPE Mistral avec Ġ (espace) et Ċ (newline)",
        "symptom": "Post LinkedIn v1: 'BonjourĠcommentĠallez-vous'",
        "solution": "✅ Utiliser vincent_tokenizer_FR_v2 (SentencePiece)",
    },
    "3_lr_trop_eleve": {
        "cause": "LR 3e-4 → 3e-6 encore trop haut pour fine-tuning",
        "symptom": "Modèle diverge après quelques k steps",
        "solution": "✅ Phase 2A: 1e-6, Phase 2B: 5e-7",
    },
    "4_no_early_stopping": {
        "cause": "Pas de validation pendant training",
        "symptom": "Continue même quand qualité régresse",
        "solution": "✅ Implémenter EarlyStoppingValidator",
    },
}

# ============================================================================
# PLAN D'ACTION - 8 ÉTAPES
# ============================================================================

PLAN_ACTION = {
    "step_1": {
        "titre": "VÉRIFIER TOKENIZER V2 (Artefacts BPE)",
        "description": "Tester vincent_tokenizer_FR_v2 pour confirmer absence Ġ, Ċ",
        "script": "test_tokenizer_v2.py",
        "commande": "python test_tokenizer_v2.py",
        "critere_succes": "✅ Pas d'artefacts BPE détectés",
        "temps_estime": "5 min",
        "status": "🔄 EN COURS (sentencepiece non installé)",
    },
    
    "step_2": {
        "titre": "CRÉER DATASET VALIDÉ 100% FR",
        "description": "conversations_validated_fr.pt: OASST2 + Dolly FR uniquement",
        "script": "scripts/create_validated_conversations_fr.py",
        "commande": """
python scripts/create_validated_conversations_fr.py \\
  --sources oasst2,dolly_fr \\
  --max-tokens 50000000 \\
  --tokenizer-path trained_models/tokenizers/vincent_tokenizer_FR_v2 \\
  --output data_clean/conversations_validated_fr.pt
        """,
        "critere_succes": "✅ >95% conversations validées françaises",
        "temps_estime": "10-15 min",
        "status": "⏳ PRÊT (script créé)",
    },
    
    "step_3": {
        "titre": "IMPLÉMENTER EARLY STOPPING",
        "description": "Ajouter EarlyStoppingValidator dans train script",
        "fichier": "scripts/v2/train_phase2_conversations.py",
        "modifications": [
            "Classe EarlyStoppingValidator avec métriques qualité",
            "Tests auto sur prompts FR à chaque eval_interval",
            "Arrêt si 5 régressions consécutives",
        ],
        "critere_succes": "✅ Training s'arrête si qualité régresse",
        "temps_estime": "30 min",
        "status": "⏳ À FAIRE",
    },
    
    "step_4": {
        "titre": "PHASE 2A: INSTRUCTION TUNING PROGRESSIF",
        "description": "100k → 105k steps avec LR 1e-6",
        "dataset": "50% base (Wikipedia) + 50% conversations_validated_fr.pt",
        "commande": """
python scripts/v2/train_phase2_conversations.py \\
  --resume-from trained_models/runs/checkpoint_step_100000.pt \\
  --dataset-path data_clean/conversations_validated_fr.pt \\
  --tokenizer-path trained_models/tokenizers/vincent_tokenizer_FR_v2 \\
  --run-name french_v2_phase2a \\
  --max-steps 105000 \\
  --lr 1e-6 \\
  --batch-size 10 \\
  --checkpoint-interval 500 \\
  --eval-interval 100
        """,
        "critere_succes": "✅ Checkpoint 105k avec génération cohérente français",
        "temps_estime": "1-2h GPU",
        "status": "⏳ Après steps 1-3",
    },
    
    "step_5": {
        "titre": "PHASE 2B: FULL INSTRUCTION TUNING",
        "description": "105k → 120k steps avec LR 5e-7",
        "dataset": "100% conversations_validated_fr.pt",
        "commande": """
python scripts/v2/train_phase2_conversations.py \\
  --resume-from trained_models/runs/french_v2_phase2a/checkpoint_step_105000.pt \\
  --dataset-path data_clean/conversations_validated_fr.pt \\
  --tokenizer-path trained_models/tokenizers/vincent_tokenizer_FR_v2 \\
  --run-name french_v2_phase2b \\
  --max-steps 120000 \\
  --lr 5e-7 \\
  --batch-size 10 \\
  --checkpoint-interval 1000 \\
  --eval-interval 100 \\
  --enable-early-stopping
        """,
        "critere_succes": "✅ Modèle conversationnel français fonctionnel",
        "temps_estime": "3-4h GPU",
        "status": "⏳ Après step 4",
    },
    
    "step_6": {
        "titre": "EXPORT HUGGINGFACE FORMAT",
        "description": "Convertir checkpoint → format Transformers",
        "script": "scripts/export_to_huggingface.py",
        "fichiers_generes": [
            "config.json (architecture GPT)",
            "pytorch_model.bin (weights)",
            "tokenizer.model (SentencePiece)",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        "commande": """
python scripts/export_to_huggingface.py \\
  --checkpoint trained_models/runs/french_v2_phase2b/checkpoint_step_120000.pt \\
  --output-dir trained_models/french_llm_v2_hf \\
  --tokenizer-path trained_models/tokenizers/vincent_tokenizer_FR_v2
        """,
        "critere_succes": "✅ Modèle loadable avec AutoModelForCausalLM",
        "temps_estime": "10 min",
        "status": "⏳ Après step 5",
    },
    
    "step_7": {
        "titre": "CONVERTIR EN GGUF POUR LM STUDIO",
        "description": "Quantization Q4_K_M, Q5_K_M, Q8_0",
        "script": "scripts/convert_to_gguf_mixed.py",
        "commande": """
python scripts/convert_to_gguf_mixed.py \\
  --model-dir trained_models/french_llm_v2_hf \\
  --output-dir trained_models/french_llm_v2_gguf \\
  --quantization Q4_K_M,Q5_K_M,Q8_0
        """,
        "fichiers_generes": [
            "french_llm_v2_q4_k_m.gguf (4-bit, ~2GB)",
            "french_llm_v2_q5_k_m.gguf (5-bit, ~2.5GB)",
            "french_llm_v2_q8_0.gguf (8-bit, ~3.5GB)",
        ],
        "critere_succes": "✅ Loadable dans LM Studio, inférence rapide",
        "temps_estime": "15-20 min",
        "status": "⏳ Après step 6",
    },
    
    "step_8": {
        "titre": "PUBLICATION & COMMUNICATION",
        "description": "HuggingFace Hub + GitHub + LinkedIn",
        "taches": [
            {
                "plateforme": "HuggingFace",
                "actions": [
                    "Créer repo: Vincent-PRO-AI/french-llm-v2",
                    "Upload model files (HF format + GGUF)",
                    "Créer model card avec:",
                    "  - Benchmarks (perplexity, BLEU, samples)",
                    "  - Architecture details",
                    "  - Training process transparent",
                    "  - Limitations honnêtes",
                    "  - Exemples d'utilisation",
                ],
            },
            {
                "plateforme": "GitHub",
                "actions": [
                    "Push sur Vincent-PRO-AI/french-llm-from-scratch",
                    "README complet avec reproduction exacte",
                    "Release v2.0.0 avec:",
                    "  - Checkpoints finaux",
                    "  - Scripts d'entraînement",
                    "  - Dataset preparation code",
                    "  - Docker image",
                ],
            },
            {
                "plateforme": "LinkedIn",
                "actions": [
                    "Post inspiré du v1 (consulter historique)",
                    "Sections:",
                    "  - 🎯 Objectif: LLM français sans artefacts",
                    "  - 📊 Résultats: samples avant/après",
                    "  - 🔧 Tech: SentencePiece, Flash Attention, early stopping",
                    "  - 💾 Démo: Lien HF + GitHub",
                    "  - 🚀 Utilisation: LM Studio ready",
                    "Hashtags: #NLP #AI #MachineLearning #French #OpenSource",
                ],
            },
        ],
        "temps_estime": "1-2h",
        "status": "⏳ Après steps 1-7",
    },
}

# ============================================================================
# CRITÈRES DE SUCCÈS FINAUX
# ============================================================================

CRITERES_SUCCES_GLOBAL = {
    "qualite_generation": {
        "prompt": "Utilisateur: Bonjour, comment vas-tu ?\nAssistant:",
        "attendu": "Réponse polie en français, ≤50 tokens, cohérente",
        "test": "python scripts/test_generation.py --checkpoint final",
    },
    "pas_artefacts_bpe": {
        "test": "Aucun Ġ, Ċ, ▁ visible dans sortie décodée",
        "validation": "test_tokenizer_v2.py",
    },
    "francais_pur": {
        "ratio": ">95% mots français",
        "test": "Analyse avec spaCy fr_core_news_sm",
    },
    "coherence": {
        "perplexity": "<20 sur test set",
        "pas_boucles": "Aucun 3-gram répété >3 fois",
    },
    "deploiement": {
        "hf_hub": "✅ Model card avec 5-star rating",
        "github": "✅ Release v2.0.0 documentée",
        "lm_studio": "✅ GGUF loadable et inférence <100ms",
    },
}

# ============================================================================
# TIMELINE ESTIMÉE
# ============================================================================

TIMELINE = {
    "total_gpu_time": "6-8h",
    "total_prep_time": "2-3h",
    "total_publishing": "2-3h",
    "total_project": "10-14h",
    
    "breakdown": {
        "step_1": "5 min",
        "step_2": "15 min",
        "step_3": "30 min",
        "step_4": "1-2h GPU",
        "step_5": "3-4h GPU",
        "step_6": "10 min",
        "step_7": "20 min",
        "step_8": "1-2h",
    },
}

# ============================================================================
# PROCHAINE ACTION IMMÉDIATE
# ============================================================================

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              🚀 PLAN D'ACTION FINAL - FRENCH LLM V2                      ║
║                                                                           ║
║  Objectif: Modèle performant sans artefacts, publié HF/GitHub/LinkedIn  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

📊 ÉTAT ACTUEL:
   • Checkpoint 100k steps: Base pré-entraînée (Wikipedia + FineWeb FR)
   • Tokenizer V2: SentencePiece 32k (à tester pour artefacts)
   • Datasets: dolly_fr, oasst2_fr disponibles

🎯 PLAN EN 8 ÉTAPES:
   1. ✅ Tester tokenizer V2 (pas d'artefacts Ġ, Ċ)
   2. ⏳ Créer dataset validé 100% FR
   3. ⏳ Implémenter early stopping
   4. ⏳ Phase 2A: Instruction tuning progressif (100k→105k)
   5. ⏳ Phase 2B: Full instruction (105k→120k)
   6. ⏳ Export HuggingFace format
   7. ⏳ Convertir GGUF pour LM Studio
   8. ⏳ Publication HF + GitHub + LinkedIn

⏱️  TEMPS ESTIMÉ: 10-14h total (6-8h GPU)

🔥 PROCHAINE ACTION IMMÉDIATE:

Option A - TOUT AUTOMATISER (recommandé):
   $ python launch_full_v2_pipeline.py

Option B - ÉTAPE PAR ÉTAPE:
   $ python test_tokenizer_v2.py  # Step 1
   $ python scripts/create_validated_conversations_fr.py  # Step 2
   $ # Puis continuer avec training...

Que veux-tu faire ?
""")

if __name__ == "__main__":
    print("\n📝 Ce fichier documente le plan complet.")
    print("Pour lancer: Exécuter les scripts dans l'ordre des steps.\n")
