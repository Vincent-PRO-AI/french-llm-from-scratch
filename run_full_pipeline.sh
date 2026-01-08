#!/bin/bash
# 🚀 Orchestrateur complet: Tokenization + Entraînements séquentiels

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║    🚀 PIPELINE COMPLET: TOKENIZATION + 3 PHASES 🚀        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PYTHON_CMD="/home/vincent/code/repo/french-llm-from-scratch/.venv/bin/python"
WORKSPACE="/home/vincent/code/repo/french-llm-from-scratch"

# ============================================
# ÉTAPE 1: TOKENIZATION
# ============================================
echo "📊 ÉTAPE 1: TOKENIZATION DONNÉES FRANÇAISES"
echo "════════════════════════════════════════════"
echo ""
echo "Cette étape prépare les données pour 2 phases:"
echo "  • Phase 2: 80k steps (Mixed FR: Wikipedia + FineWeb + Conversations)"
echo "  • Fine-tuning: 60k steps (Conversations FR pures)"
echo ""
read -p "Lancer la tokenization? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔄 Démarrage tokenization..."
    nohup $PYTHON_CMD tokenize_fr_datasets.py > tokenization.log 2>&1 &
    TOKENIZE_PID=$!
    
    echo "   PID: $TOKENIZE_PID"
    echo "   Log: tail -f tokenization.log"
    echo ""
    echo "   Temps estimé: 30-45 minutes"
    echo ""
    
    # Attendre la fin
    wait $TOKENIZE_PID
    echo "✅ Tokenization terminée!"
    echo ""
fi

# ============================================
# ÉTAPE 2: ATTENDRE PHASE 1 (5k steps)
# ============================================
echo ""
echo "⏳ ÉTAPE 2: ATTENDRE FIN PHASE 1 (5k steps)"
echo "════════════════════════════════════════════"
echo ""
echo "Phase 1 actuelle: 5000 steps (80k + 5k = 85k au total)"
echo "Checkpoint attendu: trained_models/runs/french_llm_hf_extended/checkpoint-85000"
echo ""
read -p "Vérifier et attendre le checkpoint? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔍 Attente du checkpoint-85000..."
    
    # Attendre le checkpoint
    while [ ! -d "trained_models/runs/french_llm_hf_extended/checkpoint-85000" ]; do
        echo "   ⏳ $(date '+%H:%M:%S') - Pas encore disponible, vérification dans 60s..."
        sleep 60
    done
    
    echo "✅ Checkpoint-85000 détecté!"
    echo ""
fi

# ============================================
# ÉTAPE 3: PHASE 2 (80k steps)
# ============================================
echo ""
echo "🔥 ÉTAPE 3: PHASE 2 - 80k STEPS SUPPLÉMENTAIRES"
echo "════════════════════════════════════════════════"
echo ""
echo "Dataset: Mixed FR (Wikipedia + FineWeb + Conversations)"
echo "Steps: 85000 → 165000 (80k new)"
echo "Temps estimé: 10-12 heures"
echo ""
read -p "Lancer Phase 2? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Démarrage Phase 2..."
    nohup $PYTHON_CMD train_phase2_80k.py > training_phase2.log 2>&1 &
    PHASE2_PID=$!
    
    echo "   PID: $PHASE2_PID"
    echo "   Log: tail -f training_phase2.log"
    echo ""
    echo "   Checkpoint attendu: trained_models/runs/french_llm_phase2/checkpoint-165000"
    echo ""
    
    # Attendre
    wait $PHASE2_PID
    echo "✅ Phase 2 terminée!"
    echo ""
fi

# ============================================
# ÉTAPE 4: FINE-TUNING (60k steps)
# ============================================
echo ""
echo "🎓 ÉTAPE 4: FINE-TUNING - 60k STEPS CONVERSATIONS"
echo "══════════════════════════════════════════════════"
echo ""
echo "Dataset: Conversations FR pures"
echo "Steps: 60000"
echo "Temps estimé: 6-7 heures"
echo ""
read -p "Lancer Fine-tuning? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Démarrage Fine-tuning..."
    nohup $PYTHON_CMD train_finetune_conversations_60k.py > training_finetune.log 2>&1 &
    FINETUNE_PID=$!
    
    echo "   PID: $FINETUNE_PID"
    echo "   Log: tail -f training_finetune.log"
    echo ""
    echo "   Modèle final: trained_models/runs/french_llm_finetune_conversations/final_model"
    echo ""
    
    # Attendre
    wait $FINETUNE_PID
    echo "✅ Fine-tuning terminé!"
    echo ""
fi

# ============================================
# RÉSUMÉ FINAL
# ============================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ PIPELINE COMPLET TERMINÉ ✅               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 MODÈLES GÉNÉRÉS:"
echo ""
echo "1. 🟢 Modèle Phase 1 (85k steps):"
echo "   trained_models/runs/french_llm_hf_extended/checkpoint-85000/"
echo ""
echo "2. 🟡 Modèle Phase 2 (165k steps):"
echo "   trained_models/runs/french_llm_phase2/final_model/"
echo ""
echo "3. 🔴 Modèle Fine-tuning (60k steps conversations):"
echo "   trained_models/runs/french_llm_finetune_conversations/final_model/"
echo ""
echo "📈 STATISTIQUES COMPLÈTES:"
echo "   • Phase 1: 80k + 5k = 85k steps (base entraînement)"
echo "   • Phase 2: 80k steps (données diversifiées)"
echo "   • Phase 3: 60k steps (fine-tuning conversationnel)"
echo "   • Total: 225k steps"
echo "   • Temps total: ~28-33 heures GPU"
echo ""
echo "🚀 PROCHAINES ÉTAPES:"
echo "   1. Exporter le modèle final en SafeTensors/GGUF"
echo "   2. Tester sur LM Studio"
echo "   3. Uploader sur HuggingFace Hub"
echo ""
