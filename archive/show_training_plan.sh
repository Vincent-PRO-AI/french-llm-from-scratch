#!/bin/bash
# 📋 Affiche le plan complet d'entraînement

clear

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║            📋 PLAN COMPLET D'ENTRAÎNEMENT DU MODÈLE FR 📋            ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "┌─ 🎯 OBJECTIF FINAL ────────────────────────────────────────────────────"
echo "│"
echo "│ Créer un modèle GPT-2 FR complet avec 225k steps et fine-tuning"
echo "│ conversationnel pour qualité optimale"
echo "│"
echo "└──────────────────────────────────────────────────────────────────────────"
echo ""

echo "┌─ 📊 TIMELINE D'ENTRAÎNEMENT ───────────────────────────────────────────"
echo "│"
echo "│  ✅ PHASE 1: Base (5k steps supplémentaires)"
echo "│     ├─ Checkpoint de départ: 80k steps"
echo "│     ├─ Ajouter: 5k steps"
echo "│     ├─ Total: 85k steps"
echo "│     ├─ Dataset: Conversations FR + données antérieures"
echo "│     ├─ Temps: ~1 heure"
echo "│     └─ Status: 🔴 EN COURS ($(date '+%H:%M'))"
echo "│"
echo "│  ⏳ PHASE 2: Données diversifiées (80k steps)"
echo "│     ├─ Checkpoint de départ: Phase 1 (85k)"
echo "│     ├─ Ajouter: 80k steps"
echo "│     ├─ Total: 165k steps"
echo "│     ├─ Dataset: Wikipedia + FineWeb + Conversations"
echo "│     ├─ Temps: 10-12 heures"
echo "│     └─ Output: trained_models/runs/french_llm_phase2/final_model"
echo "│"
echo "│  ⏳ PHASE 3: Fine-tuning Conversationnel (60k steps)"
echo "│     ├─ Checkpoint de départ: Phase 2 (165k)"
echo "│     ├─ Steps: 60k"
echo "│     ├─ Dataset: Conversations FR pures"
echo "│     ├─ LR: 5e-5 (fine-tuning mode)"
echo "│     ├─ Temps: 6-7 heures"
echo "│     └─ Output: trained_models/runs/french_llm_finetune_conversations/final_model"
echo "│"
echo "└──────────────────────────────────────────────────────────────────────────"
echo ""

echo "┌─ 💾 DATASETS PRÉPARÉS ────────────────────────────────────────────────"
echo "│"

if [ -f "data_clean/tokenization_metadata.json" ]; then
    echo "│ ✅ PHASE 2 Dataset:"
    python3 << 'EOF'
import json
with open('data_clean/tokenization_metadata.json') as f:
    meta = json.load(f)
    p2 = meta['phase2']
    print(f"│    • Train: {p2['train_tokens']/1e6:.1f}M tokens ({p2['train_sequences']:,} séquences)")
    print(f"│    • Test:  {p2['test_tokens']/1e6:.1f}M tokens ({p2['test_sequences']:,} séquences)")
    print(f"│    • Capacité: {p2['estimated_steps']:,} steps (Target: {p2['target_steps']:,} steps)")
    if p2['estimated_steps'] >= p2['target_steps']:
        print(f"│    • Status: ✅ Suffisant pour {p2['target_steps']:,} steps")
    else:
        deficit = p2['target_steps'] - p2['estimated_steps']
        print(f"│    • Status: ⚠️  Déficit: {deficit:,} steps")
EOF
    echo "│"
    echo "│ ✅ FINE-TUNING Dataset:"
    python3 << 'EOF'
import json
with open('data_clean/tokenization_metadata.json') as f:
    meta = json.load(f)
    ft = meta['finetune']
    print(f"│    • Train: {ft['train_tokens']/1e6:.1f}M tokens ({ft['train_sequences']:,} séquences)")
    print(f"│    • Test:  {ft['test_tokens']/1e6:.1f}M tokens ({ft['test_sequences']:,} séquences)")
    print(f"│    • Capacité: {ft['estimated_steps']:,} steps (Target: {ft['target_steps']:,} steps)")
    if ft['estimated_steps'] >= ft['target_steps']:
        print(f"│    • Status: ✅ Suffisant pour {ft['target_steps']:,} steps")
    else:
        deficit = ft['target_steps'] - ft['estimated_steps']
        print(f"│    • Status: ⚠️  Déficit: {deficit:,} steps")
EOF
else
    echo "│ ⏳ Datasets pas encore tokenisés"
    echo "│    Lancer: python3 tokenize_fr_datasets.py"
fi

echo "│"
echo "└──────────────────────────────────────────────────────────────────────────"
echo ""

echo "┌─ 🔧 CONFIGURATION ENTRAÎNEMENT ───────────────────────────────────────"
echo "│"
echo "│ Model:              GPT-2 (124M parameters)"
echo "│ Tokenizer:          Mistral (117k vocab)"
echo "│ Sequence length:    512 tokens"
echo "│ Batch size:         4 (per_device)"
echo "│ Accumulation:       1 step"
echo "│ Learning rate:      1e-4 (Phase 2), 5e-5 (Fine-tuning)"
echo "│ Scheduler:          Cosine annealing"
echo "│ Precision:          BF16 + Gradient Checkpointing"
echo "│ Optimizer:          AdamW (torch)"
echo "│ Warmup steps:       500 (Phase 2), 300 (Fine-tuning)"
echo "│ Save interval:      Every 500 steps"
echo "│ GPU:                RTX 5080 (16GB)"
echo "│ Hardware:           R7 9700X, 64GB DDR5"
echo "│"
echo "└──────────────────────────────────────────────────────────────────────────"
echo ""

echo "┌─ ⏱️  TEMPS ESTIMÉ ──────────────────────────────────────────────────────"
echo "│"
echo "│ Phase 1:        ~1 heure    (5k steps)"
echo "│ Phase 2:        ~10-12 hours (80k steps)"
echo "│ Phase 3:        ~6-7 hours   (60k steps)"
echo "│ ────────────────────────────"
echo "│ TOTAL:          ~27-30 hours GPU"
echo "│"
echo "│ À 6-7 steps/sec: ~230k tokens/sec"
echo "│ À 4 tokens/sample: ~58k samples/sec"
echo "│"
echo "└──────────────────────────────────────────────────────────────────────────"
echo ""

echo "┌─ 🚀 COMMANDES À EXÉCUTER ──────────────────────────────────────────────"
echo "│"
echo "│ 1️⃣  Tokenizer les données:"
echo "│     python3 tokenize_fr_datasets.py"
echo "│"
echo "│ 2️⃣  Lancer tout automatiquement:"
echo "│     bash run_full_pipeline.sh"
echo "│"
echo "│ 3️⃣  Ou lancer manuellement:"
echo "│     python3 train_phase2_80k.py"
echo "│     python3 train_finetune_conversations_60k.py"
echo "│"
echo "│ 4️⃣  Monitorer en temps réel:"
echo "│     bash dashboard_live.sh"
echo "│     tail -f training_phase2.log"
echo "│     tail -f training_finetune.log"
echo "│"
echo "└──────────────────────────────────────────────────────────────────────────"
echo ""

echo "┌─ 📁 FICHIERS CLÉS ─────────────────────────────────────────────────────"
echo "│"
echo "│ Tokenization:"
echo "│   • tokenize_fr_datasets.py"
echo "│   • data_clean/phase2_train_mistral_tokenized.pt"
echo "│   • data_clean/finetune_train_mistral_tokenized.pt"
echo "│   • data_clean/tokenization_metadata.json"
echo "│"
echo "│ Training:"
echo "│   • train_phase2_80k.py"
echo "│   • train_finetune_conversations_60k.py"
echo "│   • run_full_pipeline.sh (orchestrateur)"
echo "│"
echo "│ Outputs:"
echo "│   • trained_models/runs/french_llm_phase2/checkpoint-*/model"
echo "│   • trained_models/runs/french_llm_finetune_conversations/final_model"
echo "│"
echo "└──────────────────────────────────────────────────────────────────────────"
echo ""

echo "┌─ ✅ CHECKLIST ──────────────────────────────────────────────────────────"
echo "│"
echo "│ [ ] Phase 1 complétée (85k steps)"
if [ -f "data_clean/tokenization_metadata.json" ]; then
    echo "│ [✓] Données tokenisées"
else
    echo "│ [ ] Données à tokeniser"
fi
echo "│ [ ] Phase 2 lancée et complétée (165k steps)"
echo "│ [ ] Phase 3 lancée et complétée (225k steps)"
echo "│ [ ] Export en SafeTensors/GGUF"
echo "│ [ ] Test sur LM Studio"
echo "│ [ ] Upload HuggingFace Hub"
echo "│"
echo "└──────────────────────────────────────────────────────────────────────────"
echo ""

echo "🎯 STATUS ACTUEL:"
echo ""

# Vérifier Phase 1
if [ -d "trained_models/runs/french_llm_hf_extended/checkpoint-85000" ]; then
    echo "   ✅ Phase 1: Checkpoint-85000 détecté"
else
    echo "   🔴 Phase 1: En cours ou non commencée"
fi

# Vérifier Phase 2
if [ -d "trained_models/runs/french_llm_phase2/final_model" ]; then
    echo "   ✅ Phase 2: Complètée"
elif [ -d "trained_models/runs/french_llm_phase2" ]; then
    echo "   🟡 Phase 2: En cours"
else
    echo "   ⏳ Phase 2: Non commencée"
fi

# Vérifier Fine-tuning
if [ -d "trained_models/runs/french_llm_finetune_conversations/final_model" ]; then
    echo "   ✅ Fine-tuning: Complètée"
elif [ -d "trained_models/runs/french_llm_finetune_conversations" ]; then
    echo "   🟡 Fine-tuning: En cours"
else
    echo "   ⏳ Fine-tuning: Non commencée"
fi

echo ""
