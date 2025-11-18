ash
# Pipeline complet: tokeniser → combiner → lancer fine-tuning

set -e

echo "=========================================="
echo "🚀 Pipeline Fine-tuning Conversations"
echo "=========================================="

# Chemins
TOKENIZER="trained_models/tokenizers/fineweb-32k/tokenizer.json"
CONVERSATIONS_OLD="data_clean/conversations_all_tokenized.pt"
CONVERSATIONS_NEW_DIR="data_clean/conversations_extra"
CONVERSATIONS_NEW_FILE="data_clean/conversations_extra_tokenized.pt"
CONVERSATIONS_COMBINED="data_clean/conversations_combined_tokenized.pt"
CONFIG="training_configs/finetune_conversations.json"
LOG_FILE="/tmp/finetune_conversations.log"

# Étape 1: Tokeniser les nouvelles conversations
echo ""
echo "📝 Étape 1/3: Tokenisation des nouvelles conversations"
echo "=========================================="

if [ -f "$CONVERSATIONS_NEW_FILE" ]; then
    echo "✅ Fichier déjà tokenisé: $CONVERSATIONS_NEW_FILE"
else
    .venv/bin/python scripts/pretokenize_corpus.py \
        --input-dir "$CONVERSATIONS_NEW_DIR" \
        --tokenizer "$TOKENIZER" \
        --output "$CONVERSATIONS_NEW_FILE"
fi

# Étape 2: Combiner les datasets
echo ""
echo "🔗 Étape 2/3: Combinaison des datasets"
echo "=========================================="

if [ -f "$CONVERSATIONS_COMBINED" ]; then
    echo "⚠️  Fichier combiné existe déjà: $CONVERSATIONS_COMBINED"
    read -p "Recréer? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$CONVERSATIONS_COMBINED"
    else
        echo "✅ Utilisation du fichier existant"
    fi
fi

if [ ! -f "$CONVERSATIONS_COMBINED" ]; then
    .venv/bin/python scripts/combine_tokenized_datasets.py \
        --inputs "$CONVERSATIONS_OLD" "$CONVERSATIONS_NEW_FILE" \
        --output "$CONVERSATIONS_COMBINED"
fi

# Afficher les stats
echo ""
echo "📊 Statistiques des datasets:"
echo "   - Ancien: $(ls -lh $CONVERSATIONS_OLD | awk '{print $5}')"
echo "   - Nouveau: $(ls -lh $CONVERSATIONS_NEW_FILE | awk '{print $5}')"
echo "   - Combiné: $(ls -lh $CONVERSATIONS_COMBINED | awk '{print $5}')"

# Étape 3: Lancer le fine-tuning
echo ""
echo "🎯 Étape 3/3: Lancement du fine-tuning"
echo "=========================================="
echo "   Config: $CONFIG"
echo "   Dataset: $CONVERSATIONS_COMBINED"
echo "   Log: $LOG_FILE"
echo ""

read -p "Lancer l'entraînement? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Démarrage..."
    
    # Lancer en arrière-plan avec nohup
    nohup .venv/bin/python scripts/train_subtitles_transformer.py \
        --config "$CONFIG" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "✅ Entraînement lancé (PID: $PID)"
    echo "📋 Logs: tail -f $LOG_FILE"
    echo ""
    
    # Attendre 5 secondes et afficher le début des logs
    sleep 5
    echo "📝 Premiers logs:"
    tail -20 "$LOG_FILE"
else
    echo "❌ Annulé"
fi

echo ""
echo "=========================================="
echo "✅ Pipeline terminé"
echo "=========================================="
