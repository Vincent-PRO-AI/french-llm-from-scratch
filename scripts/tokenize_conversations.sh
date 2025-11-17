#!/bin/bash
# Script pour tokeniser les nouvelles conversations

set -e

TOKENIZER="trained_models/tokenizers/fineweb-32k/tokenizer.json"
INPUT_DIR="data_clean/conversations_extra"
OUTPUT_FILE="data_clean/conversations_extra_tokenized.pt"

echo "🔧 Tokenisation des conversations supplémentaires..."
echo "   Input: $INPUT_DIR"
echo "   Output: $OUTPUT_FILE"
echo "   Tokenizer: $TOKENIZER"
echo ""

.venv/bin/python scripts/pretokenize_corpus.py \
    --input-dir "$INPUT_DIR" \
    --tokenizer "$TOKENIZER" \
    --output "$OUTPUT_FILE"

echo ""
echo "✅ Tokenisation terminée!"
echo "📊 Fichier créé: $OUTPUT_FILE"

# Afficher la taille
ls -lh "$OUTPUT_FILE"
