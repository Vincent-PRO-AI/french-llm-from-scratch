#!/bin/bash
# Tokenise les conversations massives téléchargées

set -e

echo "=================================================="
echo "🔧 TOKENISATION DES CONVERSATIONS MASSIVES"
echo "=================================================="

TOKENIZER="trained_models/tokenizers/fineweb-32k/tokenizer.json"
INPUT_DIR="data_clean/conversations_massive"
OUTPUT_BASE="data_clean"

# Vérifier que le tokenizer existe
if [ ! -f "$TOKENIZER" ]; then
    echo "❌ Tokenizer non trouvé: $TOKENIZER"
    exit 1
fi

# Vérifier que le répertoire d'entrée existe
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Répertoire non trouvé: $INPUT_DIR"
    echo "💡 Lancez d'abord: python scripts/download_massive_conversations.py"
    exit 1
fi

echo "📂 Input: $INPUT_DIR"
echo "🔤 Tokenizer: $TOKENIZER"
echo ""

# Tokeniser chaque fichier séparément
for txt_file in "$INPUT_DIR"/*.txt; do
    if [ -f "$txt_file" ]; then
        filename=$(basename "$txt_file" .txt)
        output_file="${OUTPUT_BASE}/${filename}_tokenized.pt"

        # Ignorer les fichiers vides
        if [ ! -s "$txt_file" ]; then
            echo "⚠️  Ignoré (fichier vide): $filename.txt"
            continue
        fi

        echo "🔄 Tokenisation: $filename..."
        # Ne pas arrêter tout le script si un fichier échoue
        set +e
        .venv/bin/python scripts/pretokenize_file.py \
            --input-file "$txt_file" \
            --tokenizer "$TOKENIZER" \
            --output "$output_file"
        status=$?
        set -e
        if [ $status -ne 0 ]; then
            echo "⚠️  Échec tokenisation: $filename (on continue)"
            continue
        fi

        echo "✅ Créé: $output_file"
        echo ""
    fi
done

echo "=================================================="
echo "✅ TOKENISATION TERMINÉE"
echo "=================================================="
echo ""
echo "💡 Prochaine étape: Combiner tous les datasets"
echo "   python scripts/combine_all_datasets.py"
