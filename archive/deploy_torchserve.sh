#!/bin/bash
# 🔥 TorchServe Deployment Script

set -e

echo ""
echo "=================================="
echo "🔥 TORCHSERVE DEPLOYMENT"
echo "=================================="
echo ""

# Check if torch-model-archiver is installed
if ! command -v torch-model-archiver &> /dev/null; then
    echo "📦 Installing torch-model-archiver..."
    pip install -q torchserve torch-model-archiver torch-workflow-archiver
fi

MODEL_DIR="models/french_gpt2_lm_studio"
HANDLER="torchserve_handler.py"
OUTPUT_DIR="model-store"

echo "📁 Creating model store directory..."
mkdir -p "$OUTPUT_DIR"

# Copy model files to temporary location for archiving
TEMP_MODEL="/tmp/french-gpt2-model"
rm -rf "$TEMP_MODEL"
mkdir -p "$TEMP_MODEL"
cp -r "$MODEL_DIR"/* "$TEMP_MODEL/"
cp "$HANDLER" "$TEMP_MODEL/"

echo ""
echo "🔄 Archiving model..."
cd "$TEMP_MODEL"
torch-model-archiver \
    --model-name french-gpt2 \
    --version 1.0 \
    --handler "$HANDLER" \
    --export-path "$(cd - && pwd)/$OUTPUT_DIR" \
    --force
cd -

echo ""
echo "✅ Model archived: $OUTPUT_DIR/french-gpt2.mar"
echo ""

echo "🚀 Starting TorchServe..."
torchserve \
    --start \
    --model-store "$OUTPUT_DIR" \
    --config-file config.properties \
    --models french-gpt2=french-gpt2.mar

echo ""
echo "✅ TorchServe started!"
echo ""
echo "📊 Management API: http://localhost:8081"
echo "🚀 Inference API: http://localhost:8080"
echo "📈 Metrics: http://localhost:8082/metrics"
echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:8080/predictions/french-gpt2 \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"Bonjour, \", \"max_length\": 50}'"
echo ""
