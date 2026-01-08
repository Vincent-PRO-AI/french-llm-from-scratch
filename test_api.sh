#!/bin/bash
# Simple test script for the API

API_URL="http://localhost:8000"

echo "🧪 Testing French LLM API at $API_URL"
echo ""

# Test 1: Health check
echo "[1/3] Health check..."
curl -s "$API_URL/health" | python3 -m json.tool
echo ""

# Test 2: Generate with different prompts
echo "[2/3] Generate with prompt: 'Bonjour, je suis '"
curl -s -X POST "$API_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Bonjour, je suis ",
    "max_tokens": 80,
    "temperature": 0.95,
    "top_k": 200
  }' | python3 -m json.tool

echo ""
echo "[3/3] Generate with prompt: 'Le français est '"
curl -s -X POST "$API_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Le français est ",
    "max_tokens": 60,
    "temperature": 0.7,
    "top_k": 100
  }' | python3 -m json.tool

echo ""
echo "✅ Tests complete!"
echo ""
echo "📖 Full API docs: $API_URL/docs"
echo "📖 OpenAPI schema: $API_URL/openapi.json"
