#!/bin/bash
# 🚀 Lance TOUS les services de production

set -e

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║         🚀 LANCEMENT TOUS LES SERVICES 🚀          ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Service 1: FastAPI (déjà actif)
echo "📡 FastAPI Status:"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✅ Déjà actif sur http://localhost:8000"
else
    echo "   🔄 Démarrage FastAPI..."
    nohup python3 api_server.py > api_server.log 2>&1 &
    sleep 3
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✅ FastAPI démarré avec succès!"
    else
        echo "   ❌ Erreur au démarrage de FastAPI"
        exit 1
    fi
fi

# Service 2: TorchServe (optionnel, haute performance)
echo ""
echo "🔥 TorchServe Status:"
if command -v torchserve &> /dev/null; then
    if pgrep -f "java.*torchserve" > /dev/null 2>&1; then
        echo "   ✅ TorchServe déjà actif sur http://localhost:8080"
    else
        echo "   🔄 Démarrage TorchServe..."
        mkdir -p model-store
        
        # Créer un répertoire de modèle pour TorchServe
        TORCHSERVE_MODEL_DIR="model-store/french-gpt2"
        mkdir -p "$TORCHSERVE_MODEL_DIR"
        cp -r models/french_gpt2_pytorch/* "$TORCHSERVE_MODEL_DIR/" 2>/dev/null || true
        cp torchserve_handler.py "$TORCHSERVE_MODEL_DIR/handler.py" 2>/dev/null || true
        
        # Démarrer TorchServe
        torchserve --start \
            --model-store model-store \
            --config-file config.properties 2>/dev/null || true
        
        sleep 5
        if curl -s http://localhost:8081/models > /dev/null 2>&1; then
            echo "   ✅ TorchServe démarré avec succès!"
        else
            echo "   ⚠️  TorchServe démarrage en arrière-plan..."
        fi
    fi
else
    echo "   ⏭️  TorchServe non installé (optionnel)"
    echo "      Pour installer: pip install torchserve torch-model-archiver"
fi

# Service 3: Dashboard (optionnel)
echo ""
echo "📊 Dashboard Status:"
if command -v python3 &> /dev/null && [ -f "dashboard/server.py" ]; then
    if pgrep -f "dashboard/server.py" > /dev/null 2>&1; then
        echo "   ✅ Dashboard déjà actif sur http://localhost:5000"
    else
        echo "   ⏭️  Dashboard peut être lancé avec:"
        echo "      python3 dashboard/server.py"
    fi
else
    echo "   ⏭️  Dashboard non disponible"
fi

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║               ✅ SERVICES ACTIFS ✅               ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "🔗 ENDPOINTS DISPONIBLES:"
echo ""
echo "   FastAPI REST API:"
echo "   ├─ Health:     http://localhost:8000/health"
echo "   ├─ Info:       http://localhost:8000/info"
echo "   ├─ Generate:   POST http://localhost:8000/generate"
echo "   └─ Docs:       http://localhost:8000/docs"
echo ""
if pgrep -f "java.*torchserve" > /dev/null 2>&1; then
    echo "   TorchServe (Haute Perf):"
    echo "   ├─ Inference:  http://localhost:8080/predictions/french-gpt2"
    echo "   └─ Management: http://localhost:8081/models"
    echo ""
fi
echo "   Web Interface:"
echo "   └─ Test UI:    open web_interface.html"
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║      📖 TESTER LE MODÈLE:                          ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "1️⃣  API Test rapide:"
echo "   curl -s http://localhost:8000/info | json_pp"
echo ""
echo "2️⃣  Générer du texte:"
echo "   curl -X POST http://localhost:8000/generate \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"prompt\": \"Bonjour, \", \"max_length\": 80, \"temperature\": 0.7}'"
echo ""
echo "3️⃣  Web Interface:"
echo "   open web_interface.html"
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║          📊 INFORMATIONS DU MODÈLE                 ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
curl -s http://localhost:8000/info | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'   Type: {data.get(\"model_type\", \"N/A\")}'); print(f'   Paramètres: {data.get(\"parameters\", 0) / 1e6:.1f}M'); print(f'   Layers: {data.get(\"num_layers\", 0)}'); print(f'   Heads: {data.get(\"num_heads\", 0)}'); print(f'   Vocab: {data.get(\"vocab_size\", 0):,}'); print(f'   Context: {data.get(\"context_length\", 0):,} tokens')"
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║         🎉 TOUS LES SERVICES LANCÉS! 🎉           ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
