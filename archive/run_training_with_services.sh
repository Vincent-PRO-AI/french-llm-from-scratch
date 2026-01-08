#!/bin/bash
# 🚀 Lance l'entraînement + tous les services en parallèle

set -e

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║   🔥 ENTRAÎNEMENT 5000 STEPS + TOUS LES SERVICES 🔥   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Configuration
TRAINING_LOG="training_5k_extended.log"
METRICS_FILE="training_metrics.jsonl"

# Créer un fichier de log centralisé
echo "📝 Configuration"
echo "   Log: $TRAINING_LOG"
echo "   Metrics: $METRICS_FILE"
echo ""

# Démarrer l'entraînement en arrière-plan
echo "🔥 PHASE 1: Lancement de l'entraînement (5000 steps)"
echo "   Commande: python3 continue_training_5k.py"
echo "   PID: en attente..."
echo ""

# Lancer l'entraînement en background
nohup python3 continue_training_5k.py > "$TRAINING_LOG" 2>&1 &
TRAINING_PID=$!
echo "   ✅ PID: $TRAINING_PID"
sleep 2

# Fonction pour afficher la barre de progression
show_training_progress() {
    if [ -f "$TRAINING_LOG" ]; then
        tail -5 "$TRAINING_LOG" | grep -E "(step|loss|eval)" | tail -1 || echo "   ⏳ Initialisation..."
    fi
}

# Service 1: FastAPI
echo ""
echo "🌐 PHASE 2: Démarrage FastAPI (localhost:8000)"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✅ Déjà actif"
else
    echo "   🔄 Lancement..."
    nohup python3 api_server.py > api_server.log 2>&1 &
    sleep 3
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✅ Démarré avec succès"
    else
        echo "   ⚠️  Vérification en cours..."
    fi
fi

# Service 2: TorchServe
echo ""
echo "⚡ PHASE 3: Démarrage TorchServe (localhost:8080)"
if command -v torchserve &> /dev/null; then
    if pgrep -f "java.*torchserve" > /dev/null 2>&1; then
        echo "   ✅ Déjà actif"
    else
        echo "   🔄 Lancement..."
        mkdir -p model-store
        nohup torchserve --start \
            --model-store model-store \
            --config-file config.properties > torchserve.log 2>&1 &
        sleep 3
        echo "   ✅ Démarrage en arrière-plan"
    fi
else
    echo "   ⏭️  Non installé (optionnel)"
fi

# Service 3: Dashboard
echo ""
echo "📊 PHASE 4: Démarrage Dashboard (localhost:5000)"
if [ -f "dashboard/server.py" ]; then
    if pgrep -f "dashboard/server.py" > /dev/null 2>&1; then
        echo "   ✅ Déjà actif"
    else
        echo "   🔄 Lancement..."
        nohup python3 dashboard/server.py > dashboard.log 2>&1 &
        sleep 2
        if curl -s http://localhost:5000/ > /dev/null 2>&1; then
            echo "   ✅ Démarré avec succès"
        else
            echo "   ⏳ Initialisation..."
        fi
    fi
else
    echo "   ⏭️  Fichier non trouvé"
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║            ✅ TOUS LES SERVICES LANCÉS ✅             ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

echo "📡 ENDPOINTS DISPONIBLES:"
echo ""
echo "   🔴 ENTRAÎNEMENT (en cours):"
echo "   ├─ PID:          $TRAINING_PID"
echo "   ├─ Log:          tail -f $TRAINING_LOG"
echo "   └─ Durée:        ~1h pour 5000 steps"
echo ""
echo "   🟢 API REST (FastAPI):"
echo "   ├─ Base:         http://localhost:8000"
echo "   ├─ Health:       http://localhost:8000/health"
echo "   ├─ Info:         http://localhost:8000/info"
echo "   ├─ Generate:     POST http://localhost:8000/generate"
echo "   └─ Docs:         http://localhost:8000/docs"
echo ""
echo "   🟡 Haute Performance (TorchServe):"
echo "   ├─ Inference:    http://localhost:8080/predictions/french-gpt2"
echo "   └─ Management:   http://localhost:8081/models"
echo ""
echo "   🔵 Dashboard (Monitoring):"
echo "   └─ UI:           http://localhost:5000"
echo ""
echo "   🌐 Web Interface:"
echo "   └─ Test UI:      open web_interface.html"
echo ""

echo "╔════════════════════════════════════════════════════════╗"
echo "║              🔍 MONITORER L'ENTRAÎNEMENT              ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "En temps réel:"
echo "   tail -f $TRAINING_LOG"
echo ""
echo "Voir le dernier step:"
echo "   tail -20 $TRAINING_LOG | grep -E 'step|loss'"
echo ""
echo "Tuer l'entraînement si besoin:"
echo "   kill $TRAINING_PID"
echo ""

echo "╔════════════════════════════════════════════════════════╗"
echo "║            🚀 PRÊT À UTILISER! 🚀                    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Progression actuelle:"
show_training_progress
echo ""
