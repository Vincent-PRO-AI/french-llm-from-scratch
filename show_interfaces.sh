#!/bin/bash
# 🎯 Lance toutes les interfaces de monitoring

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║    🚀 LANCEMENT TOUTES LES INTERFACES DE MONITORING 🚀    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 1. Dashboard Terminal (direct)
echo "1️⃣  Dashboard Terminal en direct"
echo "   Commande: bash dashboard_live.sh"
echo ""

# 2. Web Interface
echo "2️⃣  Web Interface (Browser)"
echo "   Ouvrir: web_interface.html"
echo "   URL: file://$(pwd)/web_interface.html"
echo ""

# 3. API Docs
echo "3️⃣  API Documentation (SwaggerUI)"
echo "   URL: http://localhost:8000/docs"
echo ""

# 4. FastAPI Health
echo "4️⃣  API Health Check"
curl -s http://localhost:8000/health 2>/dev/null | python3 -m json.tool | sed 's/^/   /'
echo ""

# 5. Logs
echo "5️⃣  Logs d'entraînement"
echo "   tail -f training_5k_extended.log"
echo ""

# 6. Monitoring GPU
echo "6️⃣  GPU Monitoring"
echo "   watch -n1 nvidia-smi"
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ PRÊT À UTILISER! ✅                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
