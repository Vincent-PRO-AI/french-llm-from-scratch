#!/bin/bash
# 📊 Script de monitoring en temps réel

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║        📊 MONITORING ENTRAÎNEMENT 5000 STEPS           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 1. Vérifier les processus
echo "🔴 Processus d'entraînement actifs:"
ps aux | grep continue_training_5k | grep -v grep | wc -l
echo ""

# 2. Vérifier les logs
echo "📝 Log file size:"
if [ -f training_5k_extended.log ]; then
    du -h training_5k_extended.log
    echo ""
    echo "Dernières lignes:"
    tail -20 training_5k_extended.log
else
    echo "   Fichier non trouvé"
fi
echo ""

# 3. Monitoring GPU
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "   GPU not available"
echo ""

# 4. Vérifier les services
echo "🌐 Services en cours:"
echo "   FastAPI:"
curl -s http://localhost:8000/health 2>/dev/null | python3 -c "import sys, json; print('   ✅ Active' if 'healthy' in sys.stdin.read() else '   ❌ Inactive')" || echo "   ❌ Inactive"

echo ""
echo "📈 CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print "   " $0}'
echo ""
