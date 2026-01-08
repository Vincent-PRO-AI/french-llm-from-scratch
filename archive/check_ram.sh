#!/bin/bash

# SCRIPT: Vérifier et optimiser la RAM
# Usage: ./check_ram.sh

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "🔍 DIAGNOSTIC COMPLET RAM"
echo "════════════════════════════════════════════════════════════════════════════"

echo ""
echo "1️⃣  RAM SYSTÈME ACTUELLE:"
free -h
echo ""

echo "2️⃣  DÉTAILS MEMINFO:"
echo "Total RAM: $(cat /proc/meminfo | grep MemTotal | awk '{print $2 / 1024 / 1024}') GB"
echo "RAM disponible: $(cat /proc/meminfo | grep MemAvailable | awk '{print $2 / 1024 / 1024}') GB"
echo ""

echo "3️⃣  BLOCS MÉMOIRE:"
lsmem 2>/dev/null || echo "lsmem non disponible"
echo ""

echo "4️⃣  LOGS MÉMOIRE DU NOYAU:"
dmesg | grep -i memory | tail -5 || echo "Pas de message de problème mémoire"
echo ""

echo "5️⃣  VÉRIFICATION SLOTS RAM (via /sys):"
ls -la /sys/firmware/dmi/tables/ 2>/dev/null || echo "Info DMI non accessible"
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 RÉSUMÉ:"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Si vous voyez ~96 GB ci-dessus:"
echo "   • RAM complète détectée ✓"
echo "   • Vous pouvez lancer: python training_optimized_batch32.py"
echo ""
echo "⚠️  Si vous voyez seulement ~45 GB:"
echo "   • RAM partiellement détectée"
echo "   • Options:"
echo "      1. Vérifier BIOS (redémarrer en F2/DEL)"
echo "      2. Vérifier enfichage physique des barettes"
echo "      3. Essayer: sudo systemctl restart systemd-logind"
echo "   • Pour maintenant: Utiliser batch_size=24 (5-6x speedup)"
echo ""

