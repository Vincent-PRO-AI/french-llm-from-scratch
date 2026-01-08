#!/usr/bin/env python3
"""
Stratégie d'optimisation RÉVISÉE avec 45-96 GB RAM disponible.
Recalcule les limites en fonction de la RAM réellement utilisable.
"""

import json
from datetime import datetime


def generate_revised_strategy():
    """Générer une stratégie révisée basée sur 45-96 GB."""
    
    print("\n" + "="*80)
    print("🎯 STRATÉGIE D'OPTIMISATION RÉVISÉE (45-96 GB)")
    print("="*80)
    
    scenarios = {
        'scenario_conservative': {
            'name': 'Conservative (45 GB confirmés)',
            'ram_available': 45,
            'ram_for_training': 35,  # Garder 10 GB de marge
            'batch_size': 24,
            'gradient_accumulation': 1,
            'amp_enabled': True,
            'data_cache': True,
            'speedup': '5-6x',
            'steps_per_sec': 10,
            'time_100k_steps': '2.8 hours',
            'notes': 'Configuration safe, pas de risque OOM'
        },
        'scenario_aggressive': {
            'name': 'Aggressive (96 GB si disponible)',
            'ram_available': 96,
            'ram_for_training': 80,  # Peut utiliser plus
            'batch_size': 64,
            'gradient_accumulation': 1,
            'amp_enabled': True,
            'data_cache': True,
            'speedup': '10-12x',
            'steps_per_sec': 20,
            'time_100k_steps': '1.4 hours',
            'notes': 'Si 96 GB sont accessibles - configuration max performance'
        }
    }
    
    for scenario_key, scenario in scenarios.items():
        print(f"\n📊 {scenario['name']}")
        print(f"   RAM disponible: {scenario['ram_available']} GB")
        print(f"   RAM pour training: {scenario['ram_for_training']} GB")
        print(f"   Batch size: {scenario['batch_size']}")
        print(f"   Speedup: {scenario['speedup']}")
        print(f"   100k steps: {scenario['time_100k_steps']}")
    
    return scenarios


def create_extended_strategy():
    """Créer un plan pour vérifier et activer la RAM."""
    
    print("\n" + "="*80)
    print("🔍 PLAN POUR VÉRIFIER/ACTIVER 96 GB RAM")
    print("="*80)
    
    plan = {
        'timestamp': datetime.now().isoformat(),
        'current_state': {
            'detected_ram': '45-46 GB',
            'physical_ram': '96 GB (2x 48 GB assumé)',
            'gap': '~50 GB non reconnue'
        },
        'possible_causes': [
            'BIOS: RAM non reconnue ou désactivée',
            'Kernel: RAM non mappée correctement',
            'Virtualization: Limite de mémoire imposée',
            'Slots: RAM non enfichées correctement'
        ],
        'diagnostic_steps': [
            {
                'step': 1,
                'action': 'Vérifier les slots physiques',
                'commands': [
                    'Redémarrer et vérifier dans BIOS',
                    'Chercher "Memory", "DIMM", "RAM speed"',
                    'Vérifier que 2x48 GB apparaissent'
                ]
            },
            {
                'step': 2,
                'action': 'Vérifier le BIOS',
                'commands': [
                    'Chercher "Memory Configuration"',
                    'Désactiver "Memory Hole" si activé',
                    'Vérifier "XMP/DOCP" (profil RAM haute perf)',
                    'Sauvegarder et redémarrer'
                ]
            },
            {
                'step': 3,
                'action': 'Vérifier Linux après redémarrage',
                'commands': [
                    'free -h  (devrait afficher ~96 GB)',
                    'cat /proc/meminfo | grep MemTotal',
                    'dmesg | grep -i memory  (pour les erreurs)'
                ]
            },
            {
                'step': 4,
                'action': 'Si toujours 45 GB seulement',
                'commands': [
                    'Vérifier les logs BIOS pour warnings',
                    'Tester chaque barrette RAM individuellement',
                    'Réenficher les RAM correctement',
                    'Consulter le manuel de la carte mère'
                ]
            }
        ],
        'immediate_actions': [
            '1. Lancer diagnostic sur 45 GB actuels',
            '2. Relancer training avec batch_size=24 (5-6x speedup)',
            '3. En parallèle, investiguer la RAM manquante',
            '4. Si 96 GB trouvés → redémarrer et passer à batch_size=64 (10-12x)'
        ]
    }
    
    return plan


def main():
    print("\n" + "="*80)
    print("📋 ANALYSE: 45 GB DÉTECTÉS vs 96 GB ANNONCÉS")
    print("="*80)
    
    # Scénarios
    scenarios = generate_revised_strategy()
    
    # Plan d'action
    plan = create_extended_strategy()
    
    # Afficher le plan
    print(f"\n🔧 ÉTAPES DE DIAGNOSTIC:")
    for step in plan['diagnostic_steps']:
        print(f"\n   Étape {step['step']}: {step['action']}")
        for cmd in step['commands']:
            print(f"      → {cmd}")
    
    print(f"\n⚡ ACTIONS IMMÉDIALES:")
    for action in plan['immediate_actions']:
        print(f"   {action}")
    
    # Sauvegarder
    full_report = {
        'timestamp': datetime.now().isoformat(),
        'current_ram': 45.8,  # GB
        'claimed_ram': 96.0,  # GB
        'scenarios': scenarios,
        'diagnostic_plan': plan
    }
    
    with open('ram_optimization_extended.json', 'w') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Rapport sauvegardé: ram_optimization_extended.json")
    
    # Afficher le résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ COMPARATIF")
    print("="*80)
    
    comparison = {
        'configuration': ['Current (45GB)', 'Conservative (45GB)', 'Aggressive (96GB)'],
        'batch_size': [8, 24, 64],
        'speedup': ['1x (baseline)', '5-6x', '10-12x'],
        'steps_per_sec': [2, 10, 20],
        '100k_steps_time': ['12.4 hours', '2.8 hours', '1.4 hours']
    }
    
    for i, config in enumerate(comparison['configuration']):
        print(f"\n{config}:")
        print(f"   Batch size: {comparison['batch_size'][i]}")
        print(f"   Speedup: {comparison['speedup'][i]}")
        print(f"   Speed: {comparison['steps_per_sec'][i]} steps/sec")
        print(f"   100k steps: {comparison['100k_steps_time'][i]}")
    
    return full_report


if __name__ == '__main__':
    main()
