#!/usr/bin/env python3
"""
GUIDE: Augmenter la RAM de la VM Hyper-V de 45 GB → 96 GB
Avec images et étapes détaillées
"""

import json
from datetime import datetime


def create_hyperv_ram_guide():
    """Guide complet pour augmenter la RAM."""
    
    guide = {
        'timestamp': datetime.now().isoformat(),
        'objective': 'Augmenter RAM Hyper-V de 45 GB → 96 GB',
        'estimated_time': '15 minutes',
        'difficulty': 'FACILE ✓',
        
        'method_1_gui': {
            'title': '🖱️ Méthode 1: GUI Hyper-V Manager (FACILE)',
            'steps': [
                {
                    'number': 1,
                    'title': 'Arrêter la VM',
                    'details': [
                        'Sur la VM Linux:',
                        '  $ sudo shutdown -h now',
                        '',
                        'Ou: Cliq droit sur la VM → Shutdown',
                        '',
                        '⏳ Attendre l\'arrêt complet (~30 sec)'
                    ]
                },
                {
                    'number': 2,
                    'title': 'Ouvrir Hyper-V Manager',
                    'details': [
                        'Sur Windows (hôte):',
                        '  • Rechercher "Hyper-V Manager"',
                        '  • Lancer l\'application',
                        '',
                        '⚠️ Assurez-vous que Hyper-V est activé:',
                        '  Panneau de configuration → Fonctionnalités',
                        '  → Hyper-V ✓'
                    ]
                },
                {
                    'number': 3,
                    'title': 'Accéder aux paramètres de la VM',
                    'details': [
                        'Dans Hyper-V Manager:',
                        '  1. Clic DROIT sur votre VM',
                        '  2. Choisir "Settings..."',
                        '',
                        '💡 Screenshot:',
                        '   ┌─────────────────────┐',
                        '   │ VM Name (arrêtée)   │',
                        '   ├─────────────────────┤',
                        '   │ ▶ Start             │',
                        '   │ ⏸ Checkpoint        │',
                        '   │ Settings... ← CLIC  │',
                        '   └─────────────────────┘'
                    ]
                },
                {
                    'number': 4,
                    'title': 'Modifier la RAM',
                    'details': [
                        'Dans la fenêtre Settings:',
                        '  1. À GAUCHE: Chercher "Memory"',
                        '  2. Cliquer sur Memory',
                        '',
                        '  3. À DROITE: Voir:',
                        '     ┌──────────────────────────┐',
                        '     │ Startup RAM: 45 GB       │',
                        '     │ (ou le montant actuel)   │',
                        '     │                          │',
                        '     │ [Changer à 96]           │',
                        '     │                          │',
                        '     │ Minimum RAM: 256 MB      │',
                        '     │ Maximum RAM: 1 TB        │',
                        '     └──────────────────────────┘'
                    ]
                },
                {
                    'number': 5,
                    'title': 'Changer le nombre',
                    'details': [
                        '1. Cliquer dans le champ "Startup RAM"',
                        '2. Supprimer le nombre actuel',
                        '3. Taper: 96384',
                        '   (96 GB en MB = 96 * 1024 = 98304)',
                        '   OU simplement: 96000',
                        '',
                        '⚠️ NOTE:',
                        '   • 96 GB = 96 * 1024 = 98,304 MB',
                        '   • Tu peux taper 96000 (arrondi)',
                        '   • Hyper-V ajustera automatiquement'
                    ]
                },
                {
                    'number': 6,
                    'title': 'Appliquer les changements',
                    'details': [
                        'En bas à droite de la fenêtre:',
                        '  ✓ Vérifier que tu vois "Apply" ou "OK"',
                        '  ✓ Cliquer sur "Apply" puis "OK"',
                        '',
                        '✅ Les changements sont sauvegardés'
                    ]
                },
                {
                    'number': 7,
                    'title': 'Redémarrer la VM',
                    'details': [
                        'Dans Hyper-V Manager:',
                        '  1. Clic DROIT sur la VM',
                        '  2. Cliquer "Start"',
                        '',
                        '⏳ Attendre le démarrage (~1-2 min)'
                    ]
                },
                {
                    'number': 8,
                    'title': 'Vérifier la nouvelle RAM',
                    'details': [
                        'Dans la VM (Linux):',
                        '  $ free -h',
                        '',
                        'Tu devrais voir:',
                        '  Mem:    96Gi   (au lieu de 45Gi)',
                        '',
                        '✅ Succès! RAM augmentée.'
                    ]
                }
            ]
        },
        
        'method_2_powershell': {
            'title': '⚡ Méthode 2: PowerShell (RAPIDE)',
            'steps': [
                {
                    'number': 1,
                    'title': 'Arrêter la VM',
                    'command': 'Stop-VM -Name "VM_NAME" -Force'
                },
                {
                    'number': 2,
                    'title': 'Augmenter la RAM',
                    'command': 'Set-VMMemory -VMName "VM_NAME" -StartupBytes 96GB'
                },
                {
                    'number': 3,
                    'title': 'Redémarrer la VM',
                    'command': 'Start-VM -Name "VM_NAME"'
                },
                {
                    'number': 4,
                    'title': 'Vérifier',
                    'command': 'Get-VMMemory -VMName "VM_NAME"'
                }
            ],
            'note': 'Remplace "VM_NAME" par le vrai nom (ex: "french-llm-training")'
        },
        
        'troubleshooting': {
            'problem_1': {
                'issue': 'Erreur: "Insufficient resources"',
                'cause': 'L\'hôte n\'a pas assez de RAM libre',
                'solution': [
                    '1. Vérifier RAM disponible sur Windows:',
                    '   Task Manager → Performance → Memory',
                    '',
                    '2. Si < 96 GB libre:',
                    '   • Fermer d\'autres VM',
                    '   • Redémarrer l\'hôte',
                    '   • Allouer progressivement (45→64→96)'
                ]
            },
            'problem_2': {
                'issue': 'Hyper-V ne reconnait pas la RAM après redémarrage',
                'cause': 'Hyper-V balloon driver',
                'solution': [
                    '1. Dans la VM Linux:',
                    '   $ free -h',
                    '',
                    '2. Si encore 45 GB:',
                    '   $ sudo systemctl restart systemd-logind',
                    '',
                    '3. Ou redémarrer la VM complètement:',
                    '   $ sudo shutdown -r now'
                ]
            },
            'problem_3': {
                'issue': 'Je peux pas voir "Memory" dans Settings',
                'cause': 'Mauvaise génération Hyper-V',
                'solution': [
                    '1. Clic droit VM → Settings',
                    '2. À GAUCHE, chercher:',
                    '   [Hardware] → Memory',
                    '',
                    '3. Ou scroller dans la liste à gauche'
                ]
            }
        },
        
        'after_increase': {
            'title': '✅ Après l\'augmentation',
            'actions': [
                {
                    'title': 'Vérifier la RAM',
                    'command': 'free -h'
                },
                {
                    'title': 'Activer le training optimisé',
                    'command': 'python launch_training_batch64.py'
                },
                {
                    'title': 'Expectation',
                    'details': [
                        'Avant: batch_8  → 2 steps/sec   → 100k steps = 13.9 hours',
                        'Après: batch_64 → 20 steps/sec  → 100k steps = 1.4 hours',
                        '',
                        'Speedup: 10x PLUS RAPIDE ⚡⚡⚡'
                    ]
                }
            ]
        },
        
        'quick_reference': {
            'title': '🚀 Quick Reference',
            'checklist': [
                '[ ] Arrêter la VM',
                '[ ] Ouvrir Hyper-V Manager',
                '[ ] Clic droit VM → Settings',
                '[ ] Memory → Startup RAM: 96GB',
                '[ ] Apply → OK',
                '[ ] Redémarrer la VM',
                '[ ] Vérifier: free -h (doit afficher 96Gi)',
                '[ ] Lancer training optimisé'
            ]
        }
    }
    
    return guide


def main():
    guide = create_hyperv_ram_guide()
    
    print("\n" + "="*80)
    print("📖 GUIDE COMPLET: AUGMENTER RAM HYPER-V")
    print("="*80)
    
    print(f"\n🎯 Objectif: {guide['objective']}")
    print(f"⏱️  Temps estimé: {guide['estimated_time']}")
    print(f"📊 Difficulté: {guide['difficulty']}")
    
    # Méthode 1
    print(f"\n\n" + "="*80)
    print(f"{guide['method_1_gui']['title']}")
    print("="*80)
    
    for step in guide['method_1_gui']['steps']:
        print(f"\n📍 Étape {step['number']}: {step['title']}")
        for detail in step['details']:
            print(f"   {detail}")
    
    # Méthode 2
    print(f"\n\n" + "="*80)
    print(f"{guide['method_2_powershell']['title']}")
    print("="*80)
    
    for step in guide['method_2_powershell']['steps']:
        print(f"\n{step['number']}. {step['title']}")
        print(f"   > {step['command']}")
    
    print(f"\n⚠️  Note: {guide['method_2_powershell']['note']}")
    
    # Troubleshooting
    print(f"\n\n" + "="*80)
    print(f"🔧 TROUBLESHOOTING")
    print("="*80)
    
    for key, issue in guide['troubleshooting'].items():
        print(f"\n❌ {issue['issue']}")
        print(f"   Cause: {issue['cause']}")
        print(f"   Solution:")
        for sol in issue['solution']:
            print(f"      {sol}")
    
    # Quick reference
    print(f"\n\n" + "="*80)
    print(f"✅ {guide['quick_reference']['title']}")
    print("="*80)
    
    for item in guide['quick_reference']['checklist']:
        print(f"   {item}")
    
    # Sauvegarder
    with open('HYPERV_RAM_INCREASE_GUIDE.json', 'w') as f:
        json.dump(guide, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n📄 Guide complet sauvegardé: HYPERV_RAM_INCREASE_GUIDE.json")
    
    print(f"\n" + "="*80)
    print(f"🎬 PROCHAINES ÉTAPES:")
    print("="*80)
    print(f"""
1️⃣  ARRÊTER la VM:
   $ sudo shutdown -h now

2️⃣  HYPER-V MANAGER (sur Windows):
   • Ouvrir Hyper-V Manager
   • Settings → Memory: 45 GB → 96 GB
   • Apply → OK

3️⃣  REDÉMARRER la VM et vérifier:
   $ free -h  (devrait afficher 96Gi)

4️⃣  CONTINUER le training avec la nouvelle config:
   $ python launch_training_batch64.py
""")


if __name__ == '__main__':
    main()
