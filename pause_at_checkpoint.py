#!/usr/bin/env python3
"""
Script pour arrêter l'entraînement au prochain checkpoint
et préparer l'export + validation LM Studio
"""
import time
import subprocess
import signal
import os
from pathlib import Path
from datetime import datetime

def monitor_and_pause():
    """Monitor l'entraînement et arrête au prochain checkpoint."""
    
    print("🔍 MONITORING - Attente du prochain checkpoint")
    print("=" * 70)
    
    run_dir = Path("trained_models/runs/french_medium_rtx5080_extended")
    training_pid = 5211  # PID du processus principal
    
    # Chercher le dernier checkpoint
    def get_latest_checkpoint():
        checkpoints = list(run_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    initial_checkpoint = get_latest_checkpoint()
    if initial_checkpoint:
        initial_step = int(initial_checkpoint.stem.split('_')[-1])
        print(f"✅ Checkpoint initial: {initial_step} steps")
    else:
        print("⚠️  Aucun checkpoint trouvé encore, attendant...")
        initial_step = 0
    
    target_step = initial_step + 5000  # Attendre le prochain (5k steps)
    print(f"🎯 Target: {target_step} steps")
    print(f"⏳ Attente estimée: ~23 minutes à 3.5 steps/sec")
    print()
    
    # Monitor
    check_interval = 60  # Vérifier chaque minute
    while True:
        latest = get_latest_checkpoint()
        
        if latest:
            current_step = int(latest.stem.split('_')[-1])
            if current_step >= target_step:
                print(f"\n✅ CHECKPOINT ATTEINT: {current_step} steps!")
                print(f"📁 Fichier: {latest}")
                print(f"📅 Timestamp: {datetime.now().strftime('%H:%M:%S')}")
                
                # Arrêter proprement
                print(f"\n🛑 Arrêt du processus d'entraînement (PID: {training_pid})...")
                os.kill(training_pid, signal.SIGTERM)
                time.sleep(5)
                
                # Vérifier que c'est arrêté
                result = subprocess.run(["ps", "-p", str(training_pid)], 
                                      capture_output=True)
                if result.returncode == 0:
                    print("⚠️  Le processus ne s'est pas arrêté, forçage...")
                    os.kill(training_pid, signal.SIGKILL)
                    time.sleep(2)
                
                print("✅ Processus arrêté")
                return latest
        
        time.sleep(check_interval)
        elapsed_checkpoint = (current_step - initial_step) if latest else 0
        remaining = target_step - (current_step if latest else initial_step)
        eta_sec = remaining / 3.5 if remaining > 0 else 0
        eta_min = eta_sec / 60
        
        print(f"⏳ {datetime.now().strftime('%H:%M:%S')} - "
              f"Checkpoint: {current_step if latest else 'en cours'}... "
              f"ETA: ~{eta_min:.0f} min")

def prepare_export(checkpoint_path):
    """Prépare l'export GGUF après pause."""
    
    print("\n" + "=" * 70)
    print("📦 PRÉPARATION EXPORT GGUF")
    print("=" * 70)
    
    step = int(checkpoint_path.stem.split('_')[-1])
    
    print(f"\n✅ Checkpoint: {checkpoint_path}")
    print(f"📊 Steps: {step}")
    print(f"💾 Taille: {checkpoint_path.stat().st_size / (1024**3):.2f} GB")
    
    # Créer le répertoire d'export
    export_dir = Path("trained_models/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    output_gguf = export_dir / f"french_medium_{step}k.gguf"
    
    print(f"\n📤 Lancement export GGUF...")
    print(f"   Output: {output_gguf}")
    print(f"   Quantization: Q4_K_M")
    
    cmd = [
        "python3", "export_to_gguf.py",
        str(checkpoint_path),
        str(output_gguf)
    ]
    
    print(f"\n   Commande: {' '.join(cmd)}")
    print(f"   (Peut prendre 5-10 minutes)...\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0 and output_gguf.exists():
        size_gb = output_gguf.stat().st_size / (1024**3)
        print(f"\n✅ Export réussi!")
        print(f"   📁 {output_gguf}")
        print(f"   📊 Taille: {size_gb:.2f} GB")
        return output_gguf
    else:
        print(f"\n❌ Export échoué")
        return None

def validation_plan(checkpoint_path, gguf_path=None):
    """Plan de validation."""
    
    print("\n" + "=" * 70)
    print("✅ PLAN DE VALIDATION")
    print("=" * 70)
    
    step = int(checkpoint_path.stem.split('_')[-1])
    
    print(f"""
Checkpoint de pause:  {step} steps
Status:               ✅ ARRÊTÉ

📋 PROCHAINES ÉTAPES:

1️⃣  EXPORT GGUF
    $ python3 export_to_gguf.py \\
        {checkpoint_path} \\
        trained_models/exports/french_medium_{step}k.gguf

2️⃣  VALIDATION
    a) Vérifier le fichier GGUF existe
       $ ls -lh trained_models/exports/french_medium_{step}k.gguf
    
    b) Charger dans LM Studio
       • Ouvrir LM Studio
       • Charger: trained_models/exports/french_medium_{step}k.gguf
       • Test avec un prompt: "Bonjour, comment vas-tu ?"
    
    c) Test Python
       $ python3 test_gguf_api.py

3️⃣  RELANCER L'ENTRAÎNEMENT (si OK)
    $ bash start_training.sh --resume {step}k --target 500k
    
    Ou manuellement:
    $ python3 scripts/train_subtitles_transformer.py \\
        --resume-from {checkpoint_path} \\
        --arch-preset medium \\
        --max-steps 500000 \\
        --batch-size 16 \\
        ... (autres paramètres)

⏱️  TEMPS ESTIMÉ
═══════════════════════════════════════════════════════════════════════════════
Export GGUF:        ~5-10 minutes
Validation:         ~5-15 minutes
TOTAL PAUSE:        ~20-30 minutes

Après validation OK:
  Steps restants:   ~{500000 - step:,} steps
  Temps estimé:     ~{(500000-step)/3.5/3600:.0f} heures

═══════════════════════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    try:
        # Étape 1: Attendre et arrêter
        checkpoint = monitor_and_pause()
        
        # Étape 2: Préparer export (optionnel ici)
        print("\n" + "=" * 70)
        print("💡 Export GGUF:")
        print("   Exécutez: python3 quick_export_gguf.py")
        print("   OU: python3 export_to_gguf.py <checkpoint_path> <output.gguf>")
        
        # Étape 3: Plan de validation
        validation_plan(checkpoint)
        
    except KeyboardInterrupt:
        print("\n⏹️  Script arrêté par l'utilisateur")
