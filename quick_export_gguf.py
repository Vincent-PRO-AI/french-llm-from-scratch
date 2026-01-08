#!/usr/bin/env python3
"""
Script simple pour exporter le checkpoint 500k en GGUF
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 EXPORT GGUF - French LLM 500k")
    print("=" * 70)
    
    checkpoint = Path("trained_models/runs/french_medium_rtx5080_extended/checkpoint_step_500000.pt")
    output_gguf = Path("trained_models/exports/french_medium_500k.gguf")
    
    if not checkpoint.exists():
        # Chercher le dernier checkpoint disponible
        run_dir = Path("trained_models/runs/french_medium_rtx5080_extended")
        if not run_dir.exists():
            print(f"❌ Répertoire non trouvé: {run_dir}")
            return False
        
        checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            print("❌ Aucun checkpoint trouvé!")
            return False
        
        checkpoint = checkpoints[-1]
        print(f"⚠️  Checkpoint 500k non trouvé, utilisation du dernier: {checkpoint}")
    
    print(f"✅ Checkpoint: {checkpoint}")
    print(f"📤 Sortie GGUF: {output_gguf}")
    
    # Créer le répertoire de sortie
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    
    # Lancer l'export
    cmd = [
        sys.executable,
        "export_to_gguf.py",
        str(checkpoint),
        str(output_gguf)
    ]
    
    print(f"\n⏳ Export en cours... (peut prendre 5-10 minutes)")
    print(f"   {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd="/home/vincent/code/repo/french-llm-from-scratch")
    
    if result.returncode == 0 and output_gguf.exists():
        size_gb = output_gguf.stat().st_size / (1024**3)
        print(f"\n✅ Export GGUF réussi!")
        print(f"   Fichier: {output_gguf}")
        print(f"   Taille: {size_gb:.2f} GB")
        
        print(f"\n💡 Utilisation avec LM Studio:")
        print(f"   1. Ouvrir LM Studio")
        print(f"   2. Charger: {output_gguf}")
        print(f"   3. Commencer à tchatter!")
        
        return True
    else:
        print(f"❌ Export GGUF échoué!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
