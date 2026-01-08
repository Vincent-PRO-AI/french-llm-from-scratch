#!/usr/bin/env python3
"""
French LLM Multi-GPU Decision Tree
Aide pour choisir la meilleure action à prendre

Usage:
    python multi_gpu_decision_tree.py
"""

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def check_gpu_count():
    """Vérifier le nombre de GPUs"""
    import subprocess
    try:
        result = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
        gpu_count = len(result.stdout.strip().split('\n')) if result.stdout else 0
        return gpu_count, result.stdout.strip()
    except:
        return 0, ""

def decision_tree():
    print_header("🚀 FRENCH LLM MULTI-GPU DECISION TREE")
    print("Choisissez votre situation:\n")
    
    gpu_count, gpu_info = check_gpu_count()
    
    print("DÉTECTION GPU ACTUELLE:")
    if gpu_count == 0:
        print("❌ Aucun GPU détecté")
    elif gpu_count == 1:
        print(f"✅ {gpu_count} GPU détecté:")
        for line in gpu_info.split('\n'):
            print(f"   {line}")
    else:
        print(f"✅ {gpu_count} GPUs détectés:")
        for line in gpu_info.split('\n'):
            print(f"   {line}")
    
    print("\n" + "─" * 60)
    print("\nQuelle est votre situation?\n")
    print("[1] Je veux juste valider ma setup")
    print("[2] Je n'ai qu'une RTX 5080 (entraîner avec single GPU)")
    print("[3] J'ai maintenant 2 GPUs (entraîner avec multi-GPU)")
    print("[4] Voir les guides d'optimisation")
    print("[5] Je besoin d'aide pour déboguer")
    print("[0] Quitter\n")
    
    choice = input("Votre choix [0-5]: ").strip()
    
    if choice == "1":
        action_validate()
    elif choice == "2":
        action_single_gpu()
    elif choice == "3":
        action_dual_gpu()
    elif choice == "4":
        action_optimization()
    elif choice == "5":
        action_debugging()
    elif choice == "0":
        print("\n✅ Au revoir!\n")
    else:
        print("❌ Choix invalide")
        decision_tree()

def action_validate():
    print_header("📊 ACTION: Valider Votre Setup")
    print("Exécutez le validateur:\n")
    print("  python validate_multi_gpu_setup.py\n")
    print("Cela va vérifier:")
    print("  ✓ Disponibilité GPU")
    print("  ✓ Configuration Docker")
    print("  ✓ Ressources système (RAM, disque, CPU)")
    print("  ✓ Structure du projet")
    print("  ✓ Connectivité réseau\n")
    print("Après validation, revenez ici pour les actions recommandées.\n")

def action_single_gpu():
    print_header("🔥 ACTION: Entraîner Avec Single GPU (RTX 5080)")
    print("Configuration optimisée pour RTX 5080 seule:\n")
    print("ÉTAPE 1 - Validation (optionnel)")
    print("  python validate_multi_gpu_setup.py\n")
    print("ÉTAPE 2 - Lancer Training")
    print("  docker-compose up --profile training\n")
    print("PERFORMANCE ATTENDUE:")
    print("  • Throughput: ~90 tokens/sec")
    print("  • Batch size: 12")
    print("  • Gradient accumulation: 2")
    print("  • Pour 250k steps: ~77 heures\n")
    print("MONITORING:")
    print("  • Dashboard: http://localhost:5174")
    print("  • API: http://localhost:8000/api/metrics")
    print("  • GPU: nvidia-smi (watch -n 2 nvidia-smi)\n")

def action_dual_gpu():
    print_header("⚡ ACTION: Entraîner Avec Dual GPU (RTX 5080 + RTX 4090)")
    print("Configuration multi-GPU avec DDP:\n")
    print("ÉTAPE 1 - Valider les 2 GPUs")
    print("  nvidia-smi -L")
    print("  # Doit montrer: GPU 0 RTX 5080 et GPU 1 RTX 4090\n")
    print("ÉTAPE 2 - Valider Setup")
    print("  python validate_multi_gpu_setup.py")
    print("  # Tous les checks doivent être ✅\n")
    print("ÉTAPE 3 - Lancer Training")
    print("  chmod +x launch_multi_gpu_training.sh")
    print("  ./launch_multi_gpu_training.sh 250000 12\n")
    print("PERFORMANCE ATTENDUE:")
    print("  • Throughput: ~200 tokens/sec")
    print("  • Effective batch: 48 (12 × 2 × 2)")
    print("  • Pour 250k steps: ~14-15 heures")
    print("  • Speedup vs single GPU: ~2.2x\n")
    print("MONITORING:")
    print("  Terminal 1: tail -f trained_models/runs/french_medium_multi_gpu/training_log.jsonl")
    print("  Terminal 2: watch -n 2 nvidia-smi")
    print("  Browser: http://localhost:5174\n")

def action_optimization():
    print_header("📈 GUIDES D'OPTIMISATION")
    print("Fichiers disponibles:\n")
    print("1. MULTI_GPU_SETUP.md")
    print("   └─ Configuration détaillée, quick start, troubleshooting\n")
    print("2. MULTI_GPU_OPTIMIZATION.md")
    print("   └─ Performance analysis, techniques d'optimisation, debugging\n")
    print("3. MULTI_GPU_DEPLOYMENT_PHASES.md")
    print("   └─ Phases déploiement (single → dual → multi-node)\n")
    print("4. QUICK_START_MULTI_GPU.md")
    print("   └─ Quick reference pour démarrer\n")
    print("CONSEILS RAPIDES:")
    print("  • Batch trop élevé? Réduire à 8 ou 6")
    print("  • Mémoire OOM? Activer gradient checkpointing")
    print("  • Lent sur RTX 5080? Utiliser AMP (mixed precision)")
    print("  • NCCL errors? Vérifier Docker network\n")

def action_debugging():
    print_header("🔧 DEBUGGING COMMON ISSUES")
    print("Quel problème rencontrez-vous?\n")
    print("[1] Erreur NCCL")
    print("[2] Out of Memory (OOM)")
    print("[3] Training lent")
    print("[4] GPU non détecté")
    print("[5] Docker issues")
    print("[0] Retour\n")
    
    choice = input("Votre problème [0-5]: ").strip()
    
    if choice == "1":
        debug_nccl()
    elif choice == "2":
        debug_oom()
    elif choice == "3":
        debug_slow()
    elif choice == "4":
        debug_gpu()
    elif choice == "5":
        debug_docker()
    elif choice == "0":
        decision_tree()
    else:
        print("❌ Choix invalide")
        action_debugging()

def debug_nccl():
    print_header("🔍 DEBUGGING: NCCL Error")
    print("Symptôme: RuntimeError: NCCL operation failed\n")
    print("DIAGNOSTIC:")
    print("1. Vérifier que les 2 GPUs sont visibles:")
    print("   nvidia-smi -L\n")
    print("2. Vérifier Docker network:")
    print("   docker network ls")
    print("   docker network inspect llm-network\n")
    print("3. Tester communication entre containers:")
    print("   docker exec french-llm-training-master ping training-worker\n")
    print("SOLUTIONS:")
    print("  • Rebuild images: docker-compose -f docker-compose.multi-gpu.yml build --no-cache")
    print("  • Redémarrer Docker: systemctl restart docker")
    print("  • Vérifier MASTER_ADDR env var")
    print("  • Activer NCCL debug: export NCCL_DEBUG=INFO\n")
    print("RÉFÉRENCE: MULTI_GPU_OPTIMIZATION.md (section NCCL Errors)\n")

def debug_oom():
    print_header("🔍 DEBUGGING: Out of Memory")
    print("Symptôme: RuntimeError: CUDA out of memory\n")
    print("DIAGNOSTIC:")
    print("1. Vérifier mémoire GPU:")
    print("   nvidia-smi\n")
    print("2. Vérifier allocation lors du training:")
    print("   docker exec french-llm-training-worker nvidia-smi\n")
    print("SOLUTIONS (priority):")
    print("  1. Activer gradient checkpointing (économise 50% mémoire)")
    print("  2. Réduire batch size de 12 à 8 ou 6")
    print("  3. Augmenter gradient accumulation de 2 à 4")
    print("  4. Activer AMP (mixed precision)")
    print("  5. Réduire sequence length de 1024 à 512\n")
    print("DANS docker-compose.multi-gpu.yml:")
    print("  training-worker:")
    print("    command: ... --batch-size 8 --gradient-accumulation-steps 4\n")
    print("RÉFÉRENCE: MULTI_GPU_OPTIMIZATION.md (section Out of Memory)\n")

def debug_slow():
    print_header("🔍 DEBUGGING: Training Lent")
    print("Symptôme: Throughput bas (< 150 tokens/sec avec 2 GPUs)\n")
    print("DIAGNOSTIC:")
    print("1. Vérifier utilisation GPU:")
    print("   watch -n 1 nvidia-smi\n")
    print("2. Vérifier mémoire GPU:")
    print("   nvidia-smi dmon\n")
    print("3. Vérifier thermal throttling:")
    print("   nvidia-smi -q -d PERFORMANCE\n")
    print("CAUSES POSSIBLES:")
    print("  • Thermal throttling sur RTX 5080")
    print("  • Batch size trop élevé")
    print("  • Communication overhead NCCL")
    print("  • Bottleneck mémoire GPU\n")
    print("SOLUTIONS:")
    print("  • Améliorer ventilation / refroidissement")
    print("  • Réduire batch size")
    print("  • Utiliser AMP (mixed precision)")
    print("  • Vérifier NCCL: export NCCL_FAST_ALLREDUCE=1\n")
    print("RÉFÉRENCE: MULTI_GPU_OPTIMIZATION.md (section Hardware Tuning)\n")

def debug_gpu():
    print_header("🔍 DEBUGGING: GPU Non Détecté")
    print("Symptôme: nvidia-smi ne montre pas le GPU\n")
    print("DIAGNOSTIC:")
    print("1. Vérifier NVIDIA drivers:")
    print("   nvidia-smi\n")
    print("2. Si pas d'output, drivers ne sont pas installés:")
    print("   ubuntu-drivers autoinstall\n")
    print("3. Vérifier GPU matériel:")
    print("   lspci | grep -i nvidia\n")
    print("SOLUTIONS:")
    print("  • Installer NVIDIA CUDA Toolkit")
    print("  • Redémarrer machine")
    print("  • Vérifier connexion physique GPU")
    print("  • Vérifier dans BIOS que GPU est enabled\n")
    print("POUR DOCKER:")
    print("  • Installer NVIDIA Container Toolkit")
    print("  • Redémarrer Docker: systemctl restart docker\n")

def debug_docker():
    print_header("🔍 DEBUGGING: Docker Issues")
    print("Symptôme: Docker container fails ou GPU pas visible\n")
    print("SOLUTIONS:")
    print("1. Vérifier Docker daemon:")
    print("   docker ps\n")
    print("2. Vérifier NVIDIA Container Toolkit:")
    print("   docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi\n")
    print("3. Si GPU pas visible, réinstaller toolkit:")
    print("   sudo apt install nvidia-docker2")
    print("   sudo systemctl restart docker\n")
    print("4. Vérifier les logs:")
    print("   docker logs french-llm-training-master\n")
    print("5. Nettoyer et reconstruire:")
    print("   docker system prune -a")
    print("   docker-compose -f docker-compose.multi-gpu.yml build --no-cache\n")

if __name__ == "__main__":
    try:
        decision_tree()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user\n")
