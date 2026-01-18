# 🔧 CONFIGURATION DES ENVIRONNEMENTS DE CALCUL

**Date de vérification:** 17 janvier 2026  
**Système:** WSL 2 Ubuntu sur Windows

---

## 📊 RÉSUMÉ ÉTAT ACTUEL

| Environnement | Status | PyTorch | CUDA | GPU | Recommandation |
|---------------|--------|---------|------|-----|----------------|
| **Conda base** | ✅ **ACTIF** | 2.10.0 (dev) | 12.8 | 2× RTX 4090 | ⭐ **UTILISER** |
| **Conda french-llm** | ✅ Configuré | 2.10.0 (dev) | 12.8 | ✅ | ⭐ Production |
| **Venv .venv** | ⚠️ Incomplet | ❌ | ❌ | ❌ | Nécessite setup |
| **Docker** | ❌ Non disponible | - | - | - | Installer Docker Desktop |

---

## 1️⃣ CONDA (Recommandé ⭐)

### Status: ✅ **FONCTIONNEL ET OPTIMAL**

#### Environnement Actuel: `base`
```bash
Python: 3.13.9
PyTorch: 2.10.0.dev20251208+cu128
CUDA: 12.8
GPUs: 2× NVIDIA GeForce RTX 4090
```

#### Environnement Dédié: `french-llm`
```bash
Python: 3.10.19
PyTorch: 2.10.0.dev20251208+cu128
CUDA: 12.8
GPUs: ✅ Accessible
```

### Packages Vérifiés ✅
```
✅ torch 2.10.0.dev20251208+cu128
✅ transformers 4.57.3
✅ tokenizers 0.22.1 (ou 0.22.2)
✅ datasets 4.4.1
✅ accelerate 1.12.0
```

### Utilisation Recommandée

#### Option A: Conda base (Actuel)
```bash
# Déjà actif - UTILISER DIRECTEMENT
cd /home/vincent/code/repo/french-llm-from-scratch
python scripts/train_subtitles_transformer.py --help
```

#### Option B: Conda french-llm (Production)
```bash
# Activer l'environnement dédié
conda activate french-llm

# Vérifier CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Lancer training
python scripts/train_subtitles_transformer.py \
  --resume-from trained_models/checkpoint_250k_fixed.pt \
  --max-steps 300000
```

### Avantages Conda ⭐
- ✅ **GPU CUDA déjà configuré** (12.8)
- ✅ **2× RTX 4090 détectés**
- ✅ **Tous les packages installés**
- ✅ **Isolation complète** des dépendances
- ✅ **Gestion native CUDA/cuDNN**
- ✅ **Pas de conflits système**

### Mise à Jour Environnement
```bash
# Mettre à jour l'env french-llm
conda activate french-llm
conda update --all

# Ou réinstaller depuis environment.yml
conda env update -f environment.yml --prune
```

---

## 2️⃣ VENV (.venv)

### Status: ⚠️ **INCOMPLET - NÉCESSITE CONFIGURATION**

#### État Actuel
```
Directory: .venv/ existe ✅
Python: 3.10.12 ✅
PyTorch: ❌ NON INSTALLÉ
CUDA: ❌ NON CONFIGURÉ
```

### Setup Complet

#### Étape 1: Créer/Recréer venv
```bash
cd /home/vincent/code/repo/french-llm-from-scratch

# Supprimer ancien venv si incomplet
rm -rf .venv

# Créer nouveau venv
python3.10 -m venv .venv

# Activer
source .venv/bin/activate
```

#### Étape 2: Installer PyTorch avec CUDA
```bash
# PyTorch 2.x avec CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Vérifier installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Étape 3: Installer dépendances projet
```bash
pip install -r requirements.txt
```

#### Étape 4: Validation
```bash
python -c "
import torch
import transformers
import tokenizers
import datasets
print('✅ Environnement venv fonctionnel')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Quand Utiliser Venv?
- ✅ Isolation projet-spécifique
- ✅ Déploiement production (sans Conda)
- ✅ CI/CD pipelines
- ⚠️ Moins pratique pour GPU/CUDA que Conda

### Avantages
- ✅ Standard Python natif
- ✅ Léger (pas de conda)
- ✅ Portable

### Inconvénients
- ❌ Pas de gestion CUDA native
- ❌ Installation manuelle PyTorch
- ❌ Plus difficile multi-versions Python

---

## 3️⃣ DOCKER

### Status: ❌ **NON DISPONIBLE (WSL 2 sans Docker Desktop)**

#### Erreur Actuelle
```
The command 'docker' could not be found in this WSL 2 distro.
We recommend to activate the WSL integration in Docker Desktop settings.
```

### Installation Requise

#### Option A: Docker Desktop (Recommandé Windows)
```bash
# 1. Télécharger Docker Desktop pour Windows
# https://www.docker.com/products/docker-desktop/

# 2. Installer Docker Desktop

# 3. Activer WSL 2 integration
# Settings → Resources → WSL Integration → Enable for "Ubuntu"

# 4. Redémarrer WSL
wsl --shutdown
# Rouvrir Ubuntu
```

#### Option B: Docker natif WSL 2
```bash
# Installation dans WSL (sans Docker Desktop)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Ajouter utilisateur au groupe docker
sudo usermod -aG docker $USER

# Redémarrer session
wsl --shutdown
```

### Configuration Docker Projet

#### Dockerfile (Déjà existant ✅)
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# Python 3.11, PyTorch, transformers...
# VOIR: Dockerfile (52 lignes)
```

#### docker-compose.yml (Déjà existant ✅)
```yaml
services:
  backend: # Flask API (port 8000)
  frontend: # React dashboard (port 5174)
  training: # Training service (GPU)
```

### Utilisation Docker (Après Installation)

#### Démarrer environnement complet
```bash
cd /home/vincent/code/repo/french-llm-from-scratch

# Build et start
docker compose up -d

# Logs
docker compose logs -f backend

# Training
docker compose run --rm training python scripts/train_subtitles_transformer.py --help
```

#### Avantages Docker
- ✅ **Reproductibilité parfaite** (tout inclus)
- ✅ **Isolation complète** (pas de conflits)
- ✅ **Multi-service** (backend + frontend)
- ✅ **Portable** (fonctionne partout)
- ✅ **GPU support** (NVIDIA runtime)

#### Inconvénients
- ❌ **Overhead performance** (~5-10%)
- ❌ **Complexité setup** initial
- ❌ **Taille images** (~5-10 GB)
- ❌ **Pas installé actuellement**

---

## 🎯 RECOMMANDATIONS D'USAGE

### Pour Training Immédiat (MAINTENANT)

#### ✅ **UTILISER: Conda base ou french-llm**

```bash
# Déjà configuré, CUDA fonctionnel, 2× GPU ready
cd /home/vincent/code/repo/french-llm-from-scratch

# Test rapide
python test_model_300k_quick.py

# Training
python scripts/train_subtitles_transformer.py \
  --resume-from trained_models/checkpoint_250k_fixed.pt \
  --max-steps 300000
```

**Pourquoi?**
- ✅ Déjà installé et validé
- ✅ CUDA 12.8 fonctionnel
- ✅ 2× RTX 4090 détectés
- ✅ Tous packages installés
- ✅ 0 setup requis

---

### Pour Production / Déploiement

#### Option 1: Conda (Recommandé)
```bash
# Export exact environment
conda env export > environment_exact.yml

# Sur nouvelle machine
conda env create -f environment_exact.yml
conda activate french-llm
```

#### Option 2: Venv (Standard Python)
```bash
# Setup venv complet (voir section 2)
source .venv/bin/activate
pip install -r requirements.txt
```

#### Option 3: Docker (Si disponible)
```bash
# Build une fois
docker compose build

# Deploy partout
docker compose up -d
```

---

### Pour Développement Multi-Projets

#### Stratégie Recommandée: **Conda Environments**

```bash
# Projet 1: French LLM
conda activate french-llm

# Projet 2: Autre projet
conda create -n autre-projet python=3.11
conda activate autre-projet

# Retour à base
conda deactivate
```

---

## 🔧 COMMANDES UTILES

### Conda
```bash
# Lister environments
conda env list

# Activer
conda activate french-llm

# Désactiver
conda deactivate

# Supprimer
conda env remove -n french-llm

# Export
conda env export > environment.yml

# Import/Recréer
conda env create -f environment.yml
```

### Venv
```bash
# Créer
python3 -m venv .venv

# Activer (Linux/Mac)
source .venv/bin/activate

# Désactiver
deactivate

# Supprimer
rm -rf .venv

# Freeze dependencies
pip freeze > requirements.txt
```

### Docker
```bash
# Build
docker compose build

# Start all
docker compose up -d

# Stop all
docker compose down

# Logs
docker compose logs -f [service]

# Shell dans container
docker compose exec backend bash

# Run training
docker compose run --rm training python scripts/train.py
```

---

## 🐛 TROUBLESHOOTING

### Problème: "CUDA not available"

#### Conda
```bash
# Réinstaller PyTorch avec CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### Venv
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Problème: "ImportError: No module named X"

```bash
# Conda
conda install <package>

# Venv
source .venv/bin/activate
pip install <package>
```

### Problème: "CUDA version mismatch"

```bash
# Vérifier version CUDA système
nvcc --version

# Installer PyTorch compatible
# CUDA 12.1 → cu121
# CUDA 11.8 → cu118
```

---

## 📈 PERFORMANCE COMPARÉE

| Environnement | Setup Time | Training Speed | Overhead | Isolation |
|---------------|------------|----------------|----------|-----------|
| **Conda** | 5 min | 100% ⚡ | 0% | ⭐⭐⭐⭐⭐ |
| **Venv** | 10 min | 100% ⚡ | 0% | ⭐⭐⭐⭐ |
| **Docker** | 30 min | 95% | 5% | ⭐⭐⭐⭐⭐ |

---

## ✅ ÉTAT VALIDATION

### Conda base (Actuel) ✅
- [x] Python fonctionnel
- [x] PyTorch installé
- [x] CUDA accessible (12.8)
- [x] GPUs détectés (2× RTX 4090)
- [x] Transformers installé
- [x] Tous imports critiques OK

### Conda french-llm ✅
- [x] Environment créé
- [x] Python 3.10.19
- [x] PyTorch avec CUDA
- [x] Packages installés

### Venv .venv ⚠️
- [x] Directory existe
- [x] Python 3.10.12
- [ ] PyTorch ABSENT
- [ ] CUDA non configuré
- [ ] Nécessite setup complet

### Docker ❌
- [ ] Docker non installé
- [ ] Docker Compose absent
- [ ] Nécessite Docker Desktop

---

## 🎯 ACTION IMMÉDIATE

### ✅ **PRÊT À UTILISER MAINTENANT:**
```bash
# Conda base est déjà optimal
cd /home/vincent/code/repo/french-llm-from-scratch
python test_model_300k_quick.py
```

### 🔧 **Setup optionnel (si besoin venv):**
```bash
# Compléter .venv (15 minutes)
rm -rf .venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 📦 **Setup Docker (si déploiement):**
```bash
# Installer Docker Desktop
# https://www.docker.com/products/docker-desktop/
# Puis redémarrer WSL
```

---

**Recommendation finale:** ⭐ **UTILISER CONDA (déjà configuré et optimal)**
