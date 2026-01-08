# 🐍 GUIDE CONDA - French LLM

**Date:** 10 Décembre 2025  
**Environnement:** french-llm (Conda)  
**Python:** 3.10  
**PyTorch:** 2.6+ (CUDA 12.1)  

---

## 🚀 ACTIVATION

### Terminal Bash/Zsh
```bash
conda activate french-llm
```

### VS Code
- Ctrl+Shift+P → "Python: Select Interpreter"
- Choisir: `french-llm (/home/vincent/miniconda3/envs/french-llm/bin/python)`

### Jupyter/IPython
```bash
conda activate french-llm
jupyter notebook
# ou
ipython
```

---

## 📋 VÉRIFIER L'ENVIRONNEMENT

### Status
```bash
# Voir tous les envs
conda env list

# Voir l'env courant
echo $CONDA_DEFAULT_ENV

# Voir les packages installés
conda list
pip list
```

### PyTorch + CUDA
```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"
```

### Dépendances principales
```bash
python3 -c "
import transformers, datasets, accelerate
print(f'Transformers: {transformers.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'Accelerate: {accelerate.__version__}')
"
```

---

## 🔧 INSTALLER PACKAGES SUPPLÉMENTAIRES

### Via Conda
```bash
conda install -c conda-forge <package-name>
```

### Via Pip
```bash
pip install <package-name>
```

### Fichier requirements
```bash
pip install -r requirements.txt
```

---

## 📊 LANCER LES SCRIPTS

### Phase 2 - Data Processing
```bash
conda activate french-llm
python3 scripts/01_process_data.py \
  --dataset wikitext \
  --output-path ./data/processed_mistral
```

### Phase 3 - Training
```bash
conda activate french-llm
python3 scripts/02_train_mistral.py \
  --batch-size 8 \
  --num-epochs 1 \
  --dataset-path ./data/processed_mistral
```

### Phase 4 - Export GGUF
```bash
conda activate french-llm
python3 scripts/03_export_gguf.py \
  --model-path ./models/mistral_tiny \
  --output-path ./models/mistral_tiny.gguf
```

---

## 🐍 UTILISER DANS VS CODE

### Terminal intégré
- Ouvre Terminal dans VS Code (Ctrl+`)
- Il va automatiquement activer l'env Conda
- Tape les commandes normalement

### Exécuter un script
- Ouvre le script Python
- Clique "Run" (play button en haut à droite)
- OU Ctrl+Shift+D (Debug) → Sélectionne "Python: Current File"

### Jupyter dans VS Code
- Crée un fichier `.ipynb`
- VS Code va détecter l'env Conda automatiquement
- Sélectionne le kernel `french-llm`

---

## 📦 STRUCTURE CONDA

```
~/.miniconda3/envs/french-llm/
├── bin/
│   ├── python        ← Exécutable Python
│   ├── pip           ← Package manager
│   ├── jupyter       ← Jupyter
│   └── ...
├── lib/
│   ├── python3.10/site-packages/  ← Packages installés
│   ├── libcuda.so    ← CUDA libraries
│   ├── libcudnn.so   ← cuDNN libraries
│   └── ...
├── include/
│   └── python3.10/   ← Headers Python
└── ...
```

---

## ⚠️ PROBLÈMES COURANTS

### "conda: command not found"
```bash
# Solution: Initialiser conda
conda init bash
source ~/.bashrc
```

### "ModuleNotFoundError: No module named 'torch'"
```bash
# Solution: Vérifier que l'env est activé
conda activate french-llm
python3 -c "import torch; print(torch.__version__)"
```

### GPU pas détecté dans PyTorch
```bash
# Vérifier CUDA
nvidia-smi

# Vérifier dans Python
python3 -c "import torch; print(torch.cuda.is_available())"

# Solution: Réinstaller PyTorch
conda remove pytorch torchvision torchaudio pytorch-cuda -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Manque de mémoire VRAM
```bash
# Réduire batch size
--batch-size 4  # au lieu de 8

# Ou activer gradient checkpointing (voir scripts)
```

---

## 🔄 METTRE À JOUR L'ENV

### Mettre à jour un package
```bash
pip install --upgrade transformers
# ou
conda update transformers -c conda-forge
```

### Mettre à jour tout
```bash
pip install --upgrade pip
conda update --all
```

### Exporter l'env (pour partager)
```bash
conda env export > french-llm-export.yml
```

### Créer from export
```bash
conda env create -f french-llm-export.yml
```

---

## 🗑️ NETTOYER

### Supprimer l'env
```bash
conda env remove -n french-llm
```

### Nettoyer le cache
```bash
conda clean --all
pip cache purge
```

---

## 📚 DOCUMENTATION

- Conda: https://docs.conda.io/
- PyTorch: https://pytorch.org/
- Transformers: https://huggingface.co/docs/transformers/
- Datasets: https://huggingface.co/docs/datasets/

---

**Status:** ✅ Environnement Conda ready!  
**Prochaine étape:** Exécuter `conda activate french-llm` et lancer Phase 2
