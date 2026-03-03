# 🧬 Pipelines All-in-One pour Design de Peptides par IA

## 1. **BindCraft** ⭐ (Recommandé pour RTX)
- **Tâche** : Design → Docking → Validation complète
- **GPU VRAM** : 12-16GB (RTX 4090 optimal)
- **Stack** : PyTorch + RDKit + SMINA
- **Déploiement** : Docker / Local / API
- **Avantages** :
  - Pipeline complet en un seul tool
  - Génération + docking + scoring intégré
  - Optimisé pour petites molécules ET peptides
  - Support GPU AMD/NVIDIA
- **Repo** : https://github.com/AIWRL/BindCraft
- **Setup** :
  ```bash
  git clone https://github.com/AIWRL/BindCraft
  cd BindCraft
  pip install -r requirements.txt
  python inference.py --design peptide --target protein.pdb
  ```

## 2. **RFDiffusion + OmegaFold** (Structure-First)
- **Tâche** : Générer structure → Prédire backbone → Refine séquence
- **GPU VRAM** : 16-24GB (RTX 4090 needed)
- **Stack** : PyTorch + Huggingface Diffusers
- **Avantages** :
  - Design génératif par diffusion
  - Contrôle structurel précis
  - Fast inference (~30s/peptide)
- **Repo** : https://github.com/RosettaCommons/RFdiffusion
- **Pipeline type** :
  ```
  RFDiffusion (gen structure) 
  → OmegaFold (predict structure)
  → ProteinMPNN (design sequence)
  → ESMFold (validate)
  ```

## 3. **Chai-1** (Newest - All-in-One Structure)
- **Tâche** : Structure prediction pour peptides + complexes
- **GPU VRAM** : 10GB (très efficace)
- **Modèle** : Single model pour tout
- **Avantages** :
  - Plus rapide qu'AlphaFold2
  - Meilleur pour peptides que ESMFold
  - Integrated MSA + structure
- **Repo** : https://github.com/chaidiscovery/chai-1
- **Setup** :
  ```bash
  pip install chai-lab
  chai predict --sequence MKTIIALSYIFCLVFADYKDDDG --output results.pdb
  ```

## 4. **DockingKit + Diffusion** (Custom Pipelines)
- **Tâche** : Combine multiple models for full workflow
- **GPU VRAM** : 12GB min
- **Stack** : DiffDock (docking) + ESMFold (structure) + ProteinMPNN (sequence)
- **Avantages** :
  - Highly customizable
  - Mix best models
  - Modular architecture
- **Repo** : https://github.com/gcorso/DiffDock
- **Pipeline sample** :
  ```python
  # 1. Generate structure
  structure = esm_fold(sequence)
  # 2. Dock to target
  complex_struct = diffdock(structure, target_pdb)
  # 3. Score + refine
  score = run_rosetta_fast(complex_struct)
  # 4. Design variants
  new_seq = protein_mpnn(structure, constraints)
  ```

## 5. **AlphaFold2 + ProteinMPNN** (Classic Combo - Proven)
- **Tâche** : Structure → Sequence design
- **GPU VRAM** : 11GB (RTX 4090) / 8GB (RTX 5080)
- **Stack** : LocalColabFold (optimized AF2) + ProteinMPNN
- **Avantages** :
  - Très mature et stable
  - Best for accuracy
  - Excellent peptide design
- **Setup** :
  ```bash
  # LocalColabFold (faster AF2)
  git clone https://github.com/YoshitakaMo/localcolabfold
  cd localcolabfold && bash install.sh
  
  # ProteinMPNN
  git clone https://github.com/dauparas/ProteinMPNN
  ```

## Comparaison Rapide

| Tool | Speed | Accuracy | VRAM | For Peptides | Maturity |
|------|-------|----------|------|--------------|----------|
| BindCraft | ⚡⚡ | ⭐⭐⭐⭐ | 12GB | ✅ Excellent | ⭐⭐⭐⭐ |
| RFDiffusion | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 16GB | ✅ Excellent | ⭐⭐⭐⭐ |
| Chai-1 | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 10GB | ✅ Good | ⭐⭐⭐ |
| DockingKit | ⚡⚡ | ⭐⭐⭐⭐⭐ | 12GB | ✅ Great | ⭐⭐⭐ |
| AF2+MPNN | ⚡ | ⭐⭐⭐⭐⭐ | 11GB | ✅ Excellent | ⭐⭐⭐⭐⭐ |

## 🎯 Recommandation pour RTX

### Scénario 1: Speed Priority (30s/peptide)
```
Chai-1 standalone
└─ Output: Structure PDB + reliability metrics
   └─ Optional: DiffDock for binding validation
```

### Scénario 2: Accuracy Priority (5min/peptide)
```
AlphaFold2 (LocalColabFold)
├─ Input: Sequence + optional templates
├─ Output: Structure PDB + pLDDT confidence
└─ ProteinMPNN
   └─ Input: Structure
   └─ Output: Optimized sequences (variants)
```

### Scénario 3: Full Design Pipeline (10min/peptide)
```
RFDiffusion (generative design)
├─ Input: Target pocket / constraints
├─ Output: Novel structure scaffold
└─ OmegaFold (validate)
   ├─ ProteinMPNN (design sequence)
   └─ DiffDock (binding validation)
      └─ Output: Best designs ranked by binding affinity
```

### Scénario 4: BindCraft (Integrated - Recommended)
```
BindCraft End-to-End
├─ 1. Generate peptide candidates (diffusion/GNN)
├─ 2. Predict 3D structures (ESMFold-like)
├─ 3. Dock to target (integrated docking)
├─ 4. Score binding affinity
└─ 5. Output: Ranked designs with PDB + metrics
```

## 🚀 Déploiement sur RTX - Recommandé

### Option A: Docker All-in-One
```dockerfile
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# BindCraft + supporting tools
RUN git clone https://github.com/AIWRL/BindCraft && cd BindCraft && pip install -r requirements.txt
RUN pip install omegafold chai-lab

EXPOSE 8000
CMD ["python", "api_server.py"]
```

### Option B: FastAPI Wrapper (Production)
```python
from fastapi import FastAPI
from bindcraft.inference import PeptideDesigner

app = FastAPI()
designer = PeptideDesigner(gpu_id=0, batch_size=4)

@app.post("/design")
async def design_peptide(target_pdb: str, num_candidates: int = 5):
    results = designer.generate_and_dock(target_pdb, num_candidates)
    return {"designs": results}

@app.post("/validate")
async def validate(pdb_files: list):
    scores = designer.validate_structures(pdb_files)
    return {"scores": scores}
```

### Option C: Batch Processing
```bash
# Design 1000 peptides on RTX 4090
python batch_design.py \
  --target protein.pdb \
  --num_designs 1000 \
  --batch_size 4 \
  --gpu 0 \
  --method bindcraft
# Output: candidates_ranked.csv + pdb_files/
```

## ⚡ Performance Estimée (RTX 4090)

| Pipeline | Time/Peptide | Throughput | VRAM Used |
|----------|-------------|-----------|-----------|
| Chai-1 | 15s | 240/h | 8GB |
| BindCraft | 45s | 80/h | 14GB |
| AF2+MPNN | 5min | 12/h | 11GB |
| RFDiffusion | 1.5min | 40/h | 18GB |

## 💾 Modèles à Pré-charger

```bash
# Télécharger les poids (first time)
python -c "from chai_lab import Chai; Chai.download_weights()"
python -c "from RFdiffusion import get_model; get_model('RF_500M')"
python -c "from ProteinMPNN import get_checkpoint; get_checkpoint()"
```

## Quelle approche préférez-vous ?
1. **BindCraft** (tout-en-un, simple)
2. **RFDiffusion** (plus créatif/génératif)
3. **Chai-1** (plus rapide, moins VRAM)
4. **Custom pipeline** (AF2 + ProteinMPNN + DiffDock)

Je peux setup l'une de ces options immédiatement sur votre RTX !
