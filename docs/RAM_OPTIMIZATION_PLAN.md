# 🚀 PLAN D'OPTIMISATION RAM - HYPER-V

## 📊 SITUATION ACTUELLE

```
VM Hyper-V détectée:
  • RAM allouée: 45-46 GB
  • RAM max possible: 96 GB
  • Device: AMD Ryzen 7 9700X (16 cores)
  • GPU: NVIDIA RTX 5080 (17 GB VRAM)
  • OS: Linux (Ubuntu)
```

## 🎯 OBJECTIF

Augmenter RAM de **45 GB → 96 GB** pour **10-12x speedup** du training.

## 📈 IMPACT

| Configuration | Batch Size | Steps/sec | 100k steps | 250k steps |
|---------------|-----------|-----------|-----------|-----------|
| **Avant** | 8 | 2 | 13.9 h | 34.7 h |
| **Après (45GB)** | 32 | 10-16 | 1.7-2.3 h | 4.3-5.8 h |
| **Après (96GB)** | 64 | 20 | **1.4 h** | **3.5 h** |

**TOTAL SPEEDUP: 10x** ⚡⚡⚡

## 🎬 ÉTAPES (15 MINUTES)

### ÉTAPE 1: Arrêter la VM (2 min)

**Option A - Depuis la VM:**
```bash
sudo shutdown -h now
```

**Option B - Depuis Windows (Hyper-V Manager):**
- Clic droit sur la VM → Shutdown

### ÉTAPE 2: Ouvrir Hyper-V Manager (1 min)

**Sur Windows (hôte):**
1. Rechercher "Hyper-V Manager"
2. Lancer l'application
3. Voir la liste des VM

### ÉTAPE 3: Augmenter la RAM (2 min)

**Méthode GUI (FACILE):**

1. **Clic droit** sur la VM → **"Settings..."**

2. À GAUCHE: Chercher et cliquer **"Memory"**

3. À DROITE: Voir le champ **"Startup RAM"**
   ```
   ┌─────────────────────┐
   │ Startup RAM: [45 GB]│
   │ ↓ Changer à 96      │
   └─────────────────────┘
   ```

4. **Taper:** `96000` (ou `96384` pour exact)

5. **Cliquer:** "Apply" puis "OK"

---

**Méthode PowerShell (RAPIDE):**

Ouvrir PowerShell (Admin) et taper:

```powershell
# Arrêter la VM
Stop-VM -Name "NOM_DE_LA_VM" -Force

# Augmenter la RAM à 96 GB
Set-VMMemory -VMName "NOM_DE_LA_VM" -StartupBytes 96GB

# Redémarrer
Start-VM -Name "NOM_DE_LA_VM"

# Vérifier
Get-VMMemory -VMName "NOM_DE_LA_VM"
```

> **Note:** Remplace `"NOM_DE_LA_VM"` par le vrai nom (ex: `"french-llm-vm"`)

### ÉTAPE 4: Redémarrer et Vérifier (3 min)

La VM redémarre automatiquement. Attends ~1-2 min.

**Vérifier dans la VM:**
```bash
free -h
```

Tu devrais voir:
```
Mem:     96Gi   ← ✅ (au lieu de 45Gi)
```

### ÉTAPE 5: Lancer le Training (5 min)

Dans la VM, taper:
```bash
cd /home/vincent/code/repo/french-llm-from-scratch
python launch_training_optimized.py
```

Ou directement:
```bash
python scripts/train_subtitles_transformer.py \
  --resume-from trained_models/runs/french_v3_finetune_grand_60k/checkpoint_step_145000.pt \
  --arch-preset medium \
  --batch-size 64 \
  --max-steps 250000 \
  --run-name french_medium_optimized_batch64 \
  --device cuda
```

## ✅ CHECKLIST

- [ ] Arrêter la VM (`shutdown -h now`)
- [ ] Ouvrir Hyper-V Manager
- [ ] Settings → Memory → 96 GB
- [ ] Apply → OK
- [ ] Redémarrer la VM
- [ ] Vérifier: `free -h` (doit afficher ~96Gi)
- [ ] Lancer: `python launch_training_optimized.py`
- [ ] Monitoring: `watch -n 5 free -h` (pour voir la RAM utilisée)

## 🔍 TROUBLESHOOTING

### ❌ Erreur: "Insufficient resources"

**Cause:** L'hôte Windows n'a pas 96 GB libres

**Solution:**
1. Vérifier RAM disponible: `Task Manager → Memory`
2. Si insuffisant:
   - Fermer d'autres VM
   - Redémarrer l'hôte
   - Allouer progressivement (45→64→80→96)

### ❌ Toujours 45 GB après redémarrage?

**Solution:**
```bash
# Dans la VM, redémarrer complètement
sudo shutdown -r now
```

Ou dans PowerShell:
```powershell
Stop-VM -Name "NOM_VM" -Force
Start-VM -Name "NOM_VM"
```

### ❌ Ne peux pas voir "Memory" dans Settings?

**Solution:**
1. Clic droit VM → Settings
2. À GAUCHE, scroller jusqu'à: `[Hardware] → Memory`
3. Ou chercher dans le menu

## 📊 MONITORING

Pendant le training, monitorer l'utilisation RAM:

```bash
# Terminal 1: Training
python launch_training_optimized.py

# Terminal 2: Monitoring (nouveau terminal SSH)
watch -n 5 free -h
```

Ou pour plus de détails:
```bash
htop  # Pour voir détail par processus
```

## 🎯 RÉSUMÉ

| Étape | Action | Temps |
|-------|--------|-------|
| 1 | Arrêter VM | 2 min |
| 2 | Hyper-V Manager | 1 min |
| 3 | Settings: RAM 45→96 GB | 2 min |
| 4 | Redémarrer & vérifier | 3 min |
| 5 | Lancer training batch_64 | 5 min |
| **TOTAL** | **Tout** | **~15 min** |

## 🚀 IMPACT FINAL

```
AVANT:  batch_8   → 2 steps/sec
APRÈS:  batch_64  → 20 steps/sec

SPEEDUP: 10x ⚡⚡⚡

250k steps:
  • Avant: 34.7 heures
  • Après: 3.5 heures
  • Gain: 31 heures économisées! 🎉
```

---

**Questions?** Consulte `HYPERV_RAM_INCREASE_GUIDE.py` pour le guide détaillé avec screenshots.

**Prêt?** Commence par `shutdown -h now` ! 🚀
