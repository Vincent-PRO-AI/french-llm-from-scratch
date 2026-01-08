# 🤖 Modèles Français Disponibles et Testables

**Mis à jour**: 2026-01-07

---

## 📊 Résumé Exécutif

Vous avez **6 modèles** entraînés avec **42 checkpoints** disponibles.

### 🌟 Meilleurs Modèles Recommandés

| Rang | Modèle | Steps | Status | Qualité |
|------|--------|-------|--------|---------|
| 🥇 | **V3 Finetune Batch4** | 250,000 | ✅ Complet | ⭐⭐⭐⭐⭐ |
| 🥈 | **V3 Resample** | 160,000 | ✅ Complet | ⭐⭐⭐⭐ |
| 🥉 | **V3 Grand Modèle** | 100,000 | ✅ Complet | ⭐⭐⭐⭐ |

---

## 1. V3 Finetune Batch4 (MEILLEUR 🌟)

**Architecture**: TinyTransformerLM (260M paramètres)
- `d_model`: 1024
- `num_layers`: 18
- `num_heads`: 16
- `ff_hidden_dim`: 4096
- `vocab_size`: 32,000 (Mistral tokenizer)

**Training**:
- Steps: **150,000 → 250,000** (11 checkpoints)
- Block size: 1024
- Batch size: 4
- Data: 100% Français (conversations)

**Checkpoints Disponibles**:
```
trained_models/runs/french_medium_optimized_batch4_grad3/
├── checkpoint_step_150000.pt (3.0 GB)
├── checkpoint_step_160000.pt (3.0 GB)
├── checkpoint_step_170000.pt (3.0 GB)
├── checkpoint_step_180000.pt (3.0 GB)
├── checkpoint_step_190000.pt (3.0 GB)
├── checkpoint_step_200000.pt (3.0 GB)
├── checkpoint_step_210000.pt (3.0 GB)
├── checkpoint_step_220000.pt (3.0 GB)
├── checkpoint_step_230000.pt (3.0 GB)
├── checkpoint_step_240000.pt (3.0 GB)
└── checkpoint_step_250000.pt (3.0 GB) ← RECOMMANDÉ
```

**Usage**:
```bash
python3 scripts/test_inference_v3.py \
  --checkpoint trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt \
  --prompt "Bonjour, comment allez-vous ?"
```

---

## 2. V3 Grand Modèle (STABLE ⭐)

**Architecture**: TinyTransformerLM (260M paramètres)
- Même config que V3 Finetune Batch4

**Training**:
- Steps: **10,000 → 100,000** (10 checkpoints)
- Run: `french-llm-from-scratch-V3-mistral`
- Status: ✅ Complet et stable

**Checkpoints Disponibles**:
```
trained_models/runs/french-llm-from-scratch-V3-mistral/
├── checkpoint_step_10000.pt (3.0 GB)
├── checkpoint_step_20000.pt (3.0 GB)
├── checkpoint_step_30000.pt (3.0 GB)
├── checkpoint_step_40000.pt (3.0 GB)
├── checkpoint_step_50000.pt (3.0 GB)
├── checkpoint_step_60000.pt (3.0 GB)
├── checkpoint_step_70000.pt (3.0 GB)
├── checkpoint_step_80000.pt (3.0 GB)
├── checkpoint_step_90000.pt (3.0 GB)
└── checkpoint_step_100000.pt (3.0 GB) ← RECOMMANDÉ
```

---

## 3. V3 Resample (EXPÉRIMENTAL 🧪)

**Architecture**: TinyTransformerLM (260M paramètres)

**Training**:
- Steps: **110,000 → 160,000** (6 checkpoints)
- Variant avec resampling des données

**Checkpoints**:
```
trained_models/runs/french_v3_finetune_resample_1/
├── checkpoint_step_110000.pt (3.0 GB)
├── checkpoint_step_120000.pt (3.0 GB)
├── checkpoint_step_130000.pt (3.0 GB)
├── checkpoint_step_140000.pt (3.0 GB)
├── checkpoint_step_150000.pt (3.0 GB)
└── checkpoint_step_160000.pt (3.0 GB) ← RECOMMANDÉ
```

---

## 4. V3 Grand 60k (INCOMPLET ⚠️)

**Training**:
- Steps: **110,000 → 140,000** (4 checkpoints)
- Status: ⚠️ Training interrompu

**Checkpoints**:
```
trained_models/runs/french_v3_finetune_grand_60k/
├── checkpoint_step_110000.pt (3.0 GB)
├── checkpoint_step_120000.pt (3.0 GB)
├── checkpoint_step_130000.pt (3.0 GB)
└── checkpoint_step_140000.pt (3.0 GB)
```

---

## 5. Medium Clean (SOUS-ENTRAÎNÉ ❌)

**Architecture**: SimpleTransformer (260M paramètres - version simplifiée)
- `d_model`: 1024
- `num_layers`: **4** (vs 18 pour V3)
- `num_heads`: 16

**Training**:
- Steps: **1,000 → 5,000** (5 checkpoints)
- Status: ⚠️ Arrêté trop tôt
- Loss finale: 3.0535 (élevée)

**Checkpoints**:
```
trained_models/runs/french_medium_clean/
├── checkpoint_step_1000.pt (1.2 GB)
├── checkpoint_step_2000.pt (1.2 GB)
├── checkpoint_step_3000.pt (1.2 GB)
├── checkpoint_step_4000.pt (1.2 GB)
└── checkpoint_step_5000.pt (1.2 GB)
```

**❌ NE PAS UTILISER**: Sous-entraîné, inférence de mauvaise qualité

---

## 6. Medium Multi-GPU (MAUVAISE CONVERGENCE ❌)

**Architecture**: SimpleTransformer (260M paramètres - 4 layers)

**Training**:
- Steps: **20 → 10,000** (11 checkpoints)
- Status: ⚠️ Loss trop élevée (4.0+)

**Checkpoints**:
```
trained_models/runs/french_medium_multi_gpu/
├── checkpoint_step_20.pt (1.2 GB)
├── checkpoint_step_1000.pt (1.2 GB)
├── ...
└── checkpoint_step_10000.pt (1.2 GB)
```

**❌ NE PAS UTILISER**: Convergence échouée, loss > 4.0

---

## 🧪 Comment Tester un Modèle

### Option 1: Script d'Inférence V3 (RECOMMANDÉ)

Créer `scripts/test_inference_v3.py`:

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path
import argparse

# Architecture V3 (18 layers)
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size=32000, embed_dim=1024, num_heads=16, num_layers=18, ff_hidden_dim=4096):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.tok_embed(x)
        x = self.encoder(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, device="cuda"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\n{'='*70}")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")
    print(prompt, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            token_text = tokenizer.decode(next_token[0])
            print(token_text, end="", flush=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Bonjour, comment allez-vous ?")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("data_clean/mistral_tokenizer")
    print("✅ Tokenizer chargé")

    # Load Model
    print("🔧 Création du modèle V3 (18 layers)...")
    model = TinyTransformerLM()
    model.to(device)

    # Load Checkpoint
    print(f"📥 Chargement: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state'])
    print(f"✅ Modèle chargé (step {checkpoint['step']:,})")

    # Generate
    generate(model, tokenizer, args.prompt, device=device)

if __name__ == "__main__":
    main()
```

### Option 2: Script Interactif

```bash
cd /home/vincent/code/repo/french-llm-from-scratch
source .venv/bin/activate

python3 scripts/test_inference_v3.py \
  --checkpoint trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt
```

### Option 3: Test Multiple Prompts

```bash
# Test qualité du modèle
for prompt in \
  "Bonjour, comment allez-vous ?" \
  "Paris est la capitale de" \
  "L'intelligence artificielle permet de" \
  "Je suis très content"
do
  python3 scripts/test_inference_v3.py \
    --checkpoint trained_models/runs/french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt \
    --prompt "$prompt"
done
```

---

## 📊 Comparaison des Architectures

| Modèle | Layers | Params | Taille Checkpoint | Quality |
|--------|--------|--------|-------------------|---------|
| **V3 (TinyTransformerLM)** | 18 | 260M | 3.0 GB | ⭐⭐⭐⭐⭐ |
| **Medium (SimpleTransformer)** | 4 | 260M | 1.2 GB | ⭐⭐ |

**Note**: Les modèles V3 ont **18 layers** (architecture "Tiny") ce qui les rend beaucoup plus puissants malgré le même nombre de paramètres que les "Medium" à 4 layers.

---

## ⚠️ Problèmes Connus

### Medium Models (french_medium_clean, french_medium_multi_gpu, french_medium_optimized)

1. **Inférence cassée** - Tokens corrompus/non-décodables
2. **Sous-entraînés** - Seulement 5k-10k steps
3. **Loss élevée** - 3.0+ (convergence faible)

**Action**: Utiliser uniquement les modèles V3.

### french_medium_optimized

- ⚠️ Training **arrêté à 31.6%** (step 6,326 / 20,000)
- Pas de checkpoints sauvegardés
- Inférence **non fonctionnelle**

---

## 💡 Recommandations

### Pour Production
✅ **Utiliser**: `french_medium_optimized_batch4_grad3/checkpoint_step_250000.pt`
- Le plus entraîné (250k steps)
- Architecture V3 complète (18 layers)
- Meilleure qualité attendue

### Pour Tests/Démo
✅ **Utiliser**: `french-llm-from-scratch-V3-mistral/checkpoint_step_100000.pt`
- Stable et testé (100k steps)
- Bon compromis qualité/taille
- Architecture V3

### Pour Recherche
✅ **Utiliser**: `french_v3_finetune_resample_1/checkpoint_step_160000.pt`
- Variant expérimental
- Resampling des données
- Peut avoir des propriétés intéressantes

---

## 📄 Fichiers Utiles

```
french-llm-from-scratch/
├── data_clean/
│   ├── mistral_tokenizer/ ← Tokenizer (requis)
│   └── conversations_mega_train_mistral_tokenized.pt
├── trained_models/runs/
│   ├── french_medium_optimized_batch4_grad3/ ← MEILLEUR
│   ├── french-llm-from-scratch-V3-mistral/ ← STABLE
│   ├── french_v3_finetune_resample_1/
│   ├── french_v3_finetune_grand_60k/
│   ├── french_medium_clean/ ← Éviter
│   └── french_medium_multi_gpu/ ← Éviter
└── scripts/
    ├── test_inference_v3.py ← À créer
    └── train_subtitles_transformer.py ← Architecture source
```

---

## 🔗 Ressources

- **Tokenizer**: Mistral (32k vocab)
- **Langue**: 100% Français
- **Data**: Conversations filtrées
- **Framework**: PyTorch + Transformers

---

**Dernière mise à jour**: 2026-01-07  
**Modèles testables**: 6  
**Checkpoints disponibles**: 42  
**Meilleur modèle**: V3 Finetune Batch4 (250k steps)
