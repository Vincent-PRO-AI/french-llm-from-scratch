#!/usr/bin/env python3
"""
Script de publication automatique sur HuggingFace Hub
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def create_model_card():
    """Crée la carte du modèle."""
    
    model_card = """---
license: openrail
language: fr
tags:
  - french
  - llm
  - transformer
  - mistral-tokenizer
  - rtx5080
library_name: transformers
pipeline_tag: text-generation
---

# French LLM Medium (500K Steps)

Un modèle de langage français entraîné de zéro, 100% local sur GPU RTX 5080.

## 📊 Configuration du Modèle

| Aspect | Valeur |
|--------|--------|
| **Architecture** | Transformer (18 layers, 16 heads) |
| **Embedding dim** | 1024 |
| **FF hidden** | 4096 |
| **Paramètres** | ~260M |
| **Tokenizer** | Mistral-7B (32k vocab) |
| **Block size** | 1024 |
| **Étapes entraînement** | 500,000 |
| **Batch size** | 12 (adapté RTX 5080) |

## 🚀 Matériel d'Entraînement

- **GPU**: NVIDIA RTX 5080 (16 GB VRAM)
- **RAM**: 96 GB DDR5 5600MT/s
- **Durée**: ~15-20 heures
- **Date**: Décembre 2025
- **Optimisations**: 
  - Mixed Precision (AMP)
  - Gradient Checkpointing
  - NVMe Cache
  - torch.compile

## 📚 Données d'Entraînement

- **Wikipedia français**: ~13GB
- **FineWeb**: ~5GB  
- **LMSYS conversations**: ~2GB
- **Total tokens pré-tokenisés**: 85M+

## 💻 Utilisation

### Via Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Vincent-PRO-AI/french-llm-500k")
model = AutoModelForCausalLM.from_pretrained(
    "Vincent-PRO-AI/french-llm-500k",
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Bonjour, comment ça va ? "
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,
        top_k=50
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Via LM Studio (Format GGUF)

1. Télécharger le fichier GGUF quantisé (Q4_K_M)
2. Ouvrir LM Studio
3. Charger le modèle
4. Commencer à converser!

**Avantages LM Studio**:
- ✅ Modèle 100% local
- ✅ Pas de GPU requis (peut utiliser CPU)
- ✅ Très rapide avec RTX 5080
- ✅ Interface conviviale

## 📈 Performances

- **Loss final**: À déterminer après entraînement complet
- **Temps d'inférence**: ~50-100 ms/token (RTX 5080)
- **VRAM requis**: 8+ GB
- **Quantisé (Q4)**: ~2 GB seulement

## 🔗 Ressources Connexes

- [Entraînement depuis zéro](https://github.com/Vincent-PRO-AI/french-llm-from-scratch)
- [LM Studio](https://lmstudio.ai/)
- [HuggingFace Model Hub](https://huggingface.co/)

## 📝 Citation

```bibtex
@model{french_llm_500k,
  author = {Vincent},
  title = {French LLM Medium (500K Steps)},
  year = {2025},
  url = {https://huggingface.co/Vincent-PRO-AI/french-llm-500k}
}
```

## ⚖️ Licence

Ce modèle est fourni sous la licence OpenRAIL pour usage académique et commercial responsable.

---

**Entraîné avec amour et RTX 5080 ❤️**
"""
    
    return model_card

def publish_to_hub():
    """Publie le modèle sur HuggingFace Hub."""
    
    print("🚀 PUBLICATION HUGGINGFACE HUB")
    print("=" * 70)
    
    repo_id = "Vincent-PRO-AI/french-llm-500k"
    local_dir = Path("trained_models/runs/french_medium_rtx5080_extended")
    
    # Créer la carte du modèle
    model_card = create_model_card()
    card_path = local_dir / "README.md"
    card_path.write_text(model_card)
    print(f"✅ Carte du modèle créée: {card_path}")
    
    # Configuration HuggingFace
    config = {
        "model_type": "transformer",
        "architectures": ["TinyTransformerLM"],
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 18,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_position_embeddings": 1024,
        "torch_dtype": "float16",
        "transformers_version": "4.36.0",
    }
    
    config_path = local_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config créée: {config_path}")
    
    # Tokenizer
    tokenizer_files = [
        "data_clean/mistral_tokenizer/tokenizer.json",
        "data_clean/mistral_tokenizer/tokenizer_config.json"
    ]
    
    for src in tokenizer_files:
        src_path = Path(src)
        if src_path.exists():
            dst_path = local_dir / src_path.name
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"✅ Tokenizer copié: {dst_path}")
    
    # Push avec huggingface-cli
    print(f"\n📤 Push vers {repo_id}...")
    print("⚠️  Assurez-vous d'avoir exécuté: huggingface-cli login")
    
    cmd = [
        "huggingface-cli",
        "repo",
        "upload",
        repo_id,
        str(local_dir),
        "--repo-type", "model"
    ]
    
    print(f"\n💻 Exécution:")
    print(f"   {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Publication réussie!")
            print(f"🔗 Lien: https://huggingface.co/{repo_id}")
            return True
        else:
            print(f"⚠️  Erreur: {result.stderr}")
            print(f"\nVous pouvez aussi utiliser:")
            print(f"  git clone https://huggingface.co/{repo_id}")
            print(f"  cp -r {local_dir}/* {repo_id}/")
            print(f"  cd {repo_id} && git add . && git commit -m 'Add trained model' && git push")
            return False
    except FileNotFoundError:
        print("❌ huggingface-cli non trouvé!")
        print("💡 Installation: pip install huggingface-hub")
        return False

if __name__ == "__main__":
    try:
        success = publish_to_hub()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Publication annulée")
        sys.exit(1)
