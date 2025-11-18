#!/usr/bin/env python3
"""
Script pour publier le modèle entraîné sur Hugging Face Hub.
Usage: python scripts/publish_to_huggingface.py --checkpoint trained_models/runs/french_medium_mega_finetune_200k/checkpoint_step_200000.pt
"""
import argparse
import json
from pathlib import Path
import torch
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

def create_model_card(metadata: dict) -> str:
    """Génère une model card pour Hugging Face."""
    return f"""---
language: fr
license: mit
tags:
- french
- gpt
- from-scratch
- causal-lm
datasets:
- ultrachat
- oasst2
- dolly
base_model: custom
pipeline_tag: text-generation
---

# French LLM 260M - Checkpoint {metadata.get('steps', 'N/A')}k

## Description

Modèle de langage français entraîné from scratch sur GPU consumer (RTX 5080).

**Architecture** : GPT custom ~260M paramètres
- 18 layers transformer
- 16 attention heads  
- 1024 embedding dimension
- 4096 feed-forward hidden
- 1024 tokens context window

**Tokenizer** : BPE 32k vocabulaire (FinewEB-32k)

**Corpus d'entraînement** : {metadata.get('total_tokens', '197M')} tokens français
- UltraChat FR (119M tokens)
- OASST2 FR (conversations)
- Dolly FR (instructions)
- Wikipedia FR + FineWeb-2 FR

## Résultats

- **Training loss** : {metadata.get('train_loss', 'N/A')}
- **Validation loss** : {metadata.get('val_loss', '4.79')}
- **Steps** : {metadata.get('steps', '200k')}
- **Training time** : ~43h sur RTX 5080 16GB

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Charger le modèle (adapté après conversion)
model = AutoModelForCausalLM.from_pretrained("Vincent-PRO-AI/french-llm-260m")
tokenizer = AutoTokenizer.from_pretrained("Vincent-PRO-AI/french-llm-260m")

# Génération
prompt = "Bonjour, comment vas-tu ?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.8)
print(tokenizer.decode(outputs[0]))
```

## Limitations

⚠️ Version expérimentale :
- Peut générer du contenu incohérent sur des tâches complexes
- Artefacts BPE visibles (tokenizer maison)
- Pas de fine-tuning RLHF (alignment)

## Citation

```bibtex
@misc{{french-llm-260m,
  author = {{Vincent R. C. Soulé}},
  title = {{French LLM 260M - From Scratch Training}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://github.com/Vincent-PRO-AI/french-llm-from-scratch}}}}
}}
```

## Contact

- GitHub: [Vincent-PRO-AI](https://github.com/Vincent-PRO-AI)
- LinkedIn: [Vincent R. C. Soulé](https://linkedin.com/in/vincentrcsoulé)

## License

MIT License - Utilisation libre pour recherche et applications commerciales.
"""

def publish_model(checkpoint_path: Path, repo_name: str = "french-llm-260m", private: bool = False):
    """Publie le modèle sur Hugging Face Hub."""
    
    print(f"📦 Chargement du checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    metadata = {
        'steps': checkpoint.get('step', 200000) // 1000,  # en k
        'train_loss': f"{checkpoint.get('train_loss', 0):.4f}",
        'val_loss': f"{checkpoint.get('val_loss', 4.79):.4f}",
        'total_tokens': '197M'
    }
    
    # Créer le repo
    api = HfApi()
    print(f"🔧 Création du repo: {repo_name}")
    try:
        create_repo(repo_name, private=private, exist_ok=True)
    except Exception as e:
        print(f"⚠️  Repo existe déjà ou erreur: {e}")
    
    # Sauvegarder les artefacts localement
    output_dir = Path("hf_upload")
    output_dir.mkdir(exist_ok=True)
    
    # Model card
    model_card = create_model_card(metadata)
    (output_dir / "README.md").write_text(model_card, encoding="utf-8")
    
    # Config
    config = {
        "architectures": ["TinyTransformerLM"],
        "model_type": "gpt",
        "vocab_size": checkpoint.get('config', {}).get('vocab_size', 32000),
        "n_positions": checkpoint.get('config', {}).get('block_size', 1024),
        "n_embd": checkpoint.get('config', {}).get('embed_dim', 1024),
        "n_layer": checkpoint.get('config', {}).get('num_layers', 18),
        "n_head": checkpoint.get('config', {}).get('num_heads', 16),
        "n_inner": checkpoint.get('config', {}).get('ff_hidden_dim', 4096),
        "activation_function": "gelu",
        "resid_pdrop": checkpoint.get('config', {}).get('dropout', 0.1),
        "embd_pdrop": checkpoint.get('config', {}).get('dropout', 0.1),
        "attn_pdrop": checkpoint.get('config', {}).get('dropout', 0.1),
        "bos_token_id": 1,
        "eos_token_id": 2,
        "transformers_version": "4.36.0"
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    
    # Checkpoint PyTorch
    torch.save(checkpoint, output_dir / "pytorch_model.bin")
    
    # Upload
    print(f"⬆️  Upload vers Hugging Face Hub...")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=f"Vincent-PRO-AI/{repo_name}",
        repo_type="model"
    )
    
    print(f"✅ Modèle publié : https://huggingface.co/Vincent-PRO-AI/{repo_name}")
    print("\n📝 TODO: Ajouter le tokenizer avec:")
    print("   tokenizer.save_pretrained('hf_upload')")
    print("   puis re-upload le dossier")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Chemin vers checkpoint .pt")
    parser.add_argument("--repo-name", type=str, default="french-llm-260m", help="Nom du repo HF")
    parser.add_argument("--private", action="store_true", help="Repo privé")
    args = parser.parse_args()
    
    publish_model(args.checkpoint, args.repo_name, args.private)
