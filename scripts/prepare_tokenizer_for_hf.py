#!/usr/bin/env python3
"""
Prépare le tokenizer Fineweb-32k pour Hugging Face
Convertit tokenizers.Tokenizer vers transformers.PreTrainedTokenizer
"""
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import json
import argparse


def prepare_tokenizer(tokenizer_path: str, output_dir: str):
    """
    Convertit le tokenizer custom Fineweb-32k en format Hugging Face
    """
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔧 PRÉPARATION TOKENIZER POUR HUGGING FACE")
    print("=" * 60)
    
    # 1. Charger le tokenizer tokenizers
    print(f"\n📂 Chargement tokenizer: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"✅ Vocab size: {vocab_size:,} tokens")
    
    # 2. Créer PreTrainedTokenizerFast
    print("\n🔄 Conversion vers PreTrainedTokenizerFast...")
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",  # GPT-2 style
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",  # GPT-2 utilise eos comme pad
        model_max_length=1024,
        padding_side="right",
        truncation_side="right",
    )
    
    # 3. Configurer les tokens spéciaux
    special_tokens = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }
    
    # 4. Sauvegarder
    print(f"\n💾 Sauvegarde dans: {output_dir}")
    hf_tokenizer.save_pretrained(output_dir)
    
    # 5. Créer tokenizer_config.json enrichi
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "model_max_length": 1024,
        "padding_side": "right",
        "truncation_side": "right",
        "clean_up_tokenization_spaces": True,
        "add_prefix_space": False,
        "add_bos_token": False,
        "add_eos_token": False,
        "tokenizer_type": "BPE",
        "vocab_size": vocab_size,
        "name_or_path": "fineweb-32k-custom",
    }
    
    config_file = output_dir / "tokenizer_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Fichiers créés:")
    print(f"   - tokenizer.json")
    print(f"   - tokenizer_config.json")
    print(f"   - special_tokens_map.json")
    
    # 6. Test rapide
    print("\n🧪 Test du tokenizer:")
    test_text = "Bonjour, comment vas-tu ? Je suis un modèle de langage français."
    encoded = hf_tokenizer.encode(test_text)
    decoded = hf_tokenizer.decode(encoded)
    
    print(f"   Texte: {test_text}")
    print(f"   Tokens: {len(encoded)} tokens")
    print(f"   IDs: {encoded[:10]}...")
    print(f"   Décodé: {decoded}")
    
    print("\n" + "=" * 60)
    print("✅ TOKENIZER PRÊT POUR HUGGING FACE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Préparer tokenizer pour Hugging Face")
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='trained_models/tokenizers/fineweb-32k/tokenizer.json',
        help='Chemin vers tokenizer.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='trained_models/huggingface/french-llm-from-scratch',
        help='Dossier de sortie (même que le modèle)'
    )
    
    args = parser.parse_args()
    prepare_tokenizer(args.tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
