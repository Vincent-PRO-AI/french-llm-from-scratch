#!/usr/bin/env python3
"""
📊 PHASE 2 - Data Pipeline: Re-tokenisation Mistral
Traite les datasets bruts et les tokenise avec le tokenizer Mistral officiel.

Input: Datasets bruts (FineWeb, OASST2, Wikipedia)
Output: Datasets tokenisés sauvegardés en format .parquet ou .arrow
"""

import os
import json
import argparse
from pathlib import Path
from typing import Iterator, Dict, Any
import logging

# Configuration de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Vérifier et configurer l'environnement requis."""
    try:
        import torch
        import datasets
        from transformers import AutoTokenizer
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"✅ Datasets version: {datasets.__version__}")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Run: pip install -r requirements.txt")
        return False

def load_tokenizer():
    """Charger le tokenizer Mistral officiel."""
    try:
        from transformers import AutoTokenizer
        logger.info("📥 Charger le tokenizer Mistral...")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        logger.info(f"✅ Tokenizer chargé | Vocab size: {len(tokenizer)}")
        return tokenizer
    except Exception as e:
        logger.error(f"❌ Erreur chargement tokenizer: {e}")
        return None

def load_dataset(dataset_name: str, split: str = "train", streaming: bool = True):
    """Charger un dataset brut depuis HuggingFace."""
    try:
        from datasets import load_dataset
        logger.info(f"📥 Charger dataset: {dataset_name} ({split})...")
        
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        logger.info(f"✅ Dataset chargé | Taille: {len(dataset) if not streaming else 'streaming'}")
        return dataset
    except Exception as e:
        logger.error(f"❌ Erreur chargement dataset: {e}")
        return None

def process_text(text: str, tokenizer) -> Dict[str, Any]:
    """Traiter un texte brut et le tokeniser."""
    try:
        # Tokeniser avec ajout du token EOS
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None
        )
        
        # Ajouter token EOS à la fin
        encoded['input_ids'].append(tokenizer.eos_token_id)
        encoded['attention_mask'].append(1)
        
        return encoded
    except Exception as e:
        logger.error(f"❌ Erreur traitement texte: {e}")
        return None

def tokenize_dataset(dataset, tokenizer, output_path: str):
    """Tokeniser un dataset complet et le sauvegarder."""
    try:
        logger.info(f"🔄 Tokenisation du dataset en cours...")
        
        def tokenize_batch(batch):
            """Tokeniser un batch d'exemples."""
            texts = batch.get("text", batch.get("content", batch.get("prompt", [])))
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=2048,
                padding=False,
                return_tensors=None
            )
            return encoded
        
        # Appliquer la tokenisation par batch
        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        # Sauvegarder
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tokenized.save_to_disk(str(output_dir))
        logger.info(f"✅ Dataset sauvegardé: {output_dir}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erreur tokenisation: {e}")
        return False

def create_combined_dataset(tokenized_paths: list, output_path: str):
    """Combiner plusieurs datasets tokenisés."""
    try:
        from datasets import concatenate_datasets, load_from_disk
        
        logger.info(f"🔄 Combinaison de {len(tokenized_paths)} datasets...")
        
        datasets_list = []
        for path in tokenized_paths:
            if Path(path).exists():
                ds = load_from_disk(path)
                datasets_list.append(ds)
                logger.info(f"  ✅ Chargé: {path} ({len(ds)} exemples)")
        
        if datasets_list:
            combined = concatenate_datasets(datasets_list)
            combined.save_to_disk(output_path)
            logger.info(f"✅ Dataset combiné sauvegardé: {output_path} ({len(combined)} exemples)")
            return True
        else:
            logger.error("❌ Aucun dataset trouvé à combiner")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur combinaison: {e}")
        return False

def main():
    """Pipeline principal de tokenisation."""
    parser = argparse.ArgumentParser(description="Phase 2 - Data Processing & Tokenization")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset à charger")
    parser.add_argument("--split", type=str, default="train", help="Split du dataset")
    parser.add_argument("--output-path", type=str, default="./data/processed_mistral", help="Chemin de sortie")
    parser.add_argument("--streaming", action="store_true", help="Utiliser streaming")
    parser.add_argument("--combine", action="store_true", help="Combiner plusieurs datasets")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 PHASE 2 - DATA PIPELINE: TOKENISATION MISTRAL")
    logger.info("=" * 80)
    
    # Vérifier environnement
    if not setup_environment():
        return False
    
    # Charger tokenizer
    tokenizer = load_tokenizer()
    if not tokenizer:
        return False
    
    # Charger et tokeniser dataset
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    if not dataset:
        return False
    
    # Tokeniser
    if not tokenize_dataset(dataset, tokenizer, args.output_path):
        return False
    
    logger.info("=" * 80)
    logger.info("✅ PHASE 2 COMPLÉTÉE - Données prêtes pour training")
    logger.info("=" * 80)
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
