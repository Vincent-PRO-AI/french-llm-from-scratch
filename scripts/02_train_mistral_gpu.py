#!/usr/bin/env python3
"""
PHASE 3: Entraînement Mistral - RTX 5080 Optimisé
Configuration GPU optimale pour RTX 5080 (Blackwell, 17.1GB VRAM)
"""

import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# PyTorch settings pour RTX 5080
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm

# Configuration logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_tokenized_dataset(dataset_path, seq_length=2048):
    """Charge dataset .pt et remet en forme 2D"""
    logger.info(f"Chargement dataset: {dataset_path}")
    data = torch.load(dataset_path, map_location='cpu')
    
    if data.dim() == 1:
        # Reshape 1D tensor en sequences 2D
        num_sequences = len(data) // seq_length
        data = data[:num_sequences * seq_length].reshape(num_sequences, seq_length)
        logger.info(f"✅ Tensor reshapé: {data.shape}")
    
    return data


def create_dataloaders(train_data, test_data, batch_size=16, num_workers=4):
    """Crée dataloaders optimisés"""
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"DataLoaders créés: train={len(train_loader)}, test={len(test_loader)}")
    return train_loader, test_loader


def setup_model_and_tokenizer(device='cuda'):
    """Charge modèle Mistral et tokenizer"""
    logger.info(f"Chargement Mistral Tiny sur {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    from transformers import MistralConfig, MistralForCausalLM
    config = MistralConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=4096,
        max_position_embeddings=2048,
        use_cache=False,
    )
    
    model = MistralForCausalLM(config)
    model = model.to(device)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Modèle chargé: {total_params/1e6:.1f}M params")
    
    return model, tokenizer


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, log_dir):
    """Entraîne une epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
    
    for batch_idx, (input_ids,) in enumerate(pbar):
        input_ids = input_ids.to(device)
        
        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Logging tous les 100 batches
        if (batch_idx + 1) % 100 == 0:
            log_entry = {
                'epoch': epoch,
                'batch': batch_idx,
                'loss': avg_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            }
            with open(os.path.join(log_dir, 'training.jsonl'), 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    """Évalue sur test set"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for (input_ids,) in tqdm(test_loader, desc="Evaluating", unit="batch"):
            input_ids = input_ids.to(device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item()
    
    return total_loss / len(test_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True, help='Chemin fichier .pt')
    parser.add_argument('--output-path', type=str, required=True, help='Répertoire sortie')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=3, help='Nombre epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--use-bf16', action='store_true', help='Utiliser BFloat16')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()
    
    # Vérifications préalables
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    logger.info(f"🎯 Device: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"CUDA: {torch.version.cuda}")
    
    # Préparation répertoire
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logger.info(f"📁 Output: {output_dir}")
    
    # Chargement données
    train_data = load_tokenized_dataset(args.dataset_path)
    
    # Pour test: utilise un petit subset
    test_data = train_data[:1000]  # 1000 sequences pour validation
    
    train_loader, test_loader = create_dataloaders(
        train_data, test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Modèle
    model, tokenizer = setup_model_and_tokenizer(device)
    
    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 DÉMARRAGE ENTRAÎNEMENT")
    logger.info(f"{'='*60}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"{'='*60}\n")
    
    # Entraînement
    best_loss = float('inf')
    
    try:
        for epoch in range(args.num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch+1, log_dir)
            test_loss = evaluate(model, test_loader, device)
            
            logger.info(f"\n📊 Epoch {epoch+1}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
            
            if test_loss < best_loss:
                best_loss = test_loss
                checkpoint_path = output_dir / 'best_model.pt'
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"✅ Checkpoint sauvegardé: {checkpoint_path}")
        
        # Sauvegarder modèle final
        final_model_dir = output_dir / 'final_model'
        final_model_dir.mkdir(exist_ok=True)
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logger.info(f"✅ Modèle final sauvegardé: {final_model_dir}")
        
    except KeyboardInterrupt:
        logger.warning("Entraînement interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
