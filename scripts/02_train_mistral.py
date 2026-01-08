#!/usr/bin/env python3
"""
🧠 PHASE 3 - Training Mistral: Modèle from scratch
Créer et entraîner un modèle Mistral "Tiny" optimisé pour RTX 5080 + Ryzen 7.

Architecture: 
  - Hidden size: 1024
  - Layers: 12-16
  - Heads: 8-16
  - Context window: 2048
  - Vocab size: 32000

Optimisations:
  - BF16 precision
  - Gradient checkpointing
  - torch.compile si possible
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

# Configuration de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_model_config(hidden_size: int = 1024, num_layers: int = 12, num_heads: int = 8):
    """Créer la configuration du modèle Mistral Tiny."""
    try:
        from transformers import MistralConfig
        
        logger.info("🔧 Créer configuration Mistral Tiny...")
        
        config = MistralConfig(
            vocab_size=32000,  # Mistral standard vocab
            hidden_size=hidden_size,
            intermediate_size=4096,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads // 2 if num_heads > 2 else 1,
            max_position_embeddings=2048,
            sliding_window=2048,
            attention_dropout=0.1,
            hidden_dropout_prob=0.1,
            hidden_act="silu",
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
        )
        
        logger.info(f"✅ Configuration créée:")
        logger.info(f"   - Vocab size: {config.vocab_size}")
        logger.info(f"   - Hidden size: {config.hidden_size}")
        logger.info(f"   - Num layers: {config.num_hidden_layers}")
        logger.info(f"   - Num heads: {config.num_attention_heads}")
        logger.info(f"   - Context window: {config.max_position_embeddings}")
        
        return config
    except Exception as e:
        logger.error(f"❌ Erreur création config: {e}")
        return None

def create_model_from_scratch(config):
    """Créer un modèle Mistral from scratch."""
    try:
        from transformers import MistralForCausalLM
        
        logger.info("🔨 Créer modèle Mistral from scratch...")
        
        model = MistralForCausalLM(config)
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Modèle créé:")
        logger.info(f"   - Total parameters: {total_params:,}")
        logger.info(f"   - Trainable parameters: {trainable_params:,}")
        logger.info(f"   - Model size: {total_params * 4 / 1e9:.2f} GB (FP32)")
        
        return model
    except Exception as e:
        logger.error(f"❌ Erreur création modèle: {e}")
        return None

def setup_training(model, learning_rate: float = 5e-4, device: str = "cuda"):
    """Configurer optimizer et scheduler."""
    try:
        logger.info("⚙️  Configurer training...")
        
        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=10000,  # Total steps
            eta_min=1e-6
        )
        
        logger.info(f"✅ Training configuré:")
        logger.info(f"   - Learning rate: {learning_rate}")
        logger.info(f"   - Optimizer: AdamW")
        logger.info(f"   - Scheduler: CosineAnnealing")
        
        return optimizer, scheduler
    except Exception as e:
        logger.error(f"❌ Erreur configuration training: {e}")
        return None, None

def load_dataset_for_training(dataset_path: str, batch_size: int = 8):
    """Charger le dataset pré-tokenisé pour training."""
    try:
        from datasets import load_from_disk
        
        logger.info(f"📥 Charger dataset: {dataset_path}")
        
        dataset = load_from_disk(dataset_path)
        logger.info(f"✅ Dataset chargé: {len(dataset)} exemples")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        logger.info(f"✅ DataLoader créé: {len(dataloader)} batches")
        
        return dataloader
    except Exception as e:
        logger.error(f"❌ Erreur chargement dataset: {e}")
        return None

def train_loop(
    model,
    dataloader,
    optimizer,
    scheduler,
    num_epochs: int = 1,
    device: str = "cuda",
    checkpoint_dir: str = "./checkpoints",
    use_bf16: bool = True,
    use_compile: bool = False
):
    """Loop d'entraînement principal."""
    try:
        logger.info("=" * 80)
        logger.info("🚀 DÉMARRAGE ENTRAÎNEMENT")
        logger.info("=" * 80)
        
        # Compiler le modèle si possible
        if use_compile:
            try:
                logger.info("🔧 Compiler le modèle avec torch.compile...")
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("✅ Modèle compilé")
            except Exception as e:
                logger.warning(f"⚠️  torch.compile failed: {e}, continuing sans compilation")
        
        # Déplacer modèle sur device
        model = model.to(device)
        model.train()
        
        # Setup checkpoints
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Scaler pour BF16/AMP
        if use_bf16:
            scaler = torch.cuda.amp.GradScaler()
            logger.info("✅ BF16 precision enabled")
        
        global_step = 0
        total_loss = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\n📍 Epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                # Déplacer batch sur device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                with torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_bf16 else torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                
                if use_bf16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                
                global_step += 1
                total_loss += loss.item()
                
                # Log tous les 100 steps
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    logger.info(
                        f"  Step {global_step:5d} | Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                
                # Checkpoint tous les 2500 steps
                if global_step % 2500 == 0:
                    checkpoint_file = checkpoint_path / f"checkpoint_step_{global_step}.pt"
                    torch.save({
                        'step': global_step,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'loss': avg_loss,
                    }, checkpoint_file)
                    logger.info(f"  💾 Checkpoint sauvegardé: {checkpoint_file}")
        
        logger.info("=" * 80)
        logger.info("✅ ENTRAÎNEMENT COMPLÉTÉ")
        logger.info("=" * 80)
        
        return model
    except Exception as e:
        logger.error(f"❌ Erreur entraînement: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_model(model, output_path: str):
    """Sauvegarder le modèle au format HuggingFace."""
    try:
        logger.info(f"💾 Sauvegarder le modèle: {output_path}")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(str(output_dir))
        logger.info(f"✅ Modèle sauvegardé: {output_dir}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde modèle: {e}")
        return False

def main():
    """Pipeline principal de training."""
    parser = argparse.ArgumentParser(description="Phase 3 - Training Mistral from Scratch")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--dataset-path", type=str, default="./data/processed_mistral", help="Path to processed dataset")
    parser.add_argument("--output-path", type=str, default="./models/mistral_tiny", help="Output model path")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/mistral_tiny", help="Checkpoint directory")
    parser.add_argument("--use-bf16", action="store_true", default=True, help="Use BF16 precision")
    parser.add_argument("--use-compile", action="store_true", default=False, help="Use torch.compile")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🧠 PHASE 3 - TRAINING MISTRAL FROM SCRATCH")
    logger.info("=" * 80)
    
    # Vérifier CUDA
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available. This script requires GPU.")
        return False
    
    logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"✅ CUDA version: {torch.version.cuda}")
    
    # Créer configuration
    config = setup_model_config(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
    if not config:
        return False
    
    # Créer modèle
    model = create_model_from_scratch(config)
    if not model:
        return False
    
    # Setup training
    optimizer, scheduler = setup_training(model, learning_rate=args.learning_rate)
    if not optimizer:
        return False
    
    # Charger dataset
    dataloader = load_dataset_for_training(args.dataset_path, batch_size=args.batch_size)
    if not dataloader:
        return False
    
    # Entraîner
    model = train_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        use_bf16=args.use_bf16,
        use_compile=args.use_compile
    )
    if not model:
        return False
    
    # Sauvegarder
    if not save_model(model, args.output_path):
        return False
    
    logger.info("=" * 80)
    logger.info("✅ PHASE 3 COMPLÉTÉE - Modèle prêt pour export")
    logger.info("=" * 80)
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
