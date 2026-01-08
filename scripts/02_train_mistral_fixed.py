#!/usr/bin/env python3
"""
🧠 PHASE 3 - Training Mistral: Modèle from scratch (FIXED VERSION)
Version corrigée qui charge directement des fichiers .pt tokenizés
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
            vocab_size=32000,
            hidden_size=hidden_size,
            intermediate_size=4096,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=max(1, num_heads // 2),
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

def load_tokenized_dataset(dataset_path: str, batch_size: int = 8, seq_length: int = 2048):
    """Charger un fichier .pt tokenisé et rechaper en sequences."""
    try:
        logger.info(f"📥 Charger dataset tokenisé: {dataset_path}")
        
        # Vérifier que le fichier existe
        if not os.path.exists(dataset_path):
            logger.error(f"❌ Fichier introuvable: {dataset_path}")
            return None
        
        # Charger le fichier .pt
        data = torch.load(dataset_path, map_location='cpu')
        logger.info(f"✅ Dataset chargé: shape={data.shape if hasattr(data, 'shape') else len(data)}")
        
        # Créer un TensorDataset
        if isinstance(data, torch.Tensor):
            # Si c'est un vecteur 1D, le rechaper en sequences
            if len(data.shape) == 1:
                logger.info(f"🔧 Rechaper tenseur 1D → sequences 2D...")
                # Supprimer les éléments pour que ça soit divisible
                total_tokens = data.shape[0]
                num_sequences = total_tokens // seq_length
                data = data[:num_sequences * seq_length]
                # Rechaper
                data = data.reshape(num_sequences, seq_length)
                logger.info(f"✅ Reshapé: {data.shape}")
            
            # Créer dataset
            dataset = TensorDataset(data)
        else:
            # Si c'est déjà un dataset HF, l'utiliser directement
            dataset = data
        
        # Créer DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 0 car on peut avoir des problèmes avec les tensors
        )
        
        logger.info(f"✅ DataLoader créé: {len(dataloader)} batches")
        
        return dataloader
    except Exception as e:
        logger.error(f"❌ Erreur chargement dataset: {e}")
        import traceback
        traceback.print_exc()
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
        
        # Compiler si possible
        if use_compile:
            try:
                logger.info("🔧 Compiler le modèle avec torch.compile...")
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("✅ Modèle compilé")
            except Exception as e:
                logger.warning(f"⚠️  torch.compile failed: {e}")
        
        # Déplacer modèle
        model = model.to(device)
        model.train()  # ← IMPORTANT: mettre en mode training AVANT compile
        
        # Créer checkpoint dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training loop
        total_loss = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(device)
                else:
                    input_ids = batch.to(device)
                
                # Forward pass
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(f"   Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f} - LR: {lr:.2e}")
        
        logger.info("=" * 80)
        logger.info("✅ ENTRAÎNEMENT TERMINÉ")
        logger.info("=" * 80)
        
        # Sauvegarder le modèle
        model_save_path = os.path.join(checkpoint_dir, "final_model")
        logger.info(f"💾 Sauvegarder modèle: {model_save_path}")
        model.save_pretrained(model_save_path)
        logger.info("✅ Modèle sauvegardé")
        
    except Exception as e:
        logger.error(f"❌ Erreur training: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Training Mistral from Scratch (FIXED)")
    
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset file (.pt)")
    parser.add_argument("--output-path", type=str, default="./trained_models/french_mistral", help="Output directory")
    parser.add_argument("--use-bf16", action="store_true", help="Use BFloat16")
    parser.add_argument("--use-compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🧠 PHASE 3 - TRAINING MISTRAL FROM SCRATCH (FIXED)")
    logger.info("=" * 80)
    
    # Setup model
    config = setup_model_config(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
    if config is None:
        return
    
    model = create_model_from_scratch(config)
    if model is None:
        return
    
    # Load dataset
    dataloader = load_tokenized_dataset(args.dataset_path, batch_size=args.batch_size)
    if dataloader is None:
        logger.error("❌ Failed to load dataset")
        return
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(dataloader))
    
    # Train
    train_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=args.device,
        checkpoint_dir=args.output_path,
        use_bf16=args.use_bf16,
        use_compile=args.use_compile
    )
    
    logger.info("=" * 80)
    logger.info("✅ PHASE 3 TERMINÉE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
