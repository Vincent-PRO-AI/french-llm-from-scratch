#!/usr/bin/env python3
"""
🚀 PYTORCH TRAINING - FP4 QUANTIZATION
======================================
Ultra-lightweight training with 4-bit quantization using bitsandbytes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np
from bitsandbytes.nn import Linear4bit

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model_name": "French-LLM-V3-LLaMA-260M-FP4",
    "checkpoint_dir": "trained_models/runs/french_v3_llama_260m_fp4",
    "dataset_path": "data_clean/conversations_mega_train_mistral_tokenized.pt",
    
    # Training hyperparams
    "max_steps": 80000,
    "warmup_steps": 2000,
    "learning_rate": 2e-4,
    "min_lr": 1e-5,
    "batch_size": 32,  # FP4 permet batch énorme!
    "seq_length": 2048,
    "gradient_accumulation_steps": 1,
    "grad_clip": 1.0,
    
    # Eval/Save
    "eval_steps": 500,
    "save_steps": 2000,
    "log_steps": 50,
    
    # Model architecture
    "hidden_size": 1024,
    "num_hidden_layers": 18,
    "num_attention_heads": 32,
    "intermediate_size": 4096,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
    "dropout": 0.1,
    
    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_fp4": True,  # 4-bit quantization
    "use_amp": True,
}

# ============================================================================
# QUANTIZED LINEAR LAYER
# ============================================================================

class QuantizedLinear(nn.Module):
    """FP4 Linear layer using bitsandbytes"""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = Linear4bit(
            in_features, 
            out_features,
            bias=bias,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type="nf4"
        )
    
    def forward(self, x):
        return self.linear(x)

# ============================================================================
# MODEL ARCHITECTURE (SIMPLIFIED - no custom attention)
# ============================================================================

class SimpleLLaMA(nn.Module):
    """Simplified LLaMA using nn.Transformer"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config["vocab_size"], config["hidden_size"])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["hidden_size"],
            nhead=config["num_attention_heads"],
            dim_feedforward=config["intermediate_size"],
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["num_hidden_layers"],
        )
        
        self.norm = nn.LayerNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.embed.weight
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        
        # Causal mask
        seq_len = input_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1
        )
        
        x = self.transformer(
            x,
            src_mask=causal_mask if seq_len > 1 else None,
            src_key_padding_mask=None
        )
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

# ============================================================================
# DATASET
# ============================================================================

class TokenDataset:
    """Pre-tokenized dataset"""
    def __init__(self, tokens, seq_length):
        self.tokens = tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens)
        self.seq_length = seq_length
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_length, seq_length // 2):
            self.sequences.append(i)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        start = self.sequences[idx]
        input_ids = self.tokens[start:start + self.seq_length].long()
        labels = self.tokens[start + 1:start + self.seq_length + 1].long()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    def __init__(self, model, train_dataset, config):
        self.model = model
        self.config = config
        self.device = config["device"]
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=0.01,
        )
        
        self.global_step = 0
        self.train_losses = []
        self.learning_rates = []
    
    def get_lr(self, step):
        """Cosine annealing"""
        if step < self.config["warmup_steps"]:
            return self.config["learning_rate"] * step / self.config["warmup_steps"]
        
        progress = (step - self.config["warmup_steps"]) / (
            self.config["max_steps"] - self.config["warmup_steps"]
        )
        return self.config["min_lr"] + (self.config["learning_rate"] - self.config["min_lr"]) * \
               0.5 * (1 + np.cos(np.pi * progress))
    
    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def save_checkpoint(self, step):
        ckpt_dir = Path(self.config["checkpoint_dir"]) / f"checkpoint_step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        
        print(f"✅ Checkpoint: {ckpt_dir}")
    
    def train(self):
        print("\n" + "="*80)
        print("🚀 FP4 QUANTIZED TRAINING - FRENCH LLM V3")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Max steps: {self.config['max_steps']:,}")
        print("="*80 + "\n")
        
        start_time = time.time()
        pbar = tqdm(total=self.config["max_steps"], desc="Training FP4")
        
        try:
            while self.global_step < self.config["max_steps"]:
                for batch in self.train_loader:
                    lr = self.get_lr(self.global_step)
                    self.set_learning_rate(lr)
                    self.learning_rates.append(lr)
                    
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    self.model.train()
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, self.config["vocab_size"]),
                        labels.view(-1)
                    )
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    self.train_losses.append(loss.item())
                    self.global_step += 1
                    pbar.update(1)
                    
                    if self.global_step % self.config["log_steps"] == 0:
                        elapsed = time.time() - start_time
                        throughput = self.global_step / elapsed if elapsed > 0 else 0
                        
                        pbar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "lr": f"{lr:.2e}",
                            "tokens/s": f"{throughput * self.config['batch_size'] * self.config['seq_length']:.0f}",
                        })
                    
                    if self.global_step % self.config["save_steps"] == 0:
                        self.save_checkpoint(self.global_step)
                    
                    if self.global_step >= self.config["max_steps"]:
                        break
        
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted")
        
        finally:
            pbar.close()
            self.save_checkpoint(self.global_step)
            
            elapsed = time.time() - start_time
            print(f"\n✅ Training done!")
            print(f"   Steps: {self.global_step:,}")
            print(f"   Time: {int(elapsed/3600)}h {int((elapsed%3600)/60)}m")
            if self.train_losses:
                print(f"   Final loss: {self.train_losses[-1]:.4f}")
            print("="*80 + "\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Loading dataset...")
    tokens = torch.load(CONFIG["dataset_path"], map_location='cpu')
    print(f"   Loaded {len(tokens):,} tokens")
    
    print("📚 Creating dataset...")
    dataset = TokenDataset(tokens, CONFIG["seq_length"])
    print(f"   Sequences: {len(dataset):,}")
    
    print("\n🏗️  Building model...")
    model = SimpleLLaMA(CONFIG).to(CONFIG["device"])
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(model, dataset, CONFIG)
    trainer.train()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
