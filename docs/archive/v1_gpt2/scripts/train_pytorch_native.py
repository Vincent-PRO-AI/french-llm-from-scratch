#!/usr/bin/env python3
"""
🚀 PYTORCH NATIVE TRAINING INTERFACE FOR FRENCH LLM
===================================================
Direct PyTorch training loop with real-time GPU monitoring
Architecture: LLaMA 260M (Decoder-based causal LM)
Dataset: 100% French conversations (653M tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import time
import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model_name": "French-LLM-V3-LLaMA-260M",
    "checkpoint_dir": "trained_models/runs/french_v3_llama_260m",
    "dataset_path": "data_clean/conversations_mega_train_mistral_tokenized.pt",
    "tokenizer_path": "trained_models/tokenizers/vincent_tokenizer_FR_v2/tokenizer.model",
    
    # Training hyperparams
    "max_steps": 80000,
    "warmup_steps": 2000,
    "learning_rate": 2e-4,
    "min_lr": 1e-5,
    "batch_size": 16,  # FP16 permet batch 16 (savings 30-40% VRAM)
    "seq_length": 2048,
    "gradient_accumulation_steps": 1,  # Pas d'accumulation nécessaire
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
    "dtype": torch.float16,  # FP16 au lieu de BF16 (30-40% VRAM savings)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,  # Automatic mixed precision
    "compile_model": False,  # torch.compile (experimental)
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE)"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
    
    def forward(self, x, seq_len):
        # x: (batch, seq_len, num_heads, head_dim)
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb


class Attention(nn.Module):
    """Multi-head attention with Flash Attention-like optimization"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Causal attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with SiLU activation"""
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, dropout)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, intermediate_size, dropout)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LLaMAModel(nn.Module):
    """LLaMA-style causal language model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            TransformerBlock(
                config["hidden_size"],
                config["num_attention_heads"],
                config["intermediate_size"],
                config["dropout"]
            )
            for _ in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.embed_tokens.weight
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        
        # Causal mask
        seq_len = input_ids.shape[1]
        if attention_mask is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        else:
            causal_mask = attention_mask.unsqueeze(1) & torch.tril(
                torch.ones(seq_len, seq_len, device=input_ids.device)
            ).unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


# ============================================================================
# DATASET
# ============================================================================

class TokenDataset(Dataset):
    """Pre-tokenized dataset"""
    def __init__(self, tokens, seq_length, stride=1):
        self.tokens = tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens)
        self.seq_length = seq_length
        self.stride = stride
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_length, stride):
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
    """Main training loop"""
    def __init__(self, model, train_loader, eval_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.device = config["device"]
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if config["use_amp"] else None
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Metrics
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
    
    def get_lr(self, step):
        """Cosine annealing with warmup"""
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
    
    def train_step(self, batch, accum_step):
        """Single training step with gradient accumulation"""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        self.model.train()
        
        # Scale loss by accumulation steps
        accum_steps = self.config["gradient_accumulation_steps"]
        
        if self.config["use_amp"]:
            with torch.cuda.amp.autocast(dtype=self.config["dtype"]):
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, self.config["vocab_size"]),
                    labels.view(-1)
                )
                loss = loss / accum_steps
            
            self.scaler.scale(loss).backward()
            
            # Optimizer step only every N accumulation steps
            if (accum_step + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, self.config["vocab_size"]),
                labels.view(-1)
            )
            loss = loss / accum_steps
            loss.backward()
            
            if (accum_step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return loss.item() * accum_steps  # Return actual loss
    
    @torch.no_grad()
    def eval_step(self):
        """Evaluation loop"""
        self.model.eval()
        losses = []
        
        for batch in tqdm(self.eval_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            if self.config["use_amp"]:
                with torch.cuda.amp.autocast(dtype=self.config["dtype"]):
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, self.config["vocab_size"]),
                        labels.view(-1)
                    )
            else:
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, self.config["vocab_size"]),
                    labels.view(-1)
                )
            
            losses.append(loss.item())
        
        return np.mean(losses)
    
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        ckpt_dir = Path(self.config["checkpoint_dir"]) / f"checkpoint_step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        
        with open(ckpt_dir / "metadata.json", 'w') as f:
            json.dump({
                "step": step,
                "loss": self.train_losses[-1] if self.train_losses else 0,
                "learning_rate": self.learning_rates[-1] if self.learning_rates else 0,
            }, f)
        
        print(f"✅ Checkpoint saved: {ckpt_dir}")
    
    def train(self):
        """Main training loop with gradient accumulation"""
        print("\n" + "="*80)
        print("🚀 PYTORCH NATIVE TRAINING - FRENCH LLM V3")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {self.config['batch_size']} × {self.config['gradient_accumulation_steps']} (accumulation)")
        print(f"Effective batch: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
        print(f"Max steps: {self.config['max_steps']:,}")
        print("="*80 + "\n")
        
        start_time = time.time()
        pbar = tqdm(total=self.config["max_steps"], desc="Training")
        accum_step = 0
        
        try:
            while self.global_step < self.config["max_steps"]:
                for batch in self.train_loader:
                    # Learning rate schedule
                    lr = self.get_lr(self.global_step)
                    self.set_learning_rate(lr)
                    self.learning_rates.append(lr)
                    
                    # Training step with gradient accumulation
                    loss = self.train_step(batch, accum_step)
                    self.train_losses.append(loss)
                    
                    # Optimizer step happens every N batches in train_step
                    accum_step = (accum_step + 1) % self.config["gradient_accumulation_steps"]
                    if accum_step == 0:
                        self.global_step += 1
                        pbar.update(1)
                    
                    # Logging
                    if self.global_step % self.config["log_steps"] == 0:
                        elapsed = time.time() - start_time
                        throughput = self.global_step / elapsed if elapsed > 0 else 0
                        eta_sec = (self.config["max_steps"] - self.global_step) / throughput if throughput > 0 else 0
                        
                        pbar.set_postfix({
                            "loss": f"{loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "tokens/sec": f"{throughput * self.config['batch_size'] * self.config['seq_length']:.0f}",
                            "ETA": f"{int(eta_sec/3600)}h {int((eta_sec%3600)/60)}m"
                        })
                    
                    # Evaluation
                    if self.global_step > 0 and self.global_step % self.config["eval_steps"] == 0:
                        eval_loss = self.eval_step()
                        self.eval_losses.append(eval_loss)
                        print(f"\n[Step {self.global_step}] Eval loss: {eval_loss:.4f}")
                    
                    # Checkpointing
                    if self.global_step > 0 and self.global_step % self.config["save_steps"] == 0:
                        self.save_checkpoint(self.global_step)
                    
                    if self.global_step >= self.config["max_steps"]:
                        break
        
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        
        finally:
            pbar.close()
            
            # Final checkpoint
            self.save_checkpoint(self.global_step)
            
            elapsed = time.time() - start_time
            print(f"\n✅ Training completed!")
            print(f"   Total steps: {self.global_step:,}")
            print(f"   Total time: {int(elapsed/3600)}h {int((elapsed%3600)/60)}m")
            print(f"   Final loss: {self.train_losses[-1]:.4f}")
            print("="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Create checkpoint dir
    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("\n📊 Loading dataset...")
    tokens = torch.load(CONFIG["dataset_path"], map_location='cpu')
    print(f"   Loaded {len(tokens):,} tokens")
    
    # Create dataset
    print("📚 Creating train/eval split...")
    split_idx = int(len(tokens) * 0.85)
    train_tokens = tokens[:split_idx]
    eval_tokens = tokens[split_idx:]
    
    train_dataset = TokenDataset(train_tokens, CONFIG["seq_length"], stride=CONFIG["seq_length"]//2)
    eval_dataset = TokenDataset(eval_tokens, CONFIG["seq_length"], stride=CONFIG["seq_length"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=CONFIG["batch_size"],
        pin_memory=True,
        num_workers=2,
    )
    
    print(f"   Train sequences: {len(train_dataset):,}")
    print(f"   Eval sequences: {len(eval_dataset):,}")
    
    # Create model
    print("\n🏗️  Building model...")
    model = LLaMAModel(CONFIG).to(CONFIG["device"])
    if CONFIG["dtype"] == torch.bfloat16:
        model = model.to(torch.bfloat16)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, eval_loader, CONFIG)
    
    # Train
    trainer.train()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
