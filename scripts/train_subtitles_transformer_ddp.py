"""
# pyright: reportGeneralTypeIssues=false
Multi-GPU Distributed Data Parallel Training for French LLM
Supports 2+ GPUs with PyTorch DDP

Usage:
    # Single machine, multi-GPU
    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        train_subtitles_transformer_ddp.py

    # Multi-node (Docker Compose)
    docker-compose -f docker-compose.multi-gpu.yml up --profile multi-gpu
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import argparse
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    gpu_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)
    
    return rank, world_size, gpu_id


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def log_info(msg, rank=0):
    """Log only from rank 0"""
    if dist.get_rank() == 0:
        logger.info(msg)


class SimpleTokenDataset(Dataset):
    """Tokenized dataset loaded from memmap or .pt"""
    def __init__(self, data_path, seq_length=1024, split="train"):
        self.data_path = Path(data_path)
        self.seq_length = seq_length
        self.split = split
        self.data = None
        self.is_mmap = False
        
        # Try loading .pt file first (as seen in data_clean)
        if split == "train":
            pt_file = self.data_path / "conversations_mega_train_mistral_tokenized.pt"
        else:
            pt_file = self.data_path / "conversations_mega_test_mistral_tokenized.pt"
            
        if pt_file.exists():
            log_info(f"Loading dataset from {pt_file}")
            self.data = torch.load(pt_file, map_location="cpu")
            # Ensure it's a 1D tensor
            if isinstance(self.data, dict):
                # Handle if it's a dict (e.g. {'input_ids': ...})
                if 'input_ids' in self.data:
                    self.data = self.data['input_ids']
                else:
                    # Assume values are tensors and concatenate them or pick first
                    self.data = list(self.data.values())[0]
            
            if isinstance(self.data, list):
                 self.data = torch.tensor(self.data)

            self.data_size = len(self.data)
            self.num_sequences = max(1, (self.data_size - 1) // seq_length)
            self.is_mmap = False
        else:
            # Fallback to .bin mmap
            if split == "train":
                self.data_file = self.data_path / "train_tokens.bin"
            else:
                self.data_file = self.data_path / "val_tokens.bin"
            
            if self.data_file.exists():
                self.data_size = self.data_file.stat().st_size // 4  # uint32
                self.num_sequences = max(1, (self.data_size - 1) // seq_length)
                self.is_mmap = True
            else:
                log_info(f"Data file not found: {pt_file} or {self.data_file}")
                self.data_size = 0
                self.num_sequences = 0
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        if self.num_sequences == 0:
            # Return dummy data if file missing
            return {
                "input_ids": torch.zeros(self.seq_length, dtype=torch.long),
                "labels": torch.zeros(self.seq_length, dtype=torch.long)
            }
        
        offset = idx * self.seq_length
        
        if self.is_mmap:
            mmap = np.memmap(str(self.data_file), dtype=np.uint32, mode='r')
            # Read seq_length + 1 to have inputs and targets
            tokens = torch.from_numpy(mmap[offset:offset + self.seq_length + 1].astype(np.int64))
        else:
            data = getattr(self, "data", None)
            if data is None:
                tokens = torch.zeros(self.seq_length + 1, dtype=torch.long)
            else:
                tokens = data[offset:offset + self.seq_length + 1].long()
        
        # Pad if last batch is smaller (though we usually drop last or handle it)
        if len(tokens) < self.seq_length + 1:
            padding = torch.zeros(self.seq_length + 1 - len(tokens), dtype=torch.long)
            tokens = torch.cat([tokens, padding])

        # Shift for autoregressive
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }


class DistributedTrainer:
    """DDP Trainer for French LLM"""
    
    def __init__(self, model, optimizer, scheduler, args, rank, world_size, gpu_id):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.gpu_id = gpu_id
        
        # Wrap model with DDP
        self.model = DDP(
            model,
            device_ids=[gpu_id],
            output_device=gpu_id,
            find_unused_parameters=False,
            broadcast_buffers=True,
            gradient_as_bucket_view=True
        )
        
        # Training state
        self.global_step = 0
        self.start_step = 0
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "throughput": []
        }
        self.checkpoint_dir = Path(f"trained_models/runs/{args.run_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.checkpoint_dir / "training_log.jsonl"
        
        # TensorBoard Writer (Rank 0 only)
        self.writer = None
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir))
        
        # Load checkpoint if provided
        if args.resume_from:
            self._load_checkpoint(args.resume_from)
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming"""
        if not os.path.exists(checkpoint_path):
            log_info(f"Checkpoint not found: {checkpoint_path}")
            return
        
        log_info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        # Load model state
        self.model.module.load_state_dict(checkpoint['model_state'])
        
        # Load training state
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        if 'step' in checkpoint:
            self.global_step = checkpoint['step']
            self.start_step = checkpoint['step']
        
        log_info(f"Resumed from step {self.global_step}")
        dist.barrier()  # Sync all processes
    
    def save_checkpoint(self, step=None):
        """Save checkpoint - only rank 0"""
        if self.rank != 0:
            return
        
        step = step or self.global_step
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        checkpoint = {
            'step': step,
            'model_state': self.model.module.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'args': self.args.__dict__,
            'metrics': self.metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        log_info(f"Saved checkpoint: {checkpoint_path}")
        dist.barrier()  # Sync all processes
    
    def log_metrics(self, loss, lr):
        """Log training metrics"""
        self.metrics["train_loss"].append(loss)
        self.metrics["learning_rate"].append(lr)
        
        if self.rank == 0:
            # TensorBoard
            if self.writer:
                self.writer.add_scalar("Train/Loss", loss, self.global_step)
                self.writer.add_scalar("Train/LR", lr, self.global_step)
                self.writer.add_scalar("Train/GPU_Mem", torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024, self.global_step)

            # Write to JSONL
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "step": self.global_step,
                "loss": loss,
                "lr": lr,
                "gpu_memory_mb": torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    
    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        step_times = []
        
        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()
            
            # Move to GPU
            input_ids = batch["input_ids"].to(self.gpu_id)
            labels = batch["labels"].to(self.gpu_id)
            
            # Forward pass
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            (loss / self.args.gradient_accumulation_steps).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1
                
                # Log metrics
                loss_item = loss.item()
                lr = self.optimizer.param_groups[0]["lr"]
                self.log_metrics(loss_item, lr)
                
                # Timing
                step_time = time.time() - step_start
                step_times.append(step_time)
                throughput = self.args.batch_size * self.world_size / step_time
                self.metrics["throughput"].append(throughput)
                
                # Log progress
                if self.global_step % self.args.log_interval == 0 and self.rank == 0:
                    avg_time = np.mean(step_times[-100:])
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {loss_item:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {avg_time:.2f}s | "
                        f"Throughput: {throughput:.1f} samples/s | "
                        f"GPU Mem: {torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024:.0f}MB"
                    )
                
                # Save checkpoint
                if self.global_step % self.args.checkpoint_interval == 0:
                    self.save_checkpoint()
                
                # Check max steps
                if self.global_step >= self.args.max_steps:
                    return True  # Training complete
        
        return False  # Continue training


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU DDP Training")
    parser.add_argument("--max-steps", type=int, default=250000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data_clean")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=2500)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--run-name", type=str, default="french_medium_multi_gpu")
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, gpu_id = setup_distributed()
    
    log_info("=" * 80)
    log_info(f"Multi-GPU Training Started")
    log_info(f"Rank: {rank}/{world_size} | GPU: {gpu_id}")
    log_info(f"Max Steps: {args.max_steps} | Batch Size: {args.batch_size}")
    log_info(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    log_info(f"Effective Batch: {args.batch_size * args.gradient_accumulation_steps * world_size}")
    log_info("=" * 80)
    
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=32000, d_model=1024, nhead=16, num_layers=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
                num_layers=num_layers
            )
            self.fc = nn.Linear(d_model, vocab_size)
            self.loss_fct = nn.CrossEntropyLoss()

        def forward(self, input_ids, labels=None):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            logits = self.fc(x)
            
            loss = None
            if labels is not None:
                loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
            return SimpleNamespace(loss=loss, logits=logits)

    # Create model
    model = SimpleTransformer()
    
    # Move to GPU
    model = model.to(gpu_id)
    
    # Setup optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=1e-5
    )
    
    # Create trainer
    trainer = DistributedTrainer(
        model, optimizer, scheduler, args, rank, world_size, gpu_id
    )
    
    # Create dataset & dataloader
    dataset = SimpleTokenDataset(args.data_dir, args.seq_length, split="train")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    log_info(f"Dataset size: {len(dataset)} | Batch size: {args.batch_size}")
    
    # Training loop
    epoch = 0
    while trainer.global_step < args.max_steps:
        epoch += 1
        sampler.set_epoch(epoch)
        
        log_info(f"Epoch {epoch} - Step {trainer.global_step}/{args.max_steps}")
        
        if trainer.train_epoch(dataloader):
            break
    
    # Final checkpoint
    trainer.save_checkpoint()
    
    if trainer.writer:
        trainer.writer.close()
    
    log_info(f"Training complete!")
    log_info(f"Final step: {trainer.global_step}")
    log_info(f"Logs: {trainer.log_file}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
