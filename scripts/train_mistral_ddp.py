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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
    """Tokenized dataset loaded from .pt"""
    def __init__(self, data_path, seq_length=1024, split="train"):
        self.data_path = Path(data_path)
        self.seq_length = seq_length
        self.split = split
        
        if split == "train":
            pt_file = self.data_path / "conversations_mega_train_mistral_tokenized.pt"
        else:
            pt_file = self.data_path / "conversations_mega_test_mistral_tokenized.pt"
            
        if not pt_file.exists():
            raise FileNotFoundError(f"Data file not found: {pt_file}")
            
        log_info(f"Loading dataset from {pt_file}")
        self.data = torch.load(pt_file, map_location="cpu")
        if isinstance(self.data, dict) and 'input_ids' in self.data:
            self.data = self.data['input_ids']
        if isinstance(self.data, list):
            self.data = torch.tensor(self.data)

        self.data_size = len(self.data)
        self.num_sequences = max(1, (self.data_size - 1) // seq_length)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        offset = idx * self.seq_length
        tokens = self.data[offset:offset + self.seq_length + 1].long()
        
        if len(tokens) < self.seq_length + 1:
            padding = torch.zeros(self.seq_length + 1 - len(tokens), dtype=torch.long)
            tokens = torch.cat([tokens, padding])

        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        return {"input_ids": input_ids, "labels": labels}

class DistributedTrainer:
    """DDP Trainer for Mistral 7B (QLoRA)"""
    
    def __init__(self, model, optimizer, scheduler, args, rank, world_size, gpu_id):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.gpu_id = gpu_id
        
        # DDP Wrapper
        self.model = DDP(
            model,
            device_ids=[gpu_id],
            output_device=gpu_id,
            # For QLoRA/PEFT, we often need this because only some params are tuned
            find_unused_parameters=False 
        )
        
        # Training state
        self.global_step = 0
        self.checkpoint_dir = Path(f"trained_models/runs/{args.run_name}")
        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir))
        else:
            self.writer = None
            
        self.log_file = self.checkpoint_dir / "training_log.jsonl"
    
    def save_checkpoint(self, step=None):
        """Save LoRA weights (adapters only for efficiency)"""
        if self.rank != 0:
            return
        
        step = step or self.global_step
        save_path = self.checkpoint_dir / f"checkpoint_step_{step}"
        log_info(f"Saving LoRA adapters to {save_path}")
        
        # Save PEFT model
        self.model.module.save_pretrained(save_path)
        
        # Save training state
        state = {
            'step': step,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        torch.save(state, save_path / "training_state.pt")

    def train_epoch(self, dataloader):
        self.model.train()
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.gpu_id)
            labels = batch["labels"].to(self.gpu_id)
            
            # Forward
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward
            (loss / self.args.gradient_accumulation_steps).backward()
            
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.rank == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    if self.global_step % self.args.log_interval == 0:
                        logger.info(f"Step {self.global_step} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
                        if self.writer:
                            self.writer.add_scalar("Train/Loss", loss.item(), self.global_step)
                            self.writer.add_scalar("Train/LR", lr, self.global_step)
                
                if self.global_step % self.args.checkpoint_interval == 0:
                    self.save_checkpoint()
                
                if self.global_step >= self.args.max_steps:
                    return True
        return False

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Mistral Fine-tuning (QLoRA)")
    parser.add_argument("--model-id", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--run-name", type=str, default="mistral_french_finetune")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    
    rank, world_size, gpu_id = setup_distributed()
    
    # 1. Load Model (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    log_info(f"Loading model {args.model_id} on GPU {gpu_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": gpu_id},
        torch_dtype=torch.bfloat16,
    )
    
    # 2. Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # 3. Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)
    
    # 5. Dataset
    dataset = SimpleTokenDataset("data_clean", split="train")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    # 6. Trainer
    trainer = DistributedTrainer(model, optimizer, scheduler, args, rank, world_size, gpu_id)
    
    log_info("Starting fine-tuning...")
    while trainer.global_step < args.max_steps:
        if trainer.train_epoch(dataloader):
            break
            
    trainer.save_checkpoint()
    cleanup_distributed()

if __name__ == "__main__":
    main()
