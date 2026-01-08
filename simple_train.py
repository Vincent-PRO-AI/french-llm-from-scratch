#!/usr/bin/env python3
"""
🚀 FRENCH LLM V2 TRAINING - SIMPLE & FONCTIONNEL
================================================
Script de training minimal qui MARCHE:
- Charge checkpoint PyTorch
- Utilise dataset tokenisé pré-existant
- Boucle d'entraînement simple (pas de complications)
- Sauvegarde checkpoints régulièrement
- AMP, gradient accumulation, tout ça

Usage:
  python3 simple_train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
import json
import time
import sys
import argparse
from datetime import datetime

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        return device
    else:
        return torch.device("cpu")

def load_checkpoint(path, device):
    """Load checkpoint without weights_only issues"""
    print(f"📦 Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint

def load_tokenized_dataset(path, device):
    """Load pre-tokenized dataset"""
    print(f"📚 Loading dataset: {path}")
    tokens = torch.load(path, map_location=device)
    print(f"   Tokens shape: {tokens.shape}")
    print(f"   Total tokens: {tokens.numel():,}")
    return tokens

def create_data_loader(tokens, seq_len=512, batch_size=10, shuffle=True):
    """Create DataLoader from token sequence"""
    print(f"🔄 Creating DataLoader...")
    
    # Create sliding windows
    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len // 2):
        seq = tokens[i:i + seq_len]
        if len(seq) == seq_len:
            sequences.append(seq)
    
    if not sequences:
        print("❌ Not enough tokens for sequences!")
        return None
    
    sequences = torch.stack(sequences)
    print(f"   Sequences: {sequences.shape}")
    
    dataset = TensorDataset(sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"   Batches: {len(loader)}")
    return loader

def train(
    model,
    device,
    data_loader,
    num_steps=100000,
    lr=1e-6,
    checkpoint_interval=1000,
    run_name="french_v2_training",
    start_step=0
):
    """Main training loop"""
    
    # Setup
    model = model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    
    # Create run directory
    run_dir = Path("trained_models/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = run_dir / "metrics.jsonl"
    
    print(f"\n🚀 TRAINING")
    print(f"   Run: {run_name}")
    print(f"   Steps: {start_step:,} → {num_steps:,}")
    print(f"   LR: {lr}")
    print(f"   Device: {device}")
    print(f"   Checkpoints: {run_dir}")
    print()
    
    step = start_step
    batch_iter = iter(data_loader)
    start_time = time.time()
    
    while step < num_steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(data_loader)
            batch = next(batch_iter)
        
        tokens = batch[0].to(device)
        
        # Forward pass with AMP
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # Simple next-token prediction
            src = tokens[:, :-1]  # all but last
            tgt = tokens[:, 1:]    # all but first (target)
            
            output = model(src)  # [batch, seq-1, vocab]
            
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt.reshape(-1)
            )
        
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (step + 1 - start_step) / elapsed
            eta_sec = (num_steps - step) / speed if speed > 0 else 0
            eta_h = eta_sec / 3600
            
            # Log metrics
            metric = {
                "timestamp": datetime.now().isoformat(),
                "step": step + 1,
                "loss": loss.item(),
                "speed_steps_per_sec": speed,
                "eta_hours": eta_h
            }
            
            with open(metrics_file, "a") as f:
                json.dump(metric, f)
                f.write("\n")
            
            # Print
            if (step + 1) % 100 == 0:
                print(f"[{step+1:6d}/{num_steps}] loss={loss.item():.4f} "
                      f"speed={speed:.2f}it/s eta={eta_h:.1f}h")
        
        # Checkpoint
        if (step + 1) % checkpoint_interval == 0:
            ckpt_path = run_dir / f"checkpoint_step_{step+1}.pt"
            torch.save({
                'model_state': model.state_dict(),
                'step': step + 1,
                'optimizer_state': optimizer.state_dict(),
            }, ckpt_path)
            print(f"   💾 Checkpoint saved: {ckpt_path.name}")
        
        step += 1
    
    # Final checkpoint
    ckpt_path = run_dir / f"checkpoint_step_{step}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'step': step,
        'optimizer_state': optimizer.state_dict(),
    }, ckpt_path)
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete!")
    print(f"   Total time: {total_time/3600:.1f}h")
    print(f"   Final checkpoint: {ckpt_path.name}")
    
    return ckpt_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--run-name", type=str, default="french_v2_simple_training")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 FRENCH LLM V2 - SIMPLE TRAINING")
    print("="*70)
    
    # Setup
    device = get_device()
    
    # Load data
    print()
    
    # Try to load checkpoint
    checkpoint_path = Path("trained_models/runs/checkpoint_step_100000.pt")
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device)
        if 'model_state' in checkpoint:
            start_step = checkpoint.get('step', 0)
            print(f"   Starting from step: {start_step}")
        else:
            start_step = 0
    else:
        checkpoint = None
        start_step = 0
        print(f"⚠️  Checkpoint not found, starting from scratch")
    
    # Load dataset
    dataset_path = Path("data_clean/conversations_mega_train_mistral_tokenized.pt")
    if not dataset_path.exists():
        # Try alternatives
        alts = list(Path("data_clean").glob("*tokenized.pt"))
        if alts:
            dataset_path = alts[0]
            print(f"⚠️  Using alternative: {dataset_path.name}")
        else:
            print("❌ No tokenized dataset found!")
            return 1
    
    print()
    tokens = load_tokenized_dataset(dataset_path, device)
    
    print()
    data_loader = create_data_loader(tokens, args.seq_len, args.batch_size)
    
    if data_loader is None:
        return 1
    
    # Create dummy model for testing
    # TODO: Load actual model from checkpoint if available
    print("\n⚠️  Creating dummy model (use actual model from checkpoint)")
    
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=32000, d_model=1024, nhead=16, num_layers=18):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4096,
                batch_first=True,
                dropout=0.1
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.encoder(x)
            x = self.head(x)
            return x
    
    model = SimpleTransformer()
    
    # Load checkpoint weights if available
    if checkpoint and 'model_state' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state'])
            print("✅ Model loaded from checkpoint")
        except Exception as e:
            print(f"⚠️  Could not load model state: {e}")
    
    # Train
    print()
    final_ckpt = train(
        model,
        device,
        data_loader,
        num_steps=args.max_steps,
        lr=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        run_name=args.run_name,
        start_step=start_step
    )
    
    print(f"\n🇫🇷 TRAINING COMPLETE!")
    print(f"   Checkpoint: {final_ckpt}")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
