#!/usr/bin/env python3
"""
🚀 PHASE 2B - TRAINING 100% FRANÇAIS (110k → 130k steps)
========================================================
Lance le training sur dataset de conversations 100% françaises.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import time
import sys
from datetime import datetime

# Configuration Phase 2B
CONFIG = {
    "checkpoint": "trained_models/runs/french_v2_phase2a_final/checkpoint_step_110000.pt",
    "dataset": "data_clean/conversations_train_20M_tokenized.pt",
    "max_steps": 130000,
    "lr": 1e-6,
    "batch_size": 10,
    "seq_len": 512,
    "checkpoint_interval": 1000,
    "run_name": "french_v2_phase2b_100pct_fr"
}


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        return device
    return torch.device("cpu")


def load_checkpoint(path, device):
    """Charge checkpoint."""
    print(f"📦 Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


def load_dataset(path, device):
    """Charge dataset pré-tokenisé."""
    print(f"📚 Loading dataset: {path}")
    tokens = torch.load(path, map_location=device)
    print(f"   Tokens: {tokens.shape} ({tokens.numel():,} tokens)")
    return tokens


def create_data_loader(tokens, seq_len=512, batch_size=10, shuffle=True):
    """Crée le DataLoader."""
    print(f"🔄 Creating DataLoader...")
    
    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len // 2):
        seq = tokens[i:i + seq_len]
        if len(seq) == seq_len:
            sequences.append(seq)
    
    if not sequences:
        print("❌ Pas assez de tokens!")
        return None
    
    sequences = torch.stack(sequences)
    print(f"   Sequences: {sequences.shape}")
    
    dataset = TensorDataset(sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"   Batches: {len(loader)}")
    
    return loader


class SimpleTransformer(nn.Module):
    """Modèle Transformer simple."""
    def __init__(self, vocab_size=32000, d_model=1024, nhead=16, num_layers=18, dim_feedforward=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src):
        x = self.embedding(src) * (self.d_model ** 0.5)
        x = self.encoder(x)
        return self.head(x)


def train(model, device, data_loader, num_steps, lr, checkpoint_interval, run_name, start_step=110000):
    """Boucle d'entraînement."""
    
    model = model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    
    # Create run directory
    run_dir = Path("trained_models/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = run_dir / "metrics.jsonl"
    
    print(f"\n🚀 TRAINING PHASE 2B")
    print(f"   Run: {run_name}")
    print(f"   Steps: {start_step:,} → {num_steps:,}")
    print(f"   LR: {lr}")
    print(f"   Dataset: 100% French conversations")
    print()
    
    start_time = time.time()
    best_loss = float('inf')
    
    step = start_step
    batch_idx = 0
    
    while step < num_steps:
        for batch in data_loader:
            if step >= num_steps:
                break
            
            sequences = batch[0].to(device)
            
            # Forward
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                src = sequences[:, :-1]
                tgt = sequences[:, 1:]
                
                output = model(src)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            
            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Log
            elapsed = time.time() - start_time
            speed = (step - start_step) / elapsed if elapsed > 0 else 0
            eta = (num_steps - step) / speed if speed > 0 else 0
            
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
            
            # Metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "loss": current_loss,
                "best_loss": best_loss,
                "speed_steps_per_sec": speed,
                "eta_hours": eta / 3600
            }
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            # Affichage
            if step % 50 == 0:
                print(f"[{step:,}] Loss: {current_loss:.2f} | Speed: {speed:.3f} steps/s | ETA: {eta/3600:.1f}h")
            
            # Checkpoint
            if step % checkpoint_interval == 0:
                ckpt_path = run_dir / f"checkpoint_step_{step}.pt"
                torch.save({
                    'model_state': model.state_dict(),
                    'step': step,
                    'loss': current_loss,
                    'optimizer_state': optimizer.state_dict()
                }, ckpt_path)
                print(f"💾 Checkpoint saved: {ckpt_path.name}")
            
            step += 1
            batch_idx += 1
    
    print(f"\n✅ Training completed!")
    print(f"   Final loss: {current_loss:.2f}")
    print(f"   Best loss: {best_loss:.2f}")
    print(f"   Time: {elapsed/3600:.1f}h")
    
    return run_dir / f"checkpoint_step_{step-1}.pt"


def main():
    print("\n" + "="*70)
    print("🚀 PHASE 2B - 100% FRENCH TRAINING")
    print("="*70)
    
    # Vérifications
    checkpoint_path = Path(CONFIG["checkpoint"])
    dataset_path = Path(CONFIG["dataset"])
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1
    
    print(f"\n✅ Checkpoint: {checkpoint_path.name}")
    print(f"✅ Dataset: {dataset_path.name} (100% FR, 653MB)")
    print(f"✅ Config:")
    print(f"   Max steps: {CONFIG['max_steps']:,}")
    print(f"   Learning rate: {CONFIG['lr']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Seq length: {CONFIG['seq_len']}")
    
    # Setup
    device = get_device()
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    start_step = checkpoint.get('step', 110000)
    
    # Load dataset
    print()
    tokens = load_dataset(dataset_path, device)
    
    # Create DataLoader
    print()
    data_loader = create_data_loader(
        tokens,
        seq_len=CONFIG["seq_len"],
        batch_size=CONFIG["batch_size"]
    )
    
    if data_loader is None:
        return 1
    
    # Create model
    print("\n🏗️  Creating model...")
    model = SimpleTransformer(
        vocab_size=32000,
        d_model=1024,
        nhead=16,
        num_layers=18,
        dim_feedforward=4096
    )
    
    # Load weights
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print("✅ Model weights loaded")
    
    # Train
    print()
    final_ckpt = train(
        model,
        device,
        data_loader,
        num_steps=CONFIG["max_steps"],
        lr=CONFIG["lr"],
        checkpoint_interval=CONFIG["checkpoint_interval"],
        run_name=CONFIG["run_name"],
        start_step=start_step
    )
    
    print(f"\n🎉 PHASE 2B COMPLETE!")
    print(f"   Checkpoint: {final_ckpt}")
    print(f"   Run: {CONFIG['run_name']}")
    
    print("\n" + "="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
