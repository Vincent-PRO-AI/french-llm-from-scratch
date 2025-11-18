#!/usr/bin/env python3
"""
quick_train.py

Minimal training harness to test the PyTorch pipeline locally.

Usage example:
  python scripts/quick_train.py --data-dir data_clean/test --max-steps 200 --batch-size 8

This script:
- loads all .txt files under --data-dir
- uses a byte-level tokenizer (raw bytes 0-255)
- creates a tiny Transformer-style model and trains to predict next byte
- saves checkpoints under trained_models/runs/run-<ts>/checkpoint.pt

This is intentionally small and self-contained for quick experiments.
"""

import argparse
import os
import time
from pathlib import Path
import json
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data_clean/test")
    p.add_argument("--log-dir", type=str, default="trained_models/runs")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save-every", type=int, default=100)
    return p.parse_args()


def read_texts(data_dir):
    p = Path(data_dir)
    texts = []
    if not p.exists():
        return texts
    for fn in sorted(p.rglob("*.txt")):
        try:
            with open(fn, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except Exception:
            try:
                with open(fn, "r", encoding="latin-1") as f:
                    texts.append(f.read())
            except Exception:
                continue
    return texts


class BytesDataset(Dataset):
    def __init__(self, texts, seq_len=128):
        # concatenate texts with separators
        data = b"".join((t.encode("utf-8", errors="ignore") + b"\n\n") for t in texts)
        # fallback to random data if empty
        if len(data) < seq_len + 1:
            data = (b"Hello world. \n" * 1024)
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(1, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        # sample a random window for training stability
        start = random.randint(0, max(0, len(self.data) - self.seq_len - 1))
        chunk = self.data[start : start + self.seq_len + 1]
        x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
        y = torch.tensor(list(chunk[1:]), dtype=torch.long)
        return x, y


class TinyMiniGPT(nn.Module):
    def __init__(self, vocab_size=256, emb=128, n_layers=4, n_heads=4, seq_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, emb))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb, nhead=n_heads, dim_feedforward=emb * 4, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(emb)
        self.head = nn.Linear(emb, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (B, L)
        b, l = x.shape
        tok = self.tok_emb(x)  # B L E
        if l != self.seq_len:
            # trim or pad pos_emb
            pos = self.pos_emb[:, :l, :]
        else:
            pos = self.pos_emb
        h = tok + pos
        # transformer expects (L, B, E)
        h = h.permute(1, 0, 2)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)
        h = self.ln(h)
        logits = self.head(h)
        return logits


def make_run_dir(log_dir):
    ts = time.strftime("%Y%m%d-%H%M%S")
    runname = f"run-{ts}"
    path = Path(log_dir) / runname
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_checkpoint(run_dir, step, model, optimizer, metadata=None):
    ckpt = dict(step=step, model_state=model.state_dict(), optim_state=optimizer.state_dict())
    fn = Path(run_dir) / "checkpoint.pt"
    torch.save(ckpt, fn)
    if metadata:
        with open(Path(run_dir) / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    texts = read_texts(args.data_dir)
    print(f"Loaded {len(texts)} text files from {args.data_dir}")
    ds = BytesDataset(texts, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TinyMiniGPT(vocab_size=256, emb=128, n_layers=4, n_heads=4, seq_len=args.seq_len)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_f = nn.CrossEntropyLoss()

    run_dir = make_run_dir(args.log_dir)
    print("Run dir:", run_dir)
    metadata = {"data_dir": args.data_dir, "seq_len": args.seq_len, "batch_size": args.batch_size, "start_time": time.time()}

    step = 0
    while step < args.max_steps:
        for xb, yb in dl:
            model.train()
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)  # B L V
            B, L, V = logits.shape
            loss = loss_f(logits.view(B * L, V), yb.view(B * L))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"step {step}/{args.max_steps} loss={loss.item():.4f}")
            if step % args.save_every == 0 or step >= args.max_steps:
                metadata.update({"step": step, "timestamp": time.time()})
                save_checkpoint(run_dir, step, model, optimizer, metadata=metadata)
                print(f"Saved checkpoint at step {step} -> {run_dir}/checkpoint.pt")
            if step >= args.max_steps:
                break
        # end for
    # end while
    # final save
    metadata.update({"step": step, "end_time": time.time()})
    save_checkpoint(run_dir, step, model, optimizer, metadata=metadata)
    print("Training finished. checkpoint saved to", run_dir)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
