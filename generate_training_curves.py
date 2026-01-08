#!/usr/bin/env python3
"""
Generate training curves (loss and learning rate) from metrics.jsonl
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Paths
run_dir = Path("trained_models/runs/french_large_filtered_LIVE_v4")
metrics_file = run_dir / "metrics.jsonl"

# Parse metrics
steps_train = []
losses_train = []
steps_val = []
losses_val = []
steps_lr = []
lrs = []

print("Parsing metrics.jsonl...")
with open(metrics_file) as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            step = data.get("step")
            split = data.get("split")
            loss = data.get("loss")
            lr = data.get("learning_rate")
            
            if step and loss:
                if split == "train":
                    steps_train.append(step)
                    losses_train.append(loss)
                elif split == "val":
                    steps_val.append(step)
                    losses_val.append(loss)
            
            if step and lr:
                steps_lr.append(step)
                lrs.append(lr)
        except json.JSONDecodeError:
            continue
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1} lines...")

print(f"\nMetrics summary:")
print(f"  Train losses: {len(losses_train)} points (steps {min(steps_train) if steps_train else 'N/A'} to {max(steps_train) if steps_train else 'N/A'})")
print(f"  Val losses: {len(losses_val)} points")
print(f"  Learning rates: {len(lrs)} points")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Training Curves - French LLM (160k → 300k steps)", fontsize=14, fontweight="bold")

# 1. Train Loss
if steps_train and losses_train:
    axes[0, 0].plot(steps_train, losses_train, "b-", linewidth=1, label="Train Loss")
    # Smoothed line
    window = min(50, len(losses_train) // 4)
    if window > 1:
        losses_smooth = np.convolve(losses_train, np.ones(window)/window, mode="valid")
        steps_smooth = steps_train[window-1:]
        axes[0, 0].plot(steps_smooth, losses_smooth, "r-", linewidth=2, label="Smoothed (50-step MA)")
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss Over Time")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([min(losses_train)*0.9, max(losses_train)*1.05])

# 2. Validation Loss
if steps_val and losses_val:
    axes[0, 1].plot(steps_val, losses_val, "g-", linewidth=1.5, marker="o", markersize=4, label="Val Loss")
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Validation Loss (Every 1000 Steps)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

# 3. Train vs Val (overlay for final comparison)
if steps_train and losses_train and steps_val and losses_val:
    axes[1, 0].plot(steps_train, losses_train, "b-", linewidth=0.8, alpha=0.6, label="Train Loss")
    axes[1, 0].plot(steps_val, losses_val, "g-", linewidth=2, marker="o", markersize=5, label="Val Loss")
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Train vs Validation Loss")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([min(min(losses_train), min(losses_val))*0.95, 
                         max(max(losses_train), max(losses_val))*1.05])

# 4. Learning Rate Schedule
if steps_lr and lrs:
    axes[1, 1].plot(steps_lr, lrs, "orange", linewidth=2, label="Learning Rate")
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate Schedule (CosineAnnealingLR)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

plt.tight_layout()
output_path = run_dir / "training_curves.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n✓ Saved training curves to: {output_path}")

# Print summary statistics
if losses_train:
    print(f"\nTrain Loss Summary:")
    print(f"  Initial: {losses_train[0]:.4f}")
    print(f"  Final: {losses_train[-1]:.4f}")
    print(f"  Min: {min(losses_train):.4f}")
    print(f"  Max: {max(losses_train):.4f}")
    print(f"  Improvement: {(losses_train[0] - losses_train[-1]):.4f} ({(losses_train[0] - losses_train[-1])/losses_train[0]*100:.1f}%)")

if losses_val:
    print(f"\nValidation Loss Summary:")
    print(f"  Initial: {losses_val[0]:.4f}")
    print(f"  Final: {losses_val[-1]:.4f}")
    print(f"  Min: {min(losses_val):.4f}")
    print(f"  Max: {max(losses_val):.4f}")
    print(f"  Improvement: {(losses_val[0] - losses_val[-1]):.4f} ({(losses_val[0] - losses_val[-1])/losses_val[0]*100:.1f}%)")

print("\n✓ Done!")
