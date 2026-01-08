#!/usr/bin/env python3
"""
Parse training logs from stdout format and generate curves
Format: step=XXX val_loss=Y.YYYY  or  step=XXX train_loss=X.XXXX
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Parse from /tmp/training.log
log_file = "/tmp/training.log"

metrics = defaultdict(list)
steps = set()

print("Parsing training logs...")
try:
    with open(log_file) as f:
        content = f.read()
except:
    print(f"Could not read {log_file}")
    exit(1)

# Extract step and loss pairs
# Patterns: "step=296000 val_loss=4.5010" or "step=300000 train_loss=4.8741"
for line in content.split('\n'):
    if 'step=' in line and 'loss=' in line:
        # Extract step
        step_match = re.search(r'step=(\d+)', line)
        if not step_match:
            continue
        step = int(step_match.group(1))
        
        # Try to find val_loss or train_loss
        val_loss_match = re.search(r'val_loss=([0-9.]+)', line)
        train_loss_match = re.search(r'train_loss=([0-9.]+)', line)
        
        if val_loss_match:
            loss = float(val_loss_match.group(1))
            metrics['val_loss'].append((step, loss))
            steps.add(step)
        elif train_loss_match:
            loss = float(train_loss_match.group(1))
            metrics['train_loss'].append((step, loss))
            steps.add(step)

# Sort by step
for key in metrics:
    metrics[key].sort(key=lambda x: x[0])

print(f"\nMetrics extracted:")
if metrics['train_loss']:
    print(f"  Train losses: {len(metrics['train_loss'])} points")
    print(f"    Range: {metrics['train_loss'][0][0]} → {metrics['train_loss'][-1][0]} steps")
    print(f"    Loss range: {min(m[1] for m in metrics['train_loss']):.4f} → {max(m[1] for m in metrics['train_loss']):.4f}")

if metrics['val_loss']:
    print(f"  Val losses: {len(metrics['val_loss'])} points")
    print(f"    Range: {metrics['val_loss'][0][0]} → {metrics['val_loss'][-1][0]} steps")
    print(f"    Loss range: {min(m[1] for m in metrics['val_loss']):.4f} → {max(m[1] for m in metrics['val_loss']):.4f}")

if not metrics['train_loss'] and not metrics['val_loss']:
    print("  No metrics found! The training may still be initializing.")
    exit(0)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training Progress - French LLM (160k → 300k steps)", fontsize=13, fontweight="bold")

# Extract data for plotting
steps_train = []
losses_train = []
if metrics['train_loss']:
    steps_train, losses_train = zip(*metrics['train_loss'])
    steps_train = list(steps_train)
    losses_train = list(losses_train)

steps_val = []
losses_val = []
if metrics['val_loss']:
    steps_val, losses_val = zip(*metrics['val_loss'])
    steps_val = list(steps_val)
    losses_val = list(losses_val)

# Plot 1: Training Loss
if steps_train and losses_train:
    axes[0].plot(steps_train, losses_train, 'b.-', linewidth=1, markersize=4, label='Train Loss')
    
    # Smoothed curve
    if len(losses_train) > 10:
        window = min(20, len(losses_train) // 5)
        losses_smooth = np.convolve(losses_train, np.ones(window)/window, mode='valid')
        steps_smooth = steps_train[window-1:]
        axes[0].plot(steps_smooth, losses_smooth, 'r-', linewidth=2, label=f'Smoothed ({window}-step MA)')
    
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

# Plot 2: Validation Loss (if available)
if steps_val and losses_val:
    axes[1].plot(steps_val, losses_val, 'g-o', linewidth=2, markersize=6, label='Validation Loss')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
else:
    # Show combined if val_loss not available
    axes[1].plot(steps_train, losses_train, 'b.-', linewidth=1, markersize=4, label='Training Progress')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Progress (No validation yet)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

plt.tight_layout()

# Save
run_dir = Path("trained_models/runs/french_large_filtered_LIVE_v4")
output_path = run_dir / "training_curves_live.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Curves saved to {output_path}")

# Summary
if steps_train and losses_train:
    print(f"\nTrain Loss Progress:")
    print(f"  Step {steps_train[0]:6d}: loss = {losses_train[0]:.4f}")
    print(f"  Step {steps_train[-1]:6d}: loss = {losses_train[-1]:.4f}")
    improvement = (losses_train[0] - losses_train[-1]) / losses_train[0] * 100
    print(f"  Improvement: {improvement:.1f}% ({losses_train[0] - losses_train[-1]:.4f})")

if steps_val and losses_val:
    print(f"\nValidation Loss Progress:")
    print(f"  Step {steps_val[0]:6d}: loss = {losses_val[0]:.4f}")
    print(f"  Step {steps_val[-1]:6d}: loss = {losses_val[-1]:.4f}")
    improvement_val = (losses_val[0] - losses_val[-1]) / losses_val[0] * 100
    print(f"  Improvement: {improvement_val:.1f}% ({losses_val[0] - losses_val[-1]:.4f})")

print("\n✓ Done!")
