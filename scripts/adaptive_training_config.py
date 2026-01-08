#!/usr/bin/env python3
"""
Adaptive training configuration based on available VRAM, RAM, and NVMe space.
Automatically adjusts batch_size, gradient_accumulation_steps, and enables 
CPU/NVMe offloading if needed.
"""

import os
import psutil
import torch
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResourceConfig:
    """Detected resource configuration."""
    vram_gb: float
    ram_gb: float
    nvme_free_gb: float
    batch_size: int
    gradient_accumulation_steps: int
    use_cpu_offload: bool
    use_nvme_cache: bool
    nvme_cache_path: Optional[Path]
    enable_gradient_checkpointing: bool
    use_mixed_precision: bool


def get_gpu_memory() -> float:
    """Get available GPU memory in GB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0]) / 1024
    except Exception:
        pass
    
    # Fallback: check via torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


def get_system_memory() -> float:
    """Get available system RAM in GB."""
    mem = psutil.virtual_memory()
    return mem.available / 1e9


def get_nvme_space(path: str = "/") -> float:
    """Get available NVMe space in GB."""
    try:
        stat = os.statvfs(path)
        return (stat.f_bavail * stat.f_frsize) / 1e9
    except Exception:
        return 0.0


def detect_resources() -> ResourceConfig:
    """Detect available resources and create config."""
    vram_gb = get_gpu_memory()
    ram_gb = get_system_memory()
    nvme_free_gb = get_nvme_space()
    
    print(f"[ResourceDetection]")
    print(f"  VRAM: {vram_gb:.1f} GB")
    print(f"  RAM: {ram_gb:.1f} GB")
    print(f"  NVMe free: {nvme_free_gb:.1f} GB")
    
    # Adaptive configuration based on VRAM
    if vram_gb >= 24:
        batch_size = 32
        grad_accum = 1
        use_offload = False
        gradient_ckpt = False
    elif vram_gb >= 16:
        batch_size = 16
        grad_accum = 1
        use_offload = False
        gradient_ckpt = False
    elif vram_gb >= 12:
        batch_size = 12
        grad_accum = 1
        use_offload = False
        gradient_ckpt = True  # Save memory
    elif vram_gb >= 8:
        batch_size = 8
        grad_accum = 2  # Simulate larger batch
        use_offload = True
        gradient_ckpt = True
    else:
        # RTX 5080 has 12GB, shouldn't get here, but fallback
        batch_size = 4
        grad_accum = 4
        use_offload = True
        gradient_ckpt = True
    
    # NVMe cache if lots of free space
    use_nvme_cache = nvme_free_gb > 50
    nvme_cache_path = None
    if use_nvme_cache:
        nvme_cache_path = Path("/tmp/torch_cache")
        nvme_cache_path.mkdir(exist_ok=True)
        print(f"  Using NVMe cache at {nvme_cache_path}")
    
    config = ResourceConfig(
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        nvme_free_gb=nvme_free_gb,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        use_cpu_offload=use_offload,
        use_nvme_cache=use_nvme_cache,
        nvme_cache_path=nvme_cache_path,
        enable_gradient_checkpointing=gradient_ckpt,
        use_mixed_precision=True,  # Always use if available
    )
    
    print(f"[TrainingConfig]")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  CPU offload: {config.use_cpu_offload}")
    print(f"  NVMe cache: {config.use_nvme_cache}")
    print(f"  Gradient checkpointing: {config.enable_gradient_checkpointing}")
    print(f"  Mixed precision: {config.use_mixed_precision}")
    
    return config


def apply_memory_optimizations(config: ResourceConfig, model: torch.nn.Module) -> None:
    """Apply memory optimization techniques to the model."""
    if config.enable_gradient_checkpointing:
        # Enable gradient checkpointing to save memory during backward pass
        if hasattr(model, 'encoder'):
            for layer in model.encoder.layers:
                if hasattr(layer, 'gradient_checkpointing'):
                    layer.gradient_checkpointing = True
        print("[MemoryOptim] Gradient checkpointing enabled")
    
    if config.use_cpu_offload:
        # Enable offloading of model parts to CPU
        try:
            # This is a placeholder; full implementation would require
            # more sophisticated model surgery
            print("[MemoryOptim] CPU offload prepared (model-specific setup needed)")
        except Exception as e:
            print(f"[MemoryOptim] CPU offload setup failed: {e}")


def get_cli_args(config: ResourceConfig) -> list[str]:
    """Generate CLI arguments from config."""
    args = [
        f"--batch-size={config.batch_size}",
        f"--gradient-accumulation-steps={config.gradient_accumulation_steps}",
    ]
    
    if config.enable_gradient_checkpointing:
        args.append("--gradient-checkpointing")
    
    if config.use_cpu_offload:
        args.append("--cpu-offload")
    
    if config.use_nvme_cache and config.nvme_cache_path:
        args.append(f"--nvme-cache-path={config.nvme_cache_path}")
    
    return args


if __name__ == "__main__":
    config = detect_resources()
    cli_args = get_cli_args(config)
    print("\nGenerated CLI args:")
    print(" ".join(cli_args))
