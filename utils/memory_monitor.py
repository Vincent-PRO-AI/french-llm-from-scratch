#!/usr/bin/env python3
"""Real-time memory monitoring for DDP training."""
import torch
import psutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import time


class MemoryMonitor:
    """Monitore l'utilisation mémoire pendant l'entraînement."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self.history = []
    
    def get_gpu_memory(self, device_id: int = 0) -> Dict[str, float]:
        """Retourne l'usage mémoire GPU en MB."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}
        
        props = torch.cuda.get_device_properties(device_id)
        total = props.total_memory / (1024 ** 2)
        allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
        free = total - reserved
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": total,
        }
    
    def get_system_memory(self) -> Dict[str, float]:
        """Retourne l'usage mémoire système en MB."""
        vm = psutil.virtual_memory()
        return {
            "used": vm.used / (1024 ** 2),
            "available": vm.available / (1024 ** 2),
            "total": vm.total / (1024 ** 2),
            "percent": vm.percent,
        }
    
    def get_nvme_space(self, nvme_path: Path = Path("/nvme_cache")) -> Dict[str, float]:
        """Retourne l'espace disque NVMe en MB."""
        if not nvme_path.exists():
            return {"used": 0, "free": 0, "total": 0}
        
        try:
            usage = psutil.disk_usage(str(nvme_path))
            return {
                "used": usage.used / (1024 ** 2),
                "free": usage.free / (1024 ** 2),
                "total": usage.total / (1024 ** 2),
            }
        except:
            return {"used": 0, "free": 0, "total": 0}
    
    def get_snapshot(self, device_id: int = 0) -> Dict:
        """Retourne une snapshot complète d'utilisation mémoire."""
        return {
            "timestamp": datetime.now().isoformat(),
            "gpu": self.get_gpu_memory(device_id),
            "system": self.get_system_memory(),
            "nvme": self.get_nvme_space(),
        }
    
    def print_report(self, device_id: int = 0, title: str = "Memory Report"):
        """Affiche un rapport formaté."""
        snap = self.get_snapshot(device_id)
        gpu = snap["gpu"]
        system = snap["system"]
        nvme = snap["nvme"]
        
        print(f"\n{'='*70}")
        print(f"  {title} @ {snap['timestamp']}")
        print(f"{'='*70}")
        
        # GPU
        print(f"🖥️  GPU {device_id}:")
        print(f"   Allocated: {gpu['allocated']:.1f} MB / {gpu['total']:.1f} MB")
        print(f"   Reserved:  {gpu['reserved']:.1f} MB / {gpu['total']:.1f} MB")
        print(f"   Free:      {gpu['free']:.1f} MB")
        gpu_usage_pct = (gpu['allocated'] / gpu['total']) * 100 if gpu['total'] > 0 else 0
        print(f"   Usage:     {gpu_usage_pct:.1f}%")
        
        # System RAM
        print(f"\n💾 System RAM:")
        print(f"   Used:      {system['used']:.1f} MB / {system['total']:.1f} MB")
        print(f"   Available: {system['available']:.1f} MB")
        print(f"   Usage:     {system['percent']:.1f}%")
        
        # NVMe
        if nvme['total'] > 0:
            print(f"\n📦 NVMe Cache:")
            print(f"   Used:      {nvme['used']:.1f} MB / {nvme['total']:.1f} MB")
            print(f"   Available: {nvme['free']:.1f} MB")
            nvme_usage_pct = (nvme['used'] / nvme['total']) * 100 if nvme['total'] > 0 else 0
            print(f"   Usage:     {nvme_usage_pct:.1f}%")
        
        # Warnings
        print(f"\n⚠️  Alerts:")
        if gpu_usage_pct > 90:
            print(f"   🔴 GPU pressure HIGH ({gpu_usage_pct:.1f}%)")
        elif gpu_usage_pct > 75:
            print(f"   🟡 GPU pressure MEDIUM ({gpu_usage_pct:.1f}%)")
        else:
            print(f"   🟢 GPU pressure OK ({gpu_usage_pct:.1f}%)")
        
        if system['percent'] > 85:
            print(f"   🔴 RAM pressure HIGH ({system['percent']:.1f}%)")
        elif system['percent'] > 70:
            print(f"   🟡 RAM pressure MEDIUM ({system['percent']:.1f}%)")
        else:
            print(f"   🟢 RAM pressure OK ({system['percent']:.1f}%)")
        
        print(f"{'='*70}\n")
    
    def save_snapshot(self, device_id: int = 0):
        """Sauvegarde une snapshot dans l'historique."""
        self.history.append(self.get_snapshot(device_id))
        
        if self.log_file and len(self.history) % 10 == 0:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(self.history[-1]) + '\n')


def monitor_training_continuous(
    log_file: Optional[Path] = None,
    interval: int = 30,
    num_gpus: int = 2
):
    """Monitore continu pendant l'entraînement."""
    monitor = MemoryMonitor(log_file)
    print(f"🔍 Starting continuous monitoring (every {interval}s)...")
    
    try:
        while True:
            for device_id in range(num_gpus):
                monitor.save_snapshot(device_id)
                monitor.print_report(device_id, f"GPU {device_id} Report")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n✅ Monitoring stopped")


if __name__ == "__main__":
    import sys
    
    # Test basic reporting
    monitor = MemoryMonitor()
    
    print("Memory Monitor - Initial Report")
    monitor.print_report()
    
    # Test for 10 seconds with snapshots
    print("Collecting memory snapshots for 10 seconds...")
    for i in range(5):
        monitor.save_snapshot()
        time.sleep(2)
        monitor.print_report(title=f"Snapshot {i+1}")
    
    print("✅ Memory monitor test complete")
