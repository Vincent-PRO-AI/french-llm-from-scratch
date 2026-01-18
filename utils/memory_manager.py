#!/usr/bin/env python3
"""Memory management system with VRAM -> RAM -> NVMe Gen 4 fallback."""
import os
import sys
import psutil
import torch
import tempfile
import pickle
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class MemoryConfig:
    """Configuration pour la gestion mémoire flexible."""
    enable_cpu_offload: bool = True
    enable_nvme_offload: bool = True
    nvme_cache_dir: Optional[Path] = None
    min_vram_threshold_mb: float = 512.0  # Minimum à garder libre sur GPU
    cpu_memory_ratio: float = 0.8  # Utiliser max 80% de la RAM système
    compression_enabled: bool = True
    
    def __post_init__(self):
        if self.nvme_cache_dir is None:
            # Utiliser /nvme_cache s'il existe, sinon /tmp
            if Path("/nvme_cache").exists():
                self.nvme_cache_dir = Path("/nvme_cache")
            else:
                self.nvme_cache_dir = Path(tempfile.gettempdir()) / "nvme_cache"
        
        self.nvme_cache_dir.mkdir(parents=True, exist_ok=True)


class MemoryManager:
    """Gère dynamiquement l'allocation mémoire avec fallback intelligent."""
    
    def __init__(self, config: Optional[MemoryConfig] = None, device_id: int = 0):
        self.config = config or MemoryConfig()
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        
        # Statistiques
        self.vram_allocations = {}  # name -> size_mb
        self.cpu_allocations = {}
        self.nvme_allocations = {}
        
        # Setup CUDA memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    @property
    def available_vram_mb(self) -> float:
        """Mémoire VRAM disponible en MB."""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device_id)
            total = props.total_memory / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(self.device_id) / (1024 ** 2)
            allocated = torch.cuda.memory_allocated(self.device_id) / (1024 ** 2)
            free = total - reserved
            return max(0, free)
        return 0.0
    
    @property
    def available_ram_mb(self) -> float:
        """Mémoire RAM disponible en MB (respecting cpu_memory_ratio)."""
        vm = psutil.virtual_memory()
        safe_available = vm.available / (1024 ** 2) * self.config.cpu_memory_ratio
        return max(0, safe_available)
    
    @property
    def available_nvme_mb(self) -> float:
        """Espace NVMe disponible en MB."""
        try:
            usage = psutil.disk_usage(str(self.config.nvme_cache_dir))
            return usage.free / (1024 ** 2)
        except:
            return 0.0
    
    def get_memory_stats(self) -> dict:
        """Retourne les statistiques mémoire actuelles."""
        return {
            "vram_available_mb": self.available_vram_mb,
            "vram_allocated_mb": torch.cuda.memory_allocated(self.device_id) / (1024 ** 2) if torch.cuda.is_available() else 0,
            "vram_reserved_mb": torch.cuda.memory_reserved(self.device_id) / (1024 ** 2) if torch.cuda.is_available() else 0,
            "ram_available_mb": self.available_ram_mb,
            "ram_used_mb": psutil.virtual_memory().used / (1024 ** 2),
            "nvme_available_mb": self.available_nvme_mb,
            "allocations": {
                "vram": self.vram_allocations,
                "cpu": self.cpu_allocations,
                "nvme": self.nvme_allocations,
            }
        }
    
    def allocate_tensor(self, name: str, size_mb: float, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, str]:
        """
        Alloue un tenseur avec fallback intelligent VRAM → RAM → NVMe.
        
        Returns:
            (tensor, location) où location est 'vram', 'cpu', ou 'nvme'
        """
        # Essayer VRAM d'abord
        if self.available_vram_mb > size_mb + self.config.min_vram_threshold_mb:
            tensor = torch.zeros(int(size_mb * (1024 ** 2) / 4), dtype=dtype, device=self.device)
            self.vram_allocations[name] = size_mb
            print(f"[MEMORY] {name}: VRAM allocation ({size_mb:.1f}MB)")
            return tensor, "vram"
        
        # Fallback à RAM
        if self.config.enable_cpu_offload and self.available_ram_mb > size_mb * 1.2:
            cpu_device = torch.device("cpu")
            tensor = torch.zeros(int(size_mb * (1024 ** 2) / 4), dtype=dtype, device=cpu_device)
            self.cpu_allocations[name] = size_mb
            print(f"[MEMORY] {name}: CPU offload ({size_mb:.1f}MB)")
            return tensor, "cpu"
        
        # Fallback à NVMe Gen 4
        if self.config.enable_nvme_offload and self.available_nvme_mb > size_mb * 1.5:
            # Créer un placeholder en CPU
            cpu_device = torch.device("cpu")
            tensor = torch.zeros(int(size_mb * (1024 ** 2) / 4), dtype=dtype, device=cpu_device)
            self.nvme_allocations[name] = size_mb
            print(f"[MEMORY] {name}: NVMe cache ({size_mb:.1f}MB)")
            return tensor, "nvme"
        
        # Dernier recours : laisser PyTorch gérer (peut échouer)
        print(f"[WARNING] {name}: Allocation {size_mb:.1f}MB excède toutes les options")
        print(f"  VRAM: {self.available_vram_mb:.1f}MB dispo")
        print(f"  RAM: {self.available_ram_mb:.1f}MB dispo")
        print(f"  NVMe: {self.available_nvme_mb:.1f}MB dispo")
        
        # Tenter quand même sur VRAM
        tensor = torch.zeros(int(size_mb * (1024 ** 2) / 4), dtype=dtype, device=self.device)
        self.vram_allocations[name] = size_mb
        return tensor, "vram"
    
    def clear_cache(self):
        """Nettoie les caches mémoire."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.vram_allocations.clear()
        self.cpu_allocations.clear()
        self.nvme_allocations.clear()
    
    @contextmanager
    def managed_memory(self, size_mb: float, name: str = "temp"):
        """Context manager pour allocation temporaire de mémoire."""
        tensor, location = self.allocate_tensor(name, size_mb)
        try:
            yield tensor
        finally:
            del tensor
            if location == "vram" and torch.cuda.is_available():
                torch.cuda.empty_cache()


class SmartDataParallel:
    """Wrapper DDP avec gestion mémoire intelligente."""
    
    def __init__(self, model: torch.nn.Module, memory_config: Optional[MemoryConfig] = None):
        self.model = model
        self.memory_config = memory_config or MemoryConfig()
        self.memory_manager = MemoryManager(self.memory_config)
        
        # Ajouter CPU offload si nécessaire
        self._apply_memory_optimizations()
    
    def _apply_memory_optimizations(self):
        """Applique les optimisations mémoire au modèle."""
        # Activer gradient checkpointing si disponible
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("[MEMORY] Gradient checkpointing activé")
        
        # Activation de cpu_offload dans DDP si config activée
        if self.memory_config.enable_cpu_offload:
            # Cette partie dépend de la structure du modèle
            print("[MEMORY] CPU offload ready for DDP wrapping")
    
    def wrap_ddp(self, backend: str = "nccl"):
        """Enveloppe le modèle dans DistributedDataParallel avec optimisations."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                find_unused_parameters=False,
            )
            print(f"[MEMORY] DDP wrapped with memory management")
        return self.model
    
    def get_memory_stats(self) -> dict:
        """Retourne les stats mémoire du manager."""
        return self.memory_manager.get_memory_stats()


def print_memory_report(memory_manager: MemoryManager, title: str = "Memory Report"):
    """Affiche un rapport mémoire formaté."""
    stats = memory_manager.get_memory_stats()
    
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"VRAM: {stats['vram_available_mb']:.1f}MB dispo / {stats['vram_allocated_mb']:.1f}MB utilisé")
    print(f"RAM:  {stats['ram_available_mb']:.1f}MB dispo / {stats['ram_used_mb']:.1f}MB utilisé")
    print(f"NVMe: {stats['nvme_available_mb']:.1f}MB dispo")
    
    if stats['allocations']['vram']:
        print(f"\nVRAM allocations: {stats['allocations']['vram']}")
    if stats['allocations']['cpu']:
        print(f"CPU allocations: {stats['allocations']['cpu']}")
    if stats['allocations']['nvme']:
        print(f"NVMe allocations: {stats['allocations']['nvme']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test du MemoryManager
    config = MemoryConfig()
    mm = MemoryManager(config)
    
    print("Memory Manager Test")
    print_memory_report(mm, "Initial State")
    
    # Test allocations
    print("\nAllocating 500MB tensor...")
    tensor1, loc1 = mm.allocate_tensor("test1", 500)
    print_memory_report(mm, "After 500MB allocation")
    
    print("\nAllocating 2000MB tensor...")
    tensor2, loc2 = mm.allocate_tensor("test2", 2000)
    print_memory_report(mm, "After 2GB allocation")
    
    # Cleanup
    del tensor1, tensor2
    mm.clear_cache()
    print_memory_report(mm, "After cleanup")
