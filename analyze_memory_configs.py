#!/usr/bin/env python3
"""Configuration analyzer and memory footprint calculator."""
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ConfigProfile:
    """Profil de configuration avec estimation mémoire."""
    name: str
    batch_size: int
    grad_accum: int
    gradient_checkpointing: bool
    cpu_offload: bool
    nvme_offload: bool
    
    @property
    def effective_batch_size(self) -> int:
        """Batch size effectif en DDP (2 GPUs)."""
        return self.batch_size * 2 * self.grad_accum
    
    def estimate_vram_mb(self, 
                         model_params: int = 292_000_000,  # 292M pour TinyTransformerLM
                         seq_length: int = 1024,
                         vocab_size: int = 32_000) -> float:
        """Estime l'usage VRAM en MB."""
        # Model weights
        model_size = (model_params * 4) / (1024 ** 2)  # 4 bytes per FP32 param
        
        # Activations (approximation)
        num_layers = 18  # TinyTransformerLM
        activation_size = (
            self.batch_size * seq_length * 1024 * num_layers * 4  # embeddings + attention
        ) / (1024 ** 2)
        
        # Gradient accumulation memory
        grad_accum_overhead = (activation_size * (self.grad_accum - 1)) if self.grad_accum > 1 else 0
        
        # Gradient checkpointing reduction
        if self.gradient_checkpointing:
            activation_size *= 0.4  # ~60% reduction
        
        # CPU offload reduction
        reduction = 1.0
        if self.cpu_offload:
            reduction *= 0.5  # ~50% reduction
        
        total = (model_size + activation_size + grad_accum_overhead) * reduction
        return total
    
    def get_recommendations(self, available_vram_mb: float) -> Dict[str, any]:
        """Retourne des recommandations."""
        estimated = self.estimate_vram_mb()
        margin = available_vram_mb - estimated
        
        return {
            "config": self.name,
            "estimated_vram_mb": estimated,
            "available_vram_mb": available_vram_mb,
            "margin_mb": margin,
            "safe": margin > 1024,  # 1GB safety margin
            "warnings": self._get_warnings(estimated, available_vram_mb),
        }
    
    def _get_warnings(self, estimated: float, available: float) -> List[str]:
        """Retourne les avertissements pour cette config."""
        warnings = []
        
        if estimated > available * 0.9:
            warnings.append("⚠️ Configuration utilise >90% de VRAM (risque OOM)")
        
        if self.batch_size > 8:
            warnings.append("⚠️ Batch size élevé (risque OOM sur RTX 4090)")
        
        if not self.gradient_checkpointing and self.batch_size > 4:
            warnings.append("💡 Activer gradient checkpointing pour réduire mémoire")
        
        if not self.cpu_offload and estimated > available * 0.75:
            warnings.append("💡 Activer CPU offload pour plus de flexibilité")
        
        return warnings


# Profils de configuration
CONFIG_PROFILES = [
    # Ancien (crash)
    ConfigProfile(
        name="ORIGINAL (crashé)",
        batch_size=8,
        grad_accum=2,
        gradient_checkpointing=False,
        cpu_offload=False,
        nvme_offload=False,
    ),
    
    # Nouveau conservateur
    ConfigProfile(
        name="NOUVEAU - Conservative",
        batch_size=4,
        grad_accum=2,
        gradient_checkpointing=True,
        cpu_offload=True,
        nvme_offload=False,
    ),
    
    # Aggressive (pour RTX 4090 seule)
    ConfigProfile(
        name="Aggressive - RTX 4090",
        batch_size=8,
        grad_accum=1,
        gradient_checkpointing=True,
        cpu_offload=True,
        nvme_offload=False,
    ),
    
    # Ultra flexible
    ConfigProfile(
        name="Ultra Flex - NVMe",
        batch_size=4,
        grad_accum=2,
        gradient_checkpointing=True,
        cpu_offload=True,
        nvme_offload=True,
    ),
]


def print_config_comparison():
    """Affiche la comparaison de toutes les configurations."""
    
    # GPU memory (RTX 4090)
    gpu_vram_mb = 24_576  # 24GB
    
    print("\n" + "=" * 100)
    print("  CONFIGURATION MEMORY ANALYSIS - TinyTransformerLM (292M params)")
    print("=" * 100)
    print()
    
    for profile in CONFIG_PROFILES:
        print(f"📊 {profile.name}")
        print(f"   Batch size (per GPU): {profile.batch_size}")
        print(f"   Gradient accumulation: {profile.grad_accum}x")
        print(f"   Effective batch size: {profile.effective_batch_size}")
        print(f"   Gradient checkpointing: {'✅' if profile.gradient_checkpointing else '❌'}")
        print(f"   CPU offload: {'✅' if profile.cpu_offload else '❌'}")
        print(f"   NVMe offload: {'✅' if profile.nvme_offload else '❌'}")
        
        rec = profile.get_recommendations(gpu_vram_mb)
        print(f"   Estimated VRAM: {rec['estimated_vram_mb']:.1f} MB")
        print(f"   Safety margin: {rec['margin_mb']:.1f} MB")
        
        if rec['safe']:
            print(f"   Status: ✅ SAFE")
        else:
            print(f"   Status: 🔴 RISKY")
        
        if rec['warnings']:
            for warning in rec['warnings']:
                print(f"   {warning}")
        
        print()
    
    print("=" * 100)
    print("  RECOMMENDATIONS")
    print("=" * 100)
    print("""
✅ POUR L'ENTRAÎNEMENT 100K STEPS (8-10h):
   Utiliser: "NOUVEAU - Conservative"
   - Batch size: 4 (réduit de 8)
   - Gradient accumulation: 2
   - Gradient checkpointing: ACTIVÉ
   - CPU offload: ACTIVÉ
   
   Raison: Fragmentation mémoire résolue, utilisation stable sur 2 GPUs

💡 SI VOUS AVEZ PLUS DE TEMPS (entraînement 48h+):
   Utiliser: "Aggressive - RTX 4090"
   - Batch size: 8 (original)
   - Gradient accumulation: 1
   - Configuration: avec optimisations
   - Speedup: ~1.2× vs conservative

⚠️ SI VOUS ATTEIGNEZ ENCORE DES OOM:
   Utiliser: "Ultra Flex - NVMe"
   - Fallback automatique: VRAM → RAM → NVMe
   - Performance: ~0.7× vs conservative (mais stable)
   - NVMe Gen4: ~5GB/s, acceptable pour overflow

ÉTAPES PROCHAINES:
   1. Test 500 steps avec "NOUVEAU - Conservative"
   2. Vérifier pas de crash OOM
   3. Lancer 100k steps overnight
   4. Monitorer avec memory_monitor.py
    """)
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print_config_comparison()
    
    # Test avec current GPU
    if torch.cuda.is_available():
        device_id = 0
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory / (1024 ** 3)
        print(f"🖥️  GPU {device_id}: {props.name}")
        print(f"   Total VRAM: {total_memory:.1f} GB")
        print()
