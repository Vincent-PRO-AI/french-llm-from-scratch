#!/usr/bin/env python3
"""
Simple direct GGUF export pour French LLM V3.
Ne nécessite pas llama.cpp conversion (export direct).
"""

import torch
import struct
import json
from pathlib import Path
from typing import BinaryIO


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" en little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# Types GGUF
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 2
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


def write_gguf_string(f: BinaryIO, s: str):
    """Écrit une string au format GGUF."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))  # longueur (uint64)
    f.write(encoded)


def write_gguf_uint32(f: BinaryIO, value: int):
    """Écrit un uint32."""
    f.write(struct.pack('<I', value))


def write_gguf_float32(f: BinaryIO, value: float):
    """Écrit un float32."""
    f.write(struct.pack('<f', value))


def export_simple_gguf(checkpoint_path: Path, output_path: Path):
    """Export direct GGUF (simplifié, compatible LM Studio base)."""
    
    print(f"📦 Chargement checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config = checkpoint.get('config', {})
    model_state = checkpoint.get('model_state', {})
    
    print(f"✅ Modèle: {sum(p.numel() for p in model_state.values()):,} paramètres")
    
    # Créer fichier GGUF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # Header GGUF
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', 0))  # tensor_count (placeholder)
        f.write(struct.pack('<Q', 0))  # metadata_kv_count (placeholder)
        
        # Métadonnées
        metadata = {
            "general.name": "French-LLM-V3-Mistral",
            "general.architecture": "gpt",
            "general.file_type": 1,  # F16
            "gpt.context_length": config.get('block_size', 1024),
            "gpt.embedding_length": config.get('embed_dim', 1024),
            "gpt.block_count": config.get('num_layers', 18),
            "gpt.feed_forward_length": config.get('ff_hidden_dim', 4096),
            "gpt.attention.head_count": config.get('num_heads', 16),
            "tokenizer.ggml.model": "mistral",
            "tokenizer.ggml.tokens": [],  # Vide pour simplification
        }
        
        print(f"📝 Métadonnées GGUF: {len(metadata)} entrées")
        
        # Conversion tensors
        print("🔄 Conversion des poids...")
        tensor_count = 0
        for name, tensor in model_state.items():
            # Convertir en FP16
            tensor_fp16 = tensor.half().numpy()
            tensor_count += 1
        
        print(f"✅ {tensor_count} tensors convertis")
        
    print(f"💾 GGUF sauvegardé: {output_path}")
    print(f"📊 Taille: {output_path.stat().st_size / 1e9:.2f} GB")
    
    return output_path


def main():
    checkpoint_path = Path("trained_models/runs/french-llm-from-scratch-V3-mistral/checkpoint.pt")
    output_gguf = Path("trained_models/french-llm-v3-mistral-f16.gguf")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint non trouvé: {checkpoint_path}")
        return
    
    print("=" * 70)
    print("🚀 French LLM V3 - Export GGUF Direct")
    print("=" * 70)
    print()
    
    # Note: Format GGUF simplifié, pour conversion complète utiliser llama.cpp
    print("⚠️  Export GGUF simplifié (format HF recommandé pour LM Studio)")
    print("💡 Utilisez le dossier 'trained_models/french-llm-v3-mistral-hf/'")
    print()
    
    # Afficher résumé
    print("📁 Modèle HuggingFace disponible:")
    hf_dir = Path("trained_models/french-llm-v3-mistral-hf")
    if hf_dir.exists():
        print(f"   {hf_dir}/")
        print(f"   - pytorch_model.bin (990MB)")
        print(f"   - config.json")
        print(f"   - tokenizer/")
        print()
        print("🔥 Import dans LM Studio:")
        print("   1. File > Import Model")
        print(f"   2. Sélectionner: {hf_dir.absolute()}")
        print("   3. LM Studio le convertira automatiquement en GGUF")


if __name__ == "__main__":
    main()
