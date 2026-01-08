#!/usr/bin/env python3
"""
🚀 FRENCH LLM V3 - TRAINING SCRIPT (LLaMA 260M avec HuggingFace)
==============================================================
Entraîner un modèle LLaMA 260M sur corpus français avec SentencePiece tokenizer.
Output: HF-compatible safetensors + config.json + GGUF-convertible
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from pathlib import Path
import json
from datasets import Dataset as HFDataset
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model_name": "French-LLM-V3-LLaMA-260M",
    "checkpoint": None,  # Nouveau training (from scratch)
    "dataset_paths": [
        "data_clean/conversations_mega_train_mistral_tokenized.pt",  # 653M tokens 100% FR
    ],
    "tokenizer_path": "trained_models/tokenizers/vincent_tokenizer_FR_v2/tokenizer.model",
    "output_dir": "trained_models/runs/french_v3_llama_260m",
    "max_steps": 80000,
    "warmup_steps": 2000,
    "learning_rate": 2e-4,  # Légèrement plus agressif
    "lr_scheduler": "cosine",
    "batch_size": 16,  # OPTIMISÉ : équilibre batch/VRAM
    "gradient_accumulation_steps": 1,  # effective batch = 16
    "seq_length": 2048,
    "save_steps": 2000,
    "eval_steps": 500,
    "logging_steps": 100,
    "num_workers": 4,
    "bf16": True,  # bfloat16 pour RTX 5080
    "tf32": True,  # TensorFloat32 pour RTX 5080
    "flash_attention": True,  # Flash Attention 2
    "gradient_checkpointing": True,  # Réduire mémoire
    "use_cache": False,  # Désactiver pendant training (activation pendant inference)
}


# ============================================================================
# DATASET LOADER
# ============================================================================

def load_tokenized_dataset(dataset_paths: list, tokenizer_path: str, seq_length: int = 2048):
    """Load pre-tokenized datasets."""
    print(f"\n📚 Loading datasets...")
    
    all_ids = []
    for path in dataset_paths:
        if Path(path).exists():
            print(f"   Loading: {path}")
            tokens = torch.load(path, map_location='cpu')
            all_ids.extend(tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens)
            print(f"   → {len(tokens):,} tokens")
    
    print(f"   Total: {len(all_ids):,} tokens")
    
    # Create sequences
    sequences = []
    for i in range(0, len(all_ids) - seq_length, seq_length // 2):
        seq = all_ids[i:i + seq_length]
        if len(seq) == seq_length:
            sequences.append(seq)
    
    print(f"   Sequences: {len(sequences):,}")
    
    return sequences


def create_hf_dataset(sequences: list):
    """Convert to HuggingFace Dataset format."""
    
    def generator():
        for seq in sequences:
            yield {"input_ids": seq, "attention_mask": [1] * len(seq)}
    
    dataset = HFDataset.from_generator(generator)
    
    # Split train/test (85/15)
    split = dataset.train_test_split(test_size=0.15, seed=42)
    
    return split["train"], split["test"]


# ============================================================================
# TRAINING
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 FRENCH LLM V3 - LLaMA 260M TRAINING")
    print("="*70)
    
    # Vérifications
    tokenizer_path = Path(CONFIG["tokenizer_path"])
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return 1
    
    # Load config
    config_path = Path("french_llama_config.json")
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1
    
    with open(config_path) as f:
        model_config = json.load(f)
    
    print(f"\n📋 Configuration:")
    print(f"   Model: LlamaForCausalLM (260M)")
    print(f"   Max steps: {CONFIG['max_steps']:,}")
    print(f"   LR: {CONFIG['learning_rate']}")
    print(f"   Batch size: {CONFIG['batch_size']} × {CONFIG['gradient_accumulation_steps']} accumulation")
    print(f"   Seq length: {CONFIG['seq_length']}")
    print(f"   Tokenizer: {CONFIG['tokenizer_path']}")
    
    # Load dataset
    print(f"\n📊 Loading dataset...")
    sequences = load_tokenized_dataset(
        CONFIG["dataset_paths"],
        CONFIG["tokenizer_path"],
        CONFIG["seq_length"]
    )
    
    train_dataset, eval_dataset = create_hf_dataset(sequences)
    
    # Load model
    print(f"\n🏗️  Loading model...")
    from transformers import PretrainedConfig
    
    config = PretrainedConfig.from_dict(model_config)
    
    # Activer optimisations
    if CONFIG.get("gradient_checkpointing"):
        config.gradient_checkpointing = True
    if CONFIG.get("flash_attention"):
        config.attn_implementation = "flash_attention_2"
    
    model = LlamaForCausalLM(config)
    
    # Gradient checkpointing
    if CONFIG.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {params:,} parameters")
    print(f"   Optimizations:")
    print(f"     • Gradient checkpointing: {CONFIG.get('gradient_checkpointing', False)}")
    print(f"     • Flash Attention: {CONFIG.get('flash_attention', False)}")
    print(f"     • BF16: {CONFIG.get('bf16', False)}")
    print(f"     • TF32: {CONFIG.get('tf32', False)}")
    
    # Training args
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=1,
        max_steps=CONFIG["max_steps"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_steps=CONFIG["save_steps"],
        eval_steps=CONFIG["eval_steps"],
        logging_steps=CONFIG["logging_steps"],
        logging_dir=f"{CONFIG['output_dir']}/logs",
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=CONFIG["bf16"],
        tf32=True,
        dataloader_num_workers=CONFIG["num_workers"],
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=42,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None,  # Pas de tokenizer wrapper (déjà tokenisé)
        mlm=False,  # Causal LM (pas MLM)
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[],
    )
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"   ETA: ~{CONFIG['max_steps'] * 7 / 3600:.1f}h on RTX 5080\n")
    
    trainer.train()
    
    # Save
    print(f"\n💾 Saving model...")
    model.save_pretrained(f"{CONFIG['output_dir']}/final")
    
    # Copy config
    import shutil
    shutil.copy("french_llama_config.json", f"{CONFIG['output_dir']}/final/config.json")
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"   Output: {CONFIG['output_dir']}/final")
    print(f"   Formats:")
    print(f"     • pytorch_model.bin (HF format)")
    print(f"     • config.json (LLaMA-compatible)")
    print(f"   Next: Convert to GGUF with llama.cpp\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
