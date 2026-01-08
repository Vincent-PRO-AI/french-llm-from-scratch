#!/usr/bin/env python3
"""
🚀 FRENCH LLM TRAINING - HUGGINGFACE SIMPLE
==========================================
Minimal working training script using HuggingFace Transformers
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from pathlib import Path

CONFIG = {
    "model_name": "meta-llama/Llama-2-7b",  # Petit modèle public stable
    "dataset_path": "data_clean/conversations_mega_train_mistral_tokenized.pt",
    "output_dir": "trained_models/runs/french_llm_hf",
    "max_steps": 80000,  # FULL training (80k steps)
    "batch_size": 4,  # Ultra-conservateur pour stabilité
    "learning_rate": 1e-4,
}

def load_dataset_simple():
    """Load pre-tokenized dataset"""
    print("📊 Loading tokens...")
    tokens = torch.load(CONFIG["dataset_path"], map_location='cpu')
    print(f"   Loaded: {len(tokens):,} tokens")
    
    # Create sequences
    seq_length = 512
    sequences = []
    for i in range(0, len(tokens) - seq_length, seq_length // 2):
        seq = tokens[i:i + seq_length].tolist()
        sequences.append({"input_ids": seq, "labels": seq})
    
    print(f"   Sequences: {len(sequences):,}")
    return Dataset.from_dict({
        "input_ids": [s["input_ids"] for s in sequences],
        "labels": [s["labels"] for s in sequences],
    })

def main():
    print("\n" + "="*70)
    print("🚀 HUGGINGFACE SIMPLE TRAINING")
    print("="*70)
    
    # Load model (small, stable)
    print("\n📦 Loading model (Llama-2-7b)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        print("   (Llama-2 nécessite auth HuggingFace)")
        print("   Utilisation alternative: GPT-2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Load dataset
    print("\n📚 Loading dataset...")
    dataset = load_dataset_simple()
    
    # Training args
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        max_steps=CONFIG["max_steps"],
        per_device_train_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        save_steps=500,
        logging_steps=100,
        bf16=True,  # Mixed precision
        gradient_checkpointing=True,  # Reduce memory
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    print("\n✅ Training completed!")

if __name__ == "__main__":
    main()
