#!/usr/bin/env python3
"""
🎓 FINE-TUNING: 60k steps sur conversations françaises pures
Dataset: Conversations FR uniquement
Objectif: Améliorer qualité réponses conversationnelles
"""

import os
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

def load_finetune_dataset():
    """Charge le dataset Fine-tuning (Conversations FR)"""
    print("📦 Chargement dataset Fine-tuning...")
    
    train_path = "data_clean/finetune_train_mistral_tokenized.pt"
    test_path = "data_clean/finetune_test_mistral_tokenized.pt"
    
    if not os.path.exists(train_path):
        print(f"❌ Dataset non trouvé: {train_path}")
        return None
    
    train_tokens = torch.load(train_path, map_location='cpu')
    test_tokens = torch.load(test_path, map_location='cpu')
    
    print(f"   ✅ Train: {len(train_tokens)/1e6:.1f}M tokens")
    print(f"   ✅ Test:  {len(test_tokens)/1e6:.1f}M tokens")
    
    # Créer séquences
    def create_sequences(tokens, seq_length=512):
        sequences = []
        for i in range(0, len(tokens) - seq_length + 1, seq_length):
            seq = tokens[i:i + seq_length]
            if len(seq) == seq_length:
                sequences.append({'input_ids': seq.tolist()})
        return sequences
    
    train_seqs = create_sequences(train_tokens)
    test_seqs = create_sequences(test_tokens)
    
    train_dataset = Dataset.from_list(train_seqs)
    test_dataset = Dataset.from_list(test_seqs)
    
    print(f"   ✅ Train seqs: {len(train_dataset):,}")
    print(f"   ✅ Test seqs:  {len(test_dataset):,}\n")
    
    return train_dataset, test_dataset

def main():
    print("\n" + "="*70)
    print("🎓 FINE-TUNING: 60k STEPS CONVERSATIONS FRANÇAISES")
    print("="*70 + "\n")
    
    # Configuration
    CHECKPOINT_DIR = "trained_models/runs/french_llm_phase2/final_model"
    OUTPUT_DIR = "trained_models/runs/french_llm_finetune_conversations"
    MAX_STEPS = 60000
    
    print(f"📍 Checkpoint source: Phase 2 final ({CHECKPOINT_DIR})")
    print(f"🎯 Objectif: {MAX_STEPS:,} steps")
    print(f"📂 Output: {OUTPUT_DIR}\n")
    
    # Charger modèle
    print("🤖 Chargement modèle...")
    try:
        model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_DIR)
        print(f"   ✅ Modèle chargé ({model.num_parameters() / 1e6:.1f}M params)")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return
    
    # Charger tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(CHECKPOINT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✅ Tokenizer chargé\n")
    
    # Charger dataset
    train_dataset, eval_dataset = load_finetune_dataset()
    if train_dataset is None:
        return
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Configuration d'entraînement (Learning rate plus bas pour fine-tuning)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,
        
        # Steps
        max_steps=MAX_STEPS,
        learning_rate=5e-5,  # Plus bas pour fine-tuning
        
        # Batch
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        
        # Optimization
        optim="adamw_torch",
        warmup_steps=300,  # Moins de warmup
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        
        # Precision
        bf16=True,
        gradient_checkpointing=True,
        
        # Checkpoints
        save_strategy="steps",
        save_steps=500,
        save_total_limit=15,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        
        # Logging
        logging_strategy="steps",
        logging_steps=50,
        log_level="info",
        
        # Performance
        dataloader_num_workers=4,
        remove_unused_columns=True,
        seed=42,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Afficher infos
    print("📊 CONFIGURATION FINE-TUNING")
    print("-" * 70)
    print(f"   Model:           GPT-2 (124M) - Pre-trained Phase 2")
    print(f"   Dataset:         Conversations FR pures")
    print(f"   Train samples:   {len(train_dataset):,}")
    print(f"   Batch size:      4")
    print(f"   Steps:           {MAX_STEPS:,}")
    print(f"   Learning rate:   5e-5 (cosine, fine-tuning)")
    print(f"   Precision:       BF16")
    print(f"   Estimated time:  ~6-7 heures")
    print("-" * 70 + "\n")
    
    # Lancer entraînement
    print("🚀 DÉMARRAGE FINE-TUNING\n")
    trainer.train()
    
    # Sauvegarder final
    print("\n✅ FINE-TUNING TERMINÉ!")
    final_dir = f"{OUTPUT_DIR}/final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"   Modèle sauvegardé: {final_dir}")

if __name__ == "__main__":
    main()
