#!/usr/bin/env python3
"""
🔥 PHASE 2: Entraînement 80k steps supplémentaires
Reprend depuis checkpoint-85000 et continue jusqu'à 165000 steps
Dataset: Mixed FR (Wikipedia + FineWeb + Conversations)
"""

import os
import torch
from pathlib import Path
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

def load_phase2_dataset():
    """Charge le dataset Phase 2 (Mixed FR Data)"""
    print("📦 Chargement dataset Phase 2...")
    
    train_path = "data_clean/phase2_train_mistral_tokenized.pt"
    test_path = "data_clean/phase2_test_mistral_tokenized.pt"
    
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
    print("🔥 PHASE 2: ENTRAÎNEMENT 80k STEPS SUPPLÉMENTAIRES")
    print("="*70 + "\n")
    
    # Configuration
    CHECKPOINT_DIR = "trained_models/runs/french_llm_hf_extended/checkpoint-85000"
    OUTPUT_DIR = "trained_models/runs/french_llm_phase2"
    CURRENT_STEP = 85000
    TARGET_STEP = 165000  # 85k + 80k
    ADD_STEPS = TARGET_STEP - CURRENT_STEP
    
    print(f"📍 Checkpoint actuel: Step {CURRENT_STEP:,}")
    print(f"🎯 Nouvel objectif: Step {TARGET_STEP:,} ({ADD_STEPS:,} steps)")
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
    train_dataset, eval_dataset = load_phase2_dataset()
    if train_dataset is None:
        return
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Configuration d'entraînement
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,
        
        # Steps
        max_steps=TARGET_STEP,
        learning_rate=1e-4,
        
        # Batch
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        
        # Optimization
        optim="adamw_torch",
        warmup_steps=500,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        
        # Precision
        bf16=True,
        gradient_checkpointing=True,
        
        # Checkpoints
        save_strategy="steps",
        save_steps=500,
        save_total_limit=20,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        
        # Logging
        logging_strategy="steps",
        logging_steps=100,
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
    print("📊 CONFIGURATION PHASE 2")
    print("-" * 70)
    print(f"   Model:           GPT-2 (124M)")
    print(f"   Dataset:         Mixed FR (Wikipedia + FineWeb + Conversations)")
    print(f"   Train samples:   {len(train_dataset):,}")
    print(f"   Batch size:      4")
    print(f"   Steps:           {CURRENT_STEP:,} → {TARGET_STEP:,} ({ADD_STEPS:,} new)")
    print(f"   Learning rate:   1e-4 (cosine)")
    print(f"   Precision:       BF16")
    print(f"   Estimated time:  ~10-12 heures")
    print("-" * 70 + "\n")
    
    # Lancer entraînement
    print("🚀 DÉMARRAGE PHASE 2\n")
    trainer.train(resume_from_checkpoint=CHECKPOINT_DIR)
    
    # Sauvegarder final
    print("\n✅ PHASE 2 TERMINÉE!")
    final_dir = f"{OUTPUT_DIR}/final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"   Modèle sauvegardé: {final_dir}")

if __name__ == "__main__":
    main()
