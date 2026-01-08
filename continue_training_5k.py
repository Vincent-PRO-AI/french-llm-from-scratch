#!/usr/bin/env python3
"""
Continuer l'entraînement du modèle GPT-2 français de 5000 steps supplémentaires
Résume depuis checkpoint-80000 jusqu'à 85000 steps
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
from datasets import load_from_disk, Dataset
import numpy as np

def load_dataset_simple():
    """Charge le dataset de conversations françaises pré-tokenisé"""
    print("📦 Chargement du dataset...")
    
    # Charger le dataset pré-tokenisé
    dataset_path = "data_clean/conversations_mega_train_mistral_tokenized.pt"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return None
    
    # Charger les tokens
    data = torch.load(dataset_path, map_location='cpu')
    print(f"   ✅ {len(data) / 1e6:.1f}M tokens chargés")
    
    # Créer des séquences de 512 tokens
    sequences = []
    for i in range(0, len(data) - 512, 512):
        sequences.append({'input_ids': data[i:i+512].tolist()})
    
    print(f"   ✅ {len(sequences):,} séquences créées")
    
    # Créer un dataset HuggingFace
    dataset = Dataset.from_list(sequences)
    
    # Split train/test
    split = dataset.train_test_split(test_size=0.15, seed=42)
    
    print(f"   ✅ Train: {len(split['train']):,} | Test: {len(split['test']):,}")
    
    return split['train'], split['test']

def main():
    print("\n" + "="*60)
    print("🔥 CONTINUER L'ENTRAÎNEMENT - 5000 STEPS")
    print("="*60 + "\n")
    
    # Configuration
    CHECKPOINT_DIR = "trained_models/runs/french_llm_hf/checkpoint-80000"
    OUTPUT_DIR = "trained_models/runs/french_llm_hf_extended"
    MAX_STEPS = 85000  # 80000 + 5000
    RESUME_STEP = 80000
    
    print(f"📍 Reprise depuis: {CHECKPOINT_DIR}")
    print(f"🎯 Nouvel objectif: {MAX_STEPS} steps (+ 5000)\n")
    
    # Charger le modèle depuis le checkpoint
    print("📦 Chargement du modèle depuis checkpoint...")
    try:
        model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_DIR)
        print(f"   ✅ Modèle chargé ({model.num_parameters() / 1e6:.1f}M params)")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return
    
    # Charger le tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(CHECKPOINT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✅ Tokenizer chargé\n")
    
    # Charger le dataset
    train_dataset, eval_dataset = load_dataset_simple()
    if train_dataset is None:
        return
    print()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Configuration d'entraînement
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,  # Ne pas écraser les checkpoints existants
        
        # Steps et epochs
        max_steps=MAX_STEPS,
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
        
        # Precision et gradient
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
        
        # Perf
        dataloader_num_workers=4,
        remove_unused_columns=True,
        seed=42,
    )
    
    # Créer le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Afficher les infos
    print("📊 CONFIGURATION D'ENTRAÎNEMENT")
    print("-" * 60)
    print(f"   Model:           {model.config.model_type.upper()} ({model.num_parameters()/1e6:.1f}M)")
    print(f"   Dataset:         {len(train_dataset):,} train | {len(eval_dataset):,} eval")
    print(f"   Batch Size:      4 (per_device) x 1 (accumulation)")
    print(f"   Learning Rate:   1e-4 (cosine)")
    print(f"   Max Steps:       {MAX_STEPS:,} (from {RESUME_STEP:,})")
    print(f"   Precision:       BF16 + Gradient Checkpointing")
    print(f"   Checkpoint Save: Every 500 steps")
    print(f"   Output Dir:      {OUTPUT_DIR}")
    print("-" * 60 + "\n")
    
    # Lancer l'entraînement
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT\n")
    trainer.train(resume_from_checkpoint=CHECKPOINT_DIR)
    
    # Sauvegarder le modèle final
    print("\n✅ ENTRAÎNEMENT TERMINÉ!")
    print(f"   Sauvegarde du modèle final...")
    
    final_dir = f"{OUTPUT_DIR}/final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"   ✅ Modèle sauvegardé: {final_dir}\n")
    
    # Statistiques finales
    print("📊 RÉSUMÉ FINAL")
    print("-" * 60)
    print(f"   Checkpoints créés: {len(list(Path(OUTPUT_DIR).glob('checkpoint-*')))}")
    print(f"   Dernier checkpoint: {OUTPUT_DIR}/checkpoint-{MAX_STEPS}")
    print(f"   Modèle final: {final_dir}")
    print("-" * 60)

if __name__ == "__main__":
    main()
