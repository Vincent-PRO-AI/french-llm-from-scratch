#!/usr/bin/env python3
"""
Évaluation complète du modèle French LLM.
Comprend: perplexité, qualité texte, comparaisons avec baselines.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configuration
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.model.transformer import TransformerLM
from src.data.tokenizer import FrenchTokenizer


class TextEvalDataset(Dataset):
    """Dataset pour l'évaluation."""
    
    def __init__(self, text_path, tokenizer, max_length=1024, num_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Charger le texte
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokeniser
        tokens = tokenizer.encode(text)
        
        # Créer des samples
        self.samples = []
        for i in range(0, len(tokens) - max_length - 1, max_length // 2):
            sample = tokens[i:i + max_length + 1]
            if len(sample) == max_length + 1:
                self.samples.append(torch.tensor(sample, dtype=torch.long))
        
        if num_samples:
            self.samples = self.samples[:num_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_ids': sample[:-1],
            'target_ids': sample[1:]
        }


class ModelEvaluator:
    """Classe pour évaluer le modèle."""
    
    def __init__(self, model_path, tokenizer_path, device='cuda'):
        self.device = device
        self.tokenizer = FrenchTokenizer(tokenizer_path)
        
        # Charger le modèle
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Créer une instance du modèle
        self.model = TransformerLM(
            vocab_size=self.tokenizer.vocab_size,
            d_model=1024,
            nhead=16,
            num_layers=18,
            dim_feedforward=4096,
            max_seq_length=1024,
            dropout=0.1
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
    
    def compute_perplexity(self, dataloader):
        """Calculer la perplexité."""
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing perplexity"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.tokenizer.vocab_size),
                    target_ids.reshape(-1),
                    reduction='none'
                )
                
                total_loss += loss.sum().item()
                total_tokens += target_ids.numel()
        
        perplexity = np.exp(total_loss / total_tokens)
        return perplexity
    
    def sample_text(self, prompt, max_length=100, temperature=0.7, top_k=50):
        """Générer du texte."""
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(tokens)
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                top_k_vals, top_k_indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                probs = F.softmax(top_k_vals, dim=-1)
                next_token = top_k_indices[0, torch.multinomial(probs[0], 1)]
                
                tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        generated_text = self.tokenizer.decode(tokens[0].tolist())
        return generated_text
    
    def evaluate(self, eval_text_path, output_path, num_samples=1000):
        """Évaluation complète."""
        print(f"📊 Évaluation du modèle")
        print(f"   Checkpoint: {eval_text_path}")
        print(f"   Samples: {num_samples}")
        
        # Créer le dataset
        print("\n1️⃣ Chargement du dataset...")
        dataset = TextEvalDataset(
            eval_text_path,
            self.tokenizer,
            max_length=1024,
            num_samples=num_samples
        )
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        print(f"   ✅ {len(dataset)} samples chargés")
        
        # Perplexité
        print("\n2️⃣ Calcul de la perplexité...")
        perplexity = self.compute_perplexity(dataloader)
        print(f"   ✅ Perplexité: {perplexity:.2f}")
        
        # Génération de texte
        print("\n3️⃣ Génération de texte...")
        prompts = [
            "Les intelligences artificielles",
            "La France est",
            "Pour apprendre le français"
        ]
        generations = {}
        for prompt in prompts:
            text = self.sample_text(prompt, max_length=100)
            generations[prompt] = text
            print(f"   Prompt: '{prompt}'")
            print(f"   → {text[:100]}...\n")
        
        # Résultats
        results = {
            'timestamp': datetime.now().isoformat(),
            'perplexity': float(perplexity),
            'generations': generations
        }
        
        # Sauvegarder
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Résultats sauvegardés dans {output_path}")
        return results


def main():
    # Chemins
    latest_checkpoint = max(
        Path('trained_models/runs/french_v3_finetune_grand_60k').glob('checkpoint_step_*.pt'),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    
    eval_text_path = 'data_clean/conversations_mega_test.txt'
    tokenizer_path = 'trained_models/mistral_tokenizer.model'
    output_path = 'evaluation_results.json'
    
    # Vérifier les fichiers
    if not latest_checkpoint.exists():
        print(f"❌ Checkpoint introuvable: {latest_checkpoint}")
        return
    
    if not Path(eval_text_path).exists():
        print(f"❌ Fichier d'évaluation introuvable: {eval_text_path}")
        return
    
    if not Path(tokenizer_path).exists():
        print(f"❌ Tokenizer introuvable: {tokenizer_path}")
        return
    
    # Évaluer
    evaluator = ModelEvaluator(str(latest_checkpoint), tokenizer_path)
    evaluator.evaluate(eval_text_path, output_path)


if __name__ == '__main__':
    main()
