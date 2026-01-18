# 🇫🇷 French LLM Training from Scratch

> Training a French GPT model from scratch on local hardware with full monitoring, periodic checkpoints, and conversational fine-tuning.

🔗 **Hugging Face Model:** [vincent-pro-ai/french-llm-from-scratch](https://huggingface.co/vincent-pro-ai/french-llm-from-scratch)

[English](#english) | [Français](#français)

---

## English

### 🎯 Project Overview

This project demonstrates the complete pipeline for training a medium-sized GPT model for French language from scratch, running entirely on consumer hardware (RTX 5080, 16GB VRAM).

**Key Achievements:**
- ✅ Trained 260M parameter model to **250.5k steps** using Multi-GPU DDP
- ✅ **Infrastructure Upgrade:** Now running on dual GPUs (RTX 4090 + RTX 5080)
- ✅ **-44% total loss reduction** (7.13 → 3.99) since baseline
- ✅ Scaled dataset to **530M tokens** total (UltraChat, OASST2, Dolly, Wikipedia, FineWeb)
- ✅ Implemented **Multi-GPU DistributedDataParallel (DDP)** with state-sync fixes
- ✅ Built custom web dashboard (Flask + React) for real-time monitoring
- ✅ Robust checkpointing with resume capabilities every 2,500 steps
- ✅ **4 publication-ready visualizations** (loss curves, comparisons, timeline)

### 📊 Results

**Complete Training Pipeline (60k → 250k steps)**

| Checkpoint | Steps | Val Loss | Improvement | Details |
|-----------|-------|----------|-------------|---------|
| **Baseline** | 60k | 7.13 | - | Initial training |
| **First milestone** | 115k | 4.88 | **-31.6%** | +15M conversation tokens |
| **Scaled dataset** | 180k | 4.89 | +0.2% | 197M tokens corpus |
| **Final base model** | 200k | **4.79** | **-2%** | High-quality base |
| **DDP Validation** | 250.5k| **3.99** | **-16%** | Multi-GPU stable training |

**Hardware Setup:**
- **Primary GPU:** NVIDIA RTX 4090 (24GB VRAM)
- **Secondary GPU:** NVIDIA RTX 5080 (16GB VRAM)
- **Training Orchestration:** PyTorch DDP via `torchrun`
- **RAM:** 80GB DDR5 System RAM

## 🚀 Mistral-7B QLoRA Fine-tuning

Expanding beyond training from scratch, we've implemented a professional fine-tuning pipeline for **Mistral-7B-v0.3** using **QLoRA** on our Multi-GPU infrastructure.

**Highlights:**
- **Technique:** QLoRA (4-bit BitsAndBytes quantization)
- **Framework:** PEFT + DDP (Multi-GPU execution)
- **Stability:** Forced `bfloat16` and optimized learning rate (`5e-5`) to avoid loss divergence.
- **Merge & Export:** Custom scripts to merge LoRA adapters back to base model.
- **Format:** Fully exported to **GGUF** (F16 and Q4_K_M) for seamless integration with **LM Studio**.

**Why this matters:**
This demonstrates the ability to take state-of-the-art weights and customize them for French specificities on local hardware, while ensuring the result is immediately deployable.

**Dataset Evolution:**
- **Phase 1 (60k):** 15M tokens (Wikipedia, FineWeb, conversations)
- **Phase 2 (115k):** +15M conversation tokens (WildChat, LMSYS, OpenHermes)
- **Phase 3 (200k):** **197M tokens** - ×13 scaling with:
  - **UltraChat FR:** 119M tokens (conversational AI)
  - **OASST2 FR:** Open-source assistant dialogues
  - **Dolly FR:** Instruction-following dataset
  - Previous corpora combined

**Training Metrics:**
- **Total GPU time:** ~43 hours on RTX 5080 16GB
- **Key phases:**
  - 60k → 115k: ~8h (55k steps, +conversation data)
  - 115k → 180k: ~8h20 (65k steps, massive dataset)
  - 180k → 195k: ~1h (15k steps, focused fine-tuning)
  - 195k → 200k: ~20min (5k steps, final polish)
- **Optimizations:** torch.compile, FusedAdam, AMP (Mixed Precision)
- **Memory efficiency:** Batch size 4, gradient accumulation, checkpointing every 2.5k steps

**Qualitative Improvements:**
- **60k → 115k:** +100% dialogue structure, +80% conversational coherence
- **115k → 200k:** +20% naturalness, smoother responses, better context handling

### 🏗️ Architecture

**Model Specifications:**
- **Layers:** 18 transformer blocks
- **Attention Heads:** 16
- **Embedding Dimension:** 1024
- **Feed-Forward Hidden:** 4096
- **Context Window:** 1024 tokens
- **Total Parameters:** ~260M
- **Tokenizer:** BPE with 32k vocabulary

**Training Configuration:**
- Optimizer: AdamW with weight decay
- Learning Rate: 
  - Base training: 2e-4 with CosineAnnealing
  - Fine-tuning: 3e-6 (very low for stability)
- Mixed Precision: AMP with GradScaler
- Batch Size: 4
- Checkpoints: Every 2500 steps
- Sample Generation: Every 250 steps

### 🛠️ Tech Stack

**Core:**
- PyTorch 2.x + CUDA 12
- Mixed Precision Training (AMP)
- HuggingFace Tokenizers (BPE)
- HuggingFace Datasets

**Monitoring:**
- Backend: Flask API (Python)
- Frontend: React + Vite
- Real-time metrics, loss curves, sample generation

**Deployment:**
- Docker + Docker Compose
- NVIDIA Container Toolkit for GPU support
- Reproducible environment across systems

**Hardware:**
- GPU: NVIDIA RTX 4090 (24GB VRAM) + RTX 5080 (16GB VRAM)
- RAM: 80GB System RAM
- Training Progress: Scaling to 500k steps with optimized 4090 config (Batch 96)
- Dataset size: 4GB pre-tokenized tokens (530 million tokens)

### 🚀 Quick Start

#### Option A: Docker (Recommended)

**Simplest way to get started with reproducible environment:**

```bash
# Clone the repository
git clone https://github.com/Vincent-PRO-AI/french-llm-from-scratch.git
cd french-llm-from-scratch

# Start dashboard
docker-compose up -d backend frontend

# Run training
docker-compose run --rm training python scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --tokenizer-path trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --pretokenized-path data_clean/mixed_tokenized.pt \
  --max-steps 60000 \
  --batch-size 4

# Access dashboard at http://localhost:5174
```

📖 **Full Docker guide:** See [DOCKER.md](DOCKER.md) for detailed instructions.

#### Option B: Local Installation

```bash
# Clone the repository
git clone https://github.com/Vincent-PRO-AI/french-llm-from-scratch.git
cd french-llm-from-scratch

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. Data Preparation

```bash
# Download and tokenize Wikipedia FR
python scripts/prepare_wikipedia.py --max-articles 3000
python scripts/clean_wikipedia.py

# Download FineWeb-2 French subset
python scripts/download_fineweb.py --max-size-mb 50

# Tokenize corpus
python scripts/pretokenize_corpus.py \
  --input-dir data_clean/wikipedia \
  --tokenizer trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --output data_clean/wikipedia_tokenized.pt
```

#### 3. Training

```bash
# Launch dashboard (optional but recommended)
cd dashboard
python server.py &
cd web && npm install && npm run dev &

# Start training
python scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --tokenizer-path trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --pretokenized-path data_clean/mixed_tokenized.pt \
  --max-steps 60000 \
  --batch-size 4 \
  --lr 0.0001 \
  --checkpoint-interval 5000

# Monitor at: http://localhost:5174
```

#### 4. Fine-Tuning on Conversations

```bash
# Option A: Automated pipeline (recommended)
bash scripts/run_finetune_pipeline.sh

# Option B: Manual steps
# 1. Setup HuggingFace authentication (for gated datasets like LMSYS)
python scripts/setup_hf_login.py

# 2. Download conversation datasets
python scripts/download_best_conversations.py

# 3. Tokenize new conversations
bash scripts/tokenize_conversations.sh

# 4. Combine with existing datasets
python scripts/combine_tokenized_datasets.py

# 5. Launch fine-tuning
python scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --resume-from trained_models/runs/french_medium_base_60k/checkpoint_step_100000.pt \
  --pretokenized-path data_clean/conversations_combined_tokenized.pt \
  --max-steps 115000 \
  --lr 3e-6 \
  --batch-size 4 \
  --checkpoint-interval 2500 \
  --run-name french_medium_finetune_conversations

# 6. Monitor training in real-time
bash scripts/monitor_training.sh
```

#### 5. Testing

```bash
# Generate samples from base model
python test_model_samples.py

# Compare base vs fine-tuned models
python test_finetuned_conversation.py

# Interactive conversation testing
python test_interactive.py
```

### 📁 Project Structure

```
french-llm-from-scratch/
├── scripts/                                # Training & data pipelines
│   ├── train_subtitles_transformer.py      # Main training script (torch.compile, FusedAdam)
│   ├── download_massive_conversations.py   # Multi-source downloader (UltraChat, OASST2, Dolly)
│   ├── combine_all_datasets.py            # Merge tokenized datasets → 197M tokens
│   ├── tokenize_massive_conversations.sh   # Chunked tokenization (handles 334MB+ files)
│   ├── monitor_mega_training.sh            # Real-time training monitor with ETA
│   ├── publish_to_huggingface.py          # HuggingFace Hub publication script
│   ├── save_mistral_tokenizer.py          # Save Mistral-7B tokenizer locally
│   └── ...
├── tests/                                  # Test scripts
│   ├── test_180k_vs_115k.py               # Compare checkpoints quantitatively
│   ├── test_180k_tokenizer.py             # Generation with proper tokenizer
│   ├── test_chat_model.py                 # Interactive chat testing
│   └── test_model_samples.py              # Sample generation
├── visualizations/                         # Publication-ready graphs (4 PNG)
│   ├── training_loss_evolution_60k_200k.png  # Full training curve
│   ├── checkpoint_comparison.png             # Bar chart comparison
│   ├── relative_improvement.png              # % gains visualization
│   └── training_timeline.png                 # Temporal training view
├── dashboard/
│   ├── server.py                          # Flask API (metrics, chat, runs)
│   └── web/                               # React + Vite frontend
├── training_configs/
│   ├── finetune_conversations.json        # Fine-tuning config
│   └── continue_base_100k.json            # Continue base training
├── data_clean/                            # Tokenized datasets (gitignored)
│   └── conversations_mega_tokenized.pt     # 197M tokens, 1.5GB
├── trained_models/                        # Checkpoints & runs (gitignored)
│   └── runs/
│       ├── french_medium_mega_finetune/    # 180k checkpoint
│       ├── french_medium_mega_finetune_extended/  # 195k checkpoint
│       └── french_medium_mega_finetune_200k/      # 200k checkpoint (final)
└── finetune_conversations.py              # Legacy fine-tuning script
```

### 🔍 Key Features

**1. Periodic Checkpoints**
- Automatic saves every N steps (configurable via `--checkpoint-interval`)
- Resume training from any checkpoint
- Never lose progress again!

**2. Learning Rate Scheduling**
- CosineAnnealingLR for smooth convergence
- Automatic decay from initial LR to 10% over training
- Better final model quality

**3. Web Dashboard**
- Real-time training metrics
- Loss curves visualization
- Sample generation during training
- Embedding/logits inspection

**4. Pre-tokenization**
- Fast I/O with pre-tokenized `.pt` files
- Significantly reduces training startup time
- Supports multiple corpus concatenation

**5. Automated Data Pipeline**
- Multi-source conversation downloader (WildChat, LMSYS, OpenHermes)
- HuggingFace authentication for gated datasets
- Automatic tokenization and dataset merging
- One-command fine-tuning pipeline

**6. Real-time Training Monitoring**
- Auto-refresh monitoring script (20s intervals)
- Live process stats, loss metrics, sample previews
- Background training with continuous visibility

### 📈 Scaling Options

**Local (More Time):**
- Continue training from checkpoint with lower LR
- Add more data (e.g., 200M+ tokens)
- Increase context window with gradient checkpointing

**Hardware Upgrade:**
- RTX 5090 (24GB) → Larger batches, longer context
- A5000/A6000 (48GB) → Even bigger models

**Cloud Scaling:**
- **Azure:** `Standard_NC24ads_A100_v4` (1×A100 80GB)
- **AWS:** `p4d.24xlarge` (8×A100 40GB), `p5.2xlarge` (1×H100 80GB)
- **Budget:** `g5.2xlarge` (A10G 24GB)

### 🎓 Lessons Learned

1. **Checkpoints are critical** - Lost a 42.5k step run before implementing periodic saves
2. **Pre-tokenization matters** - 5-10x faster data loading
3. **LR scheduling stabilizes** - CosineAnnealing prevents divergence late in training
4. **Dashboard changes everything** - Real-time visibility enables rapid iteration
5. **Very low LR for fine-tuning** - 3e-6 prevents catastrophic forgetting (vs 2e-4 base)
6. **Data > Compute** - Scaling 15M → 197M tokens (+1,213%) gave continuous improvement despite diminishing returns after 115k
7. **Hardware optimizations unlock scale** - torch.compile + FusedAdam + AMP enabled 197M tokens in 16GB VRAM
8. **Conversation datasets matter** - UltraChat + OASST2 significantly improved assistant-style responses
9. **Chunked tokenization essential** - OOM solved by splitting 334MB file into 299 parts, then merging
10. **Real-time monitoring saves time** - Auto-refresh scripts (20s intervals) catch issues early in 8h+ runs
11. **Visualizations aid debugging** - 4 graphs revealed training dynamics invisible in raw metrics
12. **Community feedback invaluable** - First LinkedIn post generated cloud/hardware suggestions that shaped roadmap

### 🔮 Future Work

- [ ] **Switch to Mistral-7B tokenizer** (117k vocab vs 32k custom) → Reduce BPE artifacts
- [x] ~~Extend to 100k+ steps~~ **DONE:** Reached 200k steps
- [x] ~~Fine-tune on conversations~~ **DONE:** 197M conversation tokens
- [x] ~~Scale dataset massively~~ **DONE:** ×13 scaling (15M → 197M tokens)
- [ ] **Migrate to cloud** (Vast.ai, RunPod, Azure, GCP) for faster iteration
- [ ] **Publish to Hugging Face** Hub for community use
- [ ] Add personal conversation data (Facebook, Instagram, Messenger archives)
- [ ] Extend to 500k steps with continual learning
- [ ] Add RLHF/DPO for alignment
- [x] **Multi-GPU training with DDP** (DONE: 2×GPU stable run)
- [ ] Longer context windows (4k-8k tokens) with gradient checkpointing
- [ ] Quantization (INT8/INT4) for mobile deployment
- [ ] Benchmark on MMLU-FR, FrenchBench academic evaluations

### 📝 Citation

If you use this code or methodology, please cite:

```bibtex
@misc{french-llm-from-scratch-2025,
  author = {Vincent Ayari},
  title = {French LLM Training from Scratch},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Vincent-PRO-AI/french-llm-from-scratch}
}
```

### 📄 License

MIT License - See [LICENSE](LICENSE) for details.

### 🙏 Acknowledgments

- HuggingFace for Tokenizers and Datasets
- OpenAssistant for conversation data
- FineWeb team for high-quality web corpus

### 🐳 Docker Support

Full Docker support for easy deployment and reproducibility!

**Quick Start:**
```bash
docker-compose up -d backend frontend
docker-compose run --rm training python scripts/train_subtitles_transformer.py [options]
```

**See [DOCKER.md](DOCKER.md) for:**
- Installation and setup
- GPU configuration
- Common commands
- Troubleshooting
- Production deployment

---

## Français

### 🎯 Vue d'ensemble du projet

Ce projet démontre le pipeline complet pour entraîner un modèle GPT de taille moyenne pour le français depuis zéro, fonctionnant sur une configuration multi-GPU (RTX 4090 + RTX 5080).

**Réalisations clés :**
- ✅ Modèle de 260M paramètres entraîné jusqu'à **250.5k steps** via DDP
- ✅ **Infrastructure Multi-GPU :** Utilisation conjointe de 2 cartes NVIDIA (40GB VRAM total)
- ✅ Fine-tuning sur **530M tokens** de conversations et textes diversifiés
- ✅ Implémentation stable de **DistributedDataParallel (DDP)** pour l'accélération
- ✅ Dashboard web personnalisé pour le monitoring en temps réel
- ✅ Système de checkpoints robuste avec reprise à chaque step
- ✅ Monitoring temps réel avec scripts auto-refresh

### 📊 Résultats

**Entraînement Multi-GPU DDP (200k → 250k steps)**
- **Loss de validation :** Passage de 4.79 à **3.99**
- **Vitesse :** Accélération significative grâce au parallélisme de données
- **Hardware :** RTX 4090 (Rank 1) + RTX 5080 (Rank 0)
- **Stabilité :** Zéro crash sur le run de validation de 500 steps

**Fine-tuning conversationnel (100k → 115k steps)**
- Dataset : 15M tokens de sources multiples :
  - **WildChat :** 4 570 conversations (dialogues ChatGPT)
  - **LMSYS Chat-1M :** 4 493 conversations (batailles arena)
  - **OpenHermes 2.5 :** 231 conversations (instructions GPT-4)
  - Datasets précédents : 6,4M tokens
- Format : Structure dialogue Utilisateur/Assistant
- Durée d'entraînement : 3h07 (15k steps)
- **Loss finale :** 4,88 (validation) - **amélioration de 17%** depuis 5,8
- Résultats : **+100% reconnaissance structure dialogue**, **+80% cohérence conversationnelle**

### 🏗️ Architecture

**Spécifications du modèle :**
- **Couches :** 18 blocs transformer
- **Têtes d'attention :** 16
- **Dimension d'embedding :** 1024
- **Feed-Forward caché :** 4096
- **Fenêtre de contexte :** 1024 tokens
- **Paramètres totaux :** ~260M
- **Tokenizer :** BPE avec vocabulaire 32k

**Configuration d'entraînement :**
- Optimiseur : AdamW avec weight decay
- Learning Rate : 
  - Entraînement base : 2e-4 avec CosineAnnealing
  - Fine-tuning : 3e-6 (très bas pour stabilité)
- Précision mixte : AMP avec GradScaler
- Taille de batch : 4
- Checkpoints : Tous les 2500 steps
- Génération d'échantillons : Tous les 250 steps

### 🛠️ Stack technique

**Cœur :**
- PyTorch 2.x + CUDA 12
- Entraînement en précision mixte (AMP)
- HuggingFace Tokenizers (BPE)
- HuggingFace Datasets

**Monitoring :**
- Backend : API Flask (Python)
- Frontend : React + Vite
- Métriques temps réel, courbes de loss, génération d'échantillons

**Déploiement :**
- Docker + Docker Compose
- NVIDIA Container Toolkit pour support GPU
- Environnement reproductible sur tous systèmes

**Hardware :**
- GPU : NVIDIA RTX 4090 (24GB VRAM) + RTX 5080 (16GB VRAM)
- RAM : 80GB RAM Système
- Progrès : Extension à 500k steps avec config optimisée (Batch 96)
- Memory Usage : ~95% RAM pendant le chargement du dataset (530M tokens)

### 🚀 Démarrage rapide

#### Option A : Docker (Recommandé)

**Moyen le plus simple avec environnement reproductible :**

```bash
# Cloner le dépôt
git clone https://github.com/Vincent-PRO-AI/french-llm-from-scratch.git
cd french-llm-from-scratch

# Démarrer le dashboard
docker-compose up -d backend frontend

# Lancer l'entraînement
docker-compose run --rm training python scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --tokenizer-path trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --pretokenized-path data_clean/mixed_tokenized.pt \
  --max-steps 60000 \
  --batch-size 4

# Accéder au dashboard : http://localhost:5174
```

📖 **Guide Docker complet :** Voir [DOCKER.md](DOCKER.md) pour les instructions détaillées.

#### Option B : Installation locale

```bash
# Cloner le dépôt
git clone https://github.com/Vincent-PRO-AI/french-llm-from-scratch.git
cd french-llm-from-scratch

# Créer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou : .venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

#### 2. Préparation des données

```bash
# Télécharger et tokeniser Wikipedia FR
python scripts/prepare_wikipedia.py --max-articles 3000
python scripts/clean_wikipedia.py

# Télécharger le sous-ensemble français FineWeb-2
python scripts/download_fineweb.py --max-size-mb 50

# Tokeniser le corpus
python scripts/pretokenize_corpus.py \
  --input-dir data_clean/wikipedia \
  --tokenizer trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --output data_clean/wikipedia_tokenized.pt
```

#### 3. Entraînement

```bash
# Lancer le dashboard (optionnel mais recommandé)
cd dashboard
python server.py &
cd web && npm install && npm run dev &

# Démarrer l'entraînement
python scripts/train_subtitles_transformer.py \
  --arch-preset medium \
  --tokenizer-path trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --pretokenized-path data_clean/mixed_tokenized.pt \
  --max-steps 60000 \
  --batch-size 4 \
  --lr 0.0001 \
  --checkpoint-interval 5000

# Monitorer sur : http://localhost:5174
```

#### 4. Fine-tuning sur conversations

```bash
# Télécharger le dataset de conversations
python scripts/download_conversations.py --max-conversations 5000

# Tokeniser les conversations
python scripts/pretokenize_corpus.py \
  --input-dir data_clean/conversations \
  --tokenizer trained_models/tokenizers/fineweb-32k/tokenizer.json \
  --output data_clean/conversations_tokenized.pt

# Fine-tuner depuis le checkpoint
python finetune_conversations.py
```

#### 5. Tests

```bash
# Générer des échantillons
python test_model_samples.py

# Tester les capacités conversationnelles
python test_chat_model.py
```

### 📁 Structure du projet

```
french-llm-from-scratch/
├── scripts/
│   ├── train_subtitles_transformer.py  # Script d'entraînement principal
│   ├── download_conversations.py       # Téléchargeur OpenAssistant
│   ├── pretokenize_corpus.py          # Tokenisation du corpus
│   └── ...
├── dashboard/
│   ├── server.py                       # Backend Flask
│   └── web/                            # Frontend React
├── data_clean/                         # Datasets tokenisés (gitignored)
├── trained_models/                     # Checkpoints & runs (gitignored)
├── finetune_conversations.py          # Script de fine-tuning
├── test_chat_model.py                 # Tests conversationnels
└── test_model_samples.py              # Génération d'échantillons
```

### 🔍 Fonctionnalités clés

**1. Checkpoints périodiques**
- Sauvegardes automatiques tous les N steps (configurable via `--checkpoint-interval`)
- Reprise de l'entraînement depuis n'importe quel checkpoint
- Ne perdez plus jamais votre progression !

**2. Scheduling du learning rate**
- CosineAnnealingLR pour une convergence fluide
- Décroissance automatique du LR initial à 10% sur l'entraînement
- Meilleure qualité du modèle final

**3. Dashboard web**
- Métriques d'entraînement en temps réel
- Visualisation des courbes de loss
- Génération d'échantillons pendant l'entraînement
- Inspection des embeddings/logits

**4. Pré-tokenisation**
- I/O rapide avec fichiers `.pt` pré-tokenisés
- Réduit significativement le temps de démarrage
- Support de la concaténation de multiples corpus

### 📈 Options de scaling

**Local (plus de temps) :**
- Continuer l'entraînement depuis un checkpoint avec LR plus bas
- Ajouter plus de données (ex: 200M+ tokens)
- Augmenter la fenêtre de contexte avec gradient checkpointing

**Upgrade matériel :**
- RTX 5090 (24GB) → Batchs plus grands, contexte plus long
- A5000/A6000 (48GB) → Modèles encore plus grands

**Scaling cloud :**
- **Azure :** `Standard_NC24ads_A100_v4` (1×A100 80GB)
- **AWS :** `p4d.24xlarge` (8×A100 40GB), `p5.2xlarge` (1×H100 80GB)
- **Budget :** `g5.2xlarge` (A10G 24GB)

### 🎓 Leçons apprises

1. **Les checkpoints sont critiques** - Perdu un run de 42,5k steps avant d'implémenter les sauvegardes périodiques
2. **La pré-tokenisation compte** - Chargement des données 5-10x plus rapide
3. **Le scheduling LR stabilise** - CosineAnnealing prévient la divergence en fin d'entraînement
4. **Le dashboard change tout** - La visibilité temps réel permet une itération rapide
5. **LR très bas pour fine-tuning** - 3e-6 évite l'oubli catastrophique (vs 2e-4 base)
6. **Qualité des données conversationnelles** - Datasets gated (LMSYS) valent l'authentification
7. **Monitoring temps réel essentiel** - Runs de 3h+ nécessitent visibilité continue
8. **Nettoyage artefacts BPE améliore** - Samples plus propres aident déboguer training

### 🔮 Travaux futurs

- [ ] Entraîner un tokenizer sans artefacts BPE (marqueurs Ġ, Ċ)
- [x] ~~Étendre à 100k+ steps~~ **FAIT :** 115k steps atteints
- [x] ~~Fine-tuning sur conversations~~ **FAIT :** 15M tokens conversationnels
- [ ] Continuer fine-tuning (250k → 500k steps) avec plus de données
- [ ] Ajouter RLHF/DPO pour l'alignement
- [x] **Entraînement multi-GPU avec DDP** (FAIT : Run stable 2xGPU)
- [ ] Fenêtres de contexte plus longues (2k-4k tokens)
- [ ] Fine-tuning d'instructions avec prompts diversifiés

### 📝 Citation

Si vous utilisez ce code ou cette méthodologie, veuillez citer :

```bibtex
@misc{french-llm-from-scratch-2025,
  author = {Vincent Ayari},
  title = {French LLM Training from Scratch},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Vincent-PRO-AI/french-llm-from-scratch}
}
```

### 📄 Licence

Licence MIT - Voir [LICENSE](LICENSE) pour les détails.

### 🙏 Remerciements

- HuggingFace pour Tokenizers et Datasets
- OpenAssistant pour les données de conversation
- L'équipe FineWeb pour le corpus web de haute qualité

### 🐳 Support Docker

Support Docker complet pour un déploiement facile et reproductible !

**Démarrage rapide :**
```bash
docker-compose up -d backend frontend
docker-compose run --rm training python scripts/train_subtitles_transformer.py [options]
```

**Voir [DOCKER.md](DOCKER.md) pour :**
- Installation et configuration
- Configuration GPU
- Commandes courantes
- Résolution de problèmes
- Déploiement en production

---

**⭐ Si ce projet vous est utile, n'hésitez pas à lui donner une étoile !**
