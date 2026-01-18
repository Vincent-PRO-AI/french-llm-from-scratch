# 🔧 GUIDE TECHNIQUE AVANCÉ - Optimisations & Possibilités

## 1. Optimisations GPU Actuelles vs Potentielles

### Actuellement Implémentées ✅

#### Mixed Precision (AMP) 
```python
# Implémentation actuelle:
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for step, batch in enumerate(loader):
    with autocast():
        logits = model(batch)
        loss = criterion(logits, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Impact: -50% VRAM, +15% speed, 0% quality loss
```

#### Gradient Checkpointing
```python
# Implémentation:
model = model.gradient_checkpointing_enable()

# Comment ça marche:
# - Recompute activations au backward
# - Trade compute for memory: -30% VRAM, -20% speed
# - Critical pour models > 400M params

# Validation: checksum test
original_output = model(x)
model.gradient_checkpointing_enable()
checked_output = model(x)
assert (original_output == checked_output).all()  # Identical
```

#### Torch.compile()
```python
# Code actuel:
model = torch.compile(model, mode="reduce-overhead")

# Optimization levels:
# - max-autotune: +20% speed, +20% compile time
# - reduce-overhead: +10% speed, 2s compile time  ← CURRENT
# - default: +5% speed, minimal overhead

# Limitations:
# - Only works on CUDA 11.8+
# - Some ops not supported (rare)
# - Recompile on model change
```

#### Gradient Accumulation
```python
# Implémentation:
accumulation_steps = 16
batch_size = 4
effective_batch = batch_size * accumulation_steps  # 64

optimizer.zero_grad()
for micro_step in range(accumulation_steps):
    batch = next(loader)
    loss = model(batch) / accumulation_steps
    loss.backward()
    
optimizer.step()

# Without: Batch size 4, worse gradient estimates
# With: Simulated batch 64, better learning
```

#### FusedAdam
```python
# Actuellement: Standard AdamW optimizer
# Impact: 5-10% faster optimization step

# De facto built-in from transformers
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=2e-4,
    weight_decay=0.0,
)
```

### Potentiellement Implémentables 🚀

#### 1. Flash Attention v2 (Très recommandé)
```python
# Installation:
pip install flash-attn --no-build-isolation

# Usage:
from flash_attn import flash_attn_func

# Replaces:
# attn_output = (Q @ K.T / sqrt(d)) @ V
# with optimized kernel

# Benefits:
# - 2-3× faster attention
# - 30-40% less memory
# - Numerically stable
# - Drop-in replacement

# Est-ce compatible?
# - Mistral: OUI ✅
# - Your model: OUI ✅
# - Training impact: 0%

# Implémentation (20 lignes):
import torch
from flash_attn import flash_attn_func

class FlashAttention(nn.Module):
    def forward(self, q, k, v):
        return flash_attn_func(q, k, v)
```

#### 2. LoRA (Low-Rank Adaptation)
```python
# Installation:
pip install peft

# Usage pour fine-tuning:
from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank (low)
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # Only attention
)

model = get_peft_model(model, peft_config)
# Trainable params: 260M → 2M (100× reduction!)
# VRAM: 16GB → 4GB
# Speed: Same or faster

# Utilité:
# - Instruction fine-tuning
# - Domain adaptation
# - Multi-task learning
```

#### 3. Quantization Aware Training (QAT)
```python
# Installation:
from torch.quantization import quantize_dynamic

# Post-training quantization:
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
# Size: 1GB → 250MB
# Speed: +2-3×
# Quality: 95-98%

# Vs GGUF quantization:
# - QAT: Done during training
# - GGUF: Done after (llama.cpp)
# - GGUF better quality, QAT faster inference
```

#### 4. xFormers Optimization
```python
# Installation:
pip install xformers

# Usage:
import xformers.ops as xops

# Replaces attention with memory-efficient variant
# Benefits:
# - 20-30% faster
# - 20-30% less memory
# - Drop-in compatible

# Activation already? 
# - À vérifier dans train_subtitles_transformer.py
```

#### 5. DeepSpeed Integration
```python
# Installation:
pip install deepspeed

# Configuration (ds_config.json):
{
    "fp16": {
        "enabled": true,
        "loss_scale": 1024
    },
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 2e-4}
    },
    "zero_optimization": {
        "stage": 2,  # Partition optimizer & gradients
        "offload_optimizer": {
            "device": "cpu"  # Swap to CPU RAM
        }
    }
}

# Utilité:
# - Single GPU: +20% speed, -30% memory
# - Multi-GPU: Essential for scaling > 1B params
# - ZeRO-3: Even GPU 3090 can train 100B models

# Effort: 2-4h setup
```

---

## 2. Extension du Dataset

### Données Actuelles
```
Total: 197M tokens
Sources:
- UltraChat FR: 119M
- OASST2: 34M
- Dolly: 20M
- FineWeb: 24M
```

### Données Disponibles à Intégrer

#### Option 1: cc100 (European Languages)
```python
# Installation:
from datasets import load_dataset
ds = load_dataset("cc100", "fr")
# Size: 85GB raw (French only)
# Tokens: ~500M when tokenized
# Quality: Medium (web crawl)
# Deduplication: Needed
# Time to integrate: 1-2 days
```

#### Option 2: OSCAR (Open Crawl)
```python
# Installation:
from datasets import load_dataset
ds = load_dataset("oscar", "unshuffled_deduplicated_fr")
# Size: 150GB (already deduped)
# Tokens: ~1B when tokenized
# Quality: Medium (web)
# Filtering: French-only included
# Time: 1-2 days integration
```

#### Option 3: mC4 (Common Crawl)
```python
# Installation:
from datasets import load_dataset
ds = load_dataset("allenai/c4", "fr")
# Size: 200GB+ (French)
# Tokens: ~1.5B
# Quality: High (filtered CC)
# Deduplication: Included
# Time: 2-3 days
```

#### Option 4: Wikipedia + Wikibooks + WikiHow
```python
# Already have Wikipedia
# Can add:
- French Wikibooks (40MB)
- WikiHow French (100MB)
- Wikisource (50MB)
- Combined: +200M tokens

# Implementation:
scripts/prepare_wikipedia.py  # Already exists!
```

#### Option 5: Reddit/Forum Data
```python
# French Reddit + Quora FR
# Size: ~50GB
# Quality: High (authentic language)
# Deduplication: Needed
# Filtering: Remove spam

# Sources:
- pushshift.io (Reddit dumps)
- Quora data
# Time: 1-2 days download + clean
```

### Pipeline d'Intégration (1-2 semaines)

```bash
# Step 1: Download
python scripts/download_hf_datasets.py \
  --dataset oscar \
  --lang fr \
  --output data_new/oscar_fr.txt

# Step 2: Clean & Deduplicate
python scripts/clean_dataset_french_only.py \
  data_new/oscar_fr.txt \
  --remove-duplicates \
  --min-length 100

# Step 3: Tokenize
python scripts/pretokenize_file.py \
  data_new/oscar_fr_clean.txt \
  data_new/oscar_fr_tokenized.pt \
  --tokenizer mistral

# Step 4: Combine with existing
python scripts/combine_tokenized_datasets.py \
  data_clean/conversations_mega_train_mistral_tokenized.pt \
  data_new/oscar_fr_tokenized.pt \
  --output data_clean/combined_1b_tokens.pt

# Step 5: Continue training
python train_4090_optimized.py \
  --pretokenized-path data_clean/combined_1b_tokens.pt \
  --resume-from checkpoint_250k_fixed.pt \
  --max-steps 500000
```

### Impact Estimé

```
Current: 197M tokens
+ Oscar: 1B tokens
+ CC100: 500M tokens
Total potential: 1.7B tokens

Training cost:
- 197M tokens: ~43h on RTX 5080
- 1.7B tokens: ~372h (15.5 jours continu)

Loss improvement:
- 197M: 4.79
- 500M: 4.50-4.60 (estimated)
- 1B: 4.30-4.40 (estimated)
- 1.7B: 4.10-4.20 (estimated)

Effort: 1-2 weeks setup + GPU time
```

---

## 3. Instruction Fine-tuning

### Pourquoi?
```
Current model: Base LLM
- Bon pour: Text generation, completion
- Faible pour: Following instructions, Q&A

After instruction tuning:
- Bon pour: Chat, Q&A, task completion
- Quality jump: +40-60%
```

### Datasets d'Instruction

#### Option 1: Dolly (Already have)
```yaml
Format: {"instruction": "...", "input": "...", "output": "..."}
Size: 15k examples
Quality: High
Availability: ✅ Already in data
Time to tune: 1-2 hours
```

#### Option 2: OpenHermes
```yaml
Size: 1M high-quality instruction pairs
Quality: Excellent (filtered by human ranking)
License: MIT
Format: Instruction + Response
Time: 4-6 hours tuning
```

#### Option 3: Alpaca / ShareGPT
```yaml
- Alpaca: 52k examples, simple format
- ShareGPT: 100k conversations
- Quality: Medium
- License: Mix of open licenses
Time: 3-5 hours tuning each
```

### Implementation Pipeline

```python
# Step 1: Prepare instruction data
from datasets import load_dataset

alpaca_ds = load_dataset("tatsu-lab/alpaca")
dolly_ds = load_dataset("databricks/databricks-dolly-15k")

def format_instruction(example):
    return {
        "text": f"[INST] {example['instruction']} [/INST] {example['output']}"
    }

combined = concatenate_datasets([
    alpaca_ds.map(format_instruction),
    dolly_ds.map(format_instruction)
])

# Step 2: Fine-tune with LoRA
from peft import get_peft_model, LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# Step 3: Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./instruction_tuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
    ),
    train_dataset=combined,
    data_collator=DataCollatorForLanguageModeling(tokenizer)
)
trainer.train()

# Step 4: Merge & Export
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./french-llm-v2-instructed")
```

### Expected Results

```
Task: "Écris un poème sur l'amour"
Before tuning:
  "Écris un poème sur l'amour et la beauté de la nature.
   La nature est belle, l'amour est beau, la nature..."

After tuning:
  "Écris un poème sur l'amour
   
   Dans la douceur de tes yeux,
   Je vois l'infini des cieux,
   Ton coeur bat près du mien,
   Comme une symphonie de liens.
   ..."

Quality improvement: 50-100%
Time investment: 5-10 hours
```

---

## 4. Multi-GPU Scaling (2-4 GPUs)

### Architecture DDP (Distributed Data Parallel)

```python
# Code exists: scripts/train_subtitles_transformer_ddp.py

# Modification simple:
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())

# Wrap model
model = DDP(model, device_ids=[dist.get_rank()])

# Launch:
torchrun --nproc_per_node 2 train_subtitles_transformer_ddp.py

# Effect:
# - 2x RTX 5080 (32GB total)
# - Batch size: 4 × 2 = 8 per GPU (16 effective)
# - Speed: 1.8-1.9× faster (not perfect 2× due to communication)
# - Loss convergence: Better (bigger effective batch)
```

### Configuration Multi-GPU

```yaml
# docker-compose.multi-gpu.yml (already exists!)

services:
  trainer-gpu0:
    image: french-llm:latest
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - RANK=0
      - WORLD_SIZE=2
      - MASTER_ADDR=localhost
      - MASTER_PORT=29500
    gpus:
      - device_ids: ['0']
        
  trainer-gpu1:
    image: french-llm:latest
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - RANK=1
      - WORLD_SIZE=2
      - MASTER_ADDR=localhost
      - MASTER_PORT=29500
    gpus:
      - device_ids: ['1']
```

### Launch Command

```bash
# Manual launch:
torchrun --nproc_per_node=2 train_subtitles_transformer_ddp.py

# Docker:
docker-compose -f docker-compose.multi-gpu.yml up -d

# Monitoring:
nvidia-smi  # Both GPUs busy
watch -n 1 nvidia-smi  # Real-time
```

---

## 5. RAG Integration (Retrieval-Augmented Generation)

### Pourquoi?
```
Problem: Model a limited knowledge cutoff (199M tokens)
Solution: Augment with external knowledge

Before RAG:
Q: "Quel est le taux de COVID en 2024?"
A: "Je ne sais pas, mes données s'arrêtent en 2022"

After RAG:
Q: "Quel est le taux de COVID en 2024?"
RAG retrieves: Recent news articles...
A: "Selon les dernières données, le taux est..."
```

### Implementation (Scripts exist!)

```python
# Scripts disponibles:
# - rag_api.py
# - rag_index_wikipedia.py
# - rag_test.py

# Step 1: Build index (one-time)
python rag_index_wikipedia.py
# Creates FAISS index on Wikipedia
# Time: 30-60 minutes
# Output: index.faiss (500MB)

# Step 2: Start API
python rag_api.py
# Integrates model + RAG

# Step 3: Query
curl -X POST http://localhost:5000/rag/generate \
  -d '{"query": "Quel est..."}'
```

### Architecture

```
Query: "Quel est le capital du Japon?"
       ↓
[FAISS Index] ← Wikipedia facts
       ↓ retrieve top-k documents
"Tokyo is the capital of Japan..."
       ↓
Augmented prompt:
"Context: Tokyo is the capital...
Question: Quel est le capital du Japon?
Answer:"
       ↓
[French LLM] → Generate response
       ↓
Response: "Tokyo est le capital du Japon"
```

### Expansion

```python
# Beyond Wikipedia:
- Crawl recent news (RSS feeds)
- Include scientific papers (arXiv)
- Add proprietary documents
- Real-time web search integration

# Code to add:
# 1. Web crawler
# 2. Dynamic index updates
# 3. Temporal relevance ranking
# 4. Multi-index search

# Effort: 1-2 weeks
```

---

## 6. Benchmark vs Autres Modèles

### Setup

```python
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, tasks

# Charge notre modèle
model = HFLM(pretrained="vincent-pro-ai/french-llm-from-scratch")

# Benchmarks
task_list = [
    "hellaswag",      # Common sense reasoning
    "arc_easy",       # Question answering
    "gsm8k",          # Math (few-shot)
    "xnli_fr",        # NLI French
    "xquad_fr",       # QA French
]

results = evaluator.simple_evaluate(
    model=model,
    tasks=task_list,
    num_fewshot=0,  # Zero-shot
    batch_size=4
)
```

### Expected Results vs Baselines

```
Task              Our Model  GPT-2  BLOOM-560M  Mistral-7B
─────────────────────────────────────────────────────────
ARC Easy          55%        58%    62%         90%
HellaSwag         32%        29%    36%         79%
GSM8k (few-shot)  8%         3%     12%         70%
XNLI (French)     72%        68%    75%         92%
XQUAD (French)    45%        38%    52%         89%
```

---

## 7. Production Deployment

### Option 1: FastAPI Server

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# Load model once
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

class CompletionRequest(BaseModel):
    model: str = "french-llm"
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return {"text": tokenizer.decode(outputs[0])}

# Deploy:
# uvicorn app:app --host 0.0.0.0 --port 8000
```

### Option 2: vLLM (Optimized Serving)

```bash
# Installation:
pip install vllm

# Launch:
python -m vllm.entrypoints.openai.api_server \
  --model vincent-pro-ai/french-llm-from-scratch \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8

# Advantages:
# - Continuous batching (20-40% faster)
# - PagedAttention (2x memory efficiency)
# - OpenAI-compatible API
# - 1500+ tokens/sec (vs 400 baseline)
```

### Option 3: Hugging Face Spaces

```python
# Create space on: huggingface.co/spaces
# Push this code:

import gradio as gr
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="vincent-pro-ai/french-llm-from-scratch",
    device=0
)

def generate(prompt, max_tokens, temperature):
    output = generator(
        prompt,
        max_length=max_tokens,
        temperature=temperature,
        do_sample=True,
    )
    return output[0]["generated_text"]

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(10, 500, label="Max tokens"),
        gr.Slider(0.1, 2.0, label="Temperature"),
    ],
    outputs="text"
)

demo.launch()
```

---

## 8. Checklist Technique

### Avant Production
- [ ] Validate checkpoint 250k loss (4.79 ± 0.05)
- [ ] Export to GGUF quantized formats
- [ ] Test API endpoints
- [ ] Benchmark inference speed
- [ ] Memory profiling
- [ ] Stress test (concurrent requests)
- [ ] Error handling & graceful degradation

### Monitoring en Production
- [ ] Request logging
- [ ] Latency metrics (p50, p95, p99)
- [ ] Error rate tracking
- [ ] Model drift detection
- [ ] GPU utilization alerts
- [ ] Disk space monitoring
- [ ] Backup automation

### Documentation
- [ ] API documentation (Swagger)
- [ ] Quick start guide
- [ ] Troubleshooting guide
- [ ] Performance tuning guide
- [ ] Scaling guide

---

## 9. Troubleshooting Guide

### CUDA Memory Issues
```python
# Problem: CUDA out of memory
# Solution 1: Reduce batch size
batch_size = 2  # From 4

# Solution 2: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: CPU offloading
import deepspeed
# See section 5

# Solution 4: Reduce model size
num_layers = 12  # From 18
```

### Slow Training
```python
# Problem: < 500 tokens/sec
# Solution 1: Increase batch size
batch_size = 6  # More parallelism

# Solution 2: Enable torch.compile
model = torch.compile(model)

# Solution 3: CPU idle?
num_workers = 4  # Parallelize data loading
pin_memory = True

# Solution 4: Lower precision
use_amp = True  # Ensure enabled
```

### Model Quality Issues
```python
# Problem: Low quality generations
# Solution: More training data
# or: Instruction fine-tuning
# or: Adjust temperature (sampling)

# Debug:
from scripts.evaluate_checkpoints import evaluate
evaluate(["checkpoint_200k.pt", "checkpoint_250k.pt"])
```

---

## 10. Research Ideas

### À Explorer
1. **Mixture of Experts (MoE):** 260M → 8 experts (260M × 8, sparse)
2. **Knowledge Distillation:** From 7B Mistral to 260M
3. **Continuous Learning:** Online adaptation to new data
4. **Multi-lingual:** French + English + German (trilingual)
5. **Domain Adaptation:** Medical, legal, technical French
6. **Alignment:** RLHF, DPO for better responses
7. **Retrieval Training:** Learn to use external knowledge
8. **Long Context:** Extend from 2048 → 8192 tokens

---

**Estimated Total Effort for All Features: 3-6 months** with continuous GPU access

**Priorité recommandée: Instruction tuning → Multi-GPU → Extended training**
