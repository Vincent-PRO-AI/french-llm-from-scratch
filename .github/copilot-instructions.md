# French LLM Training Project - AI Coding Agent Instructions

## Project Overview
This codebase trains a French GPT model from scratch using PyTorch. Key components:
- **Data Pipeline**: Download → Clean → Tokenize → Combine datasets (Wikipedia, FineWeb, conversations)
- **Training**: Multi-phase transformer training with checkpoints, AMP, torch.compile
- **Monitoring**: Flask API + React dashboard for real-time metrics
- **Deployment**: FastAPI service with Docker

## Architecture Patterns
- **Model**: 260M parameter transformer (18 layers, 16 heads, 1024 embed, 4096 FF)
- **Training Loop**: Custom SubtitleTrainer class with gradient accumulation, mixed precision
- **Data Flow**: Pre-tokenized datasets loaded via memmap for efficiency
- **Checkpointing**: Every 2500 steps, resume from `checkpoint_step_*.pt`
- **Logging**: JSONL metrics, text samples, GPU monitoring

## Critical Workflows
- **Environment**: Use `docker-compose up -d` for reproducible setup (backend/frontend services)
- **Training Launch**: Run `python launch_mistral_training.py` for automated pipeline
- **Data Prep**: Scripts like `combine_text_corpus.py` handle train/test splits (85/15)
- **Code Quality**: `make check` compiles, `make format` uses autopep8, `make lint` uses flake8
- **Testing**: Run individual test files (e.g., `python tests/test_model_samples.py`) to validate checkpoints

## Project-Specific Conventions
- **Naming**: Runs prefixed with `french_medium_*`, checkpoints as `checkpoint_step_*.pt`
- **Paths**: Data in `data_clean/`, models in `trained_models/runs/`, logs as `metrics.jsonl`
- **Tokenization**: Mistral tokenizer (117k vocab) for French, BPE artifacts cleaned (Ġ → space)
- **Sampling**: Generate text every 500 steps during training, logged to `samples.txt`
- **French Code**: Comments and strings in French, but logic in English patterns

## Integration Points
- **HuggingFace**: Authentication required for gated datasets (LMSYS, etc.)
- **Docker Volumes**: Mount `./data_clean` and `./trained_models` for persistence
- **GPU**: CUDA with `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` for memory management
- **Dashboard API**: Endpoints for training control, metrics streaming at `/api/metrics`

## Examples
- **Resume Training**: `python scripts/train_subtitles_transformer.py --resume-from trained_models/runs/french_medium_base_60k/checkpoint_step_60000.pt --max-steps 115000`
- **Data Combination**: `python scripts/combine_text_corpus.py` creates `conversations_mega_train.txt` and `conversations_mega_test.txt`
- **Model Loading**: `checkpoint = torch.load(path, map_location='cpu', weights_only=True); trainer.model.load_state_dict(checkpoint['model_state'])`
- **Sample Generation**: Use `trainer.sample_text()` for inference, clean BPE with `text.replace('Ġ', ' ').replace('Ċ', '\n')`

Focus on PyTorch best practices, French language handling, and incremental training phases.</content>
<parameter name="filePath">/home/vincent/code/repo/french-llm-from-scratch/.github/copilot-instructions.md