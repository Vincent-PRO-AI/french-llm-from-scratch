#!/usr/bin/env python3
"""Export checkpoint to HF format then GGUF for llama.cpp/LM Studio."""
import sys
sys.path.append('.')
import torch
from pathlib import Path
from scripts.train_subtitles_transformer import Config, SubtitleTrainer
import json

# Paths
RUN_DIR = Path('trained_models/runs/french_v3_finetune_grand_60k')
CKPT_PATH = sorted(RUN_DIR.glob('checkpoint_step_*.pt'), reverse=True)[0]
HF_OUTPUT = Path('trained_models/french-llm-v3-hf-final')
GGUF_OUTPUT = Path('trained_models/french-llm-v3-final.gguf')

print(f"Using checkpoint: {CKPT_PATH.name}")

# 1. Load checkpoint
try:
    import pathlib as _pl
    torch.serialization.add_safe_globals([_pl.PosixPath])
except Exception:
    pass

ck = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
model_state = ck.get('model_state') or ck.get('model')
ck_cfg = ck.get('config') or {}

cfg = Config()
if isinstance(ck_cfg, dict):
    cfg.embed_dim = ck_cfg.get('embed_dim', cfg.embed_dim)
    cfg.num_layers = ck_cfg.get('num_layers', cfg.num_layers)
    cfg.num_heads = ck_cfg.get('num_heads', cfg.num_heads)
    cfg.ff_hidden_dim = ck_cfg.get('ff_hidden_dim', cfg.ff_hidden_dim)
    cfg.vocab_size = ck_cfg.get('vocab_size', cfg.vocab_size)
    cfg.block_size = ck_cfg.get('block_size', cfg.block_size)

trainer = SubtitleTrainer(cfg)
model = trainer.model
model.load_state_dict(model_state, strict=False)
print(f"✅ Checkpoint loaded: {cfg.num_layers} layers, embed_dim={cfg.embed_dim}, vocab={cfg.vocab_size}")

# 2. Export to HF format
HF_OUTPUT.mkdir(parents=True, exist_ok=True)

# Save model.safetensors (HF format)
try:
    from safetensors.torch import save_file
    tensors = {k: v.clone().detach() for k, v in model.state_dict().items()}
    save_file(tensors, HF_OUTPUT / 'model.safetensors')
    print(f"✅ Saved HF model.safetensors to {HF_OUTPUT}")
except Exception:
    print("⚠️  safetensors not available, trying pytorch_model.bin")
    torch.save(model.state_dict(), HF_OUTPUT / 'pytorch_model.bin')
    print(f"✅ Saved pytorch_model.bin to {HF_OUTPUT}")

# Save config.json (HF format)
config_dict = {
    "architectures": ["MistralForCausalLM"],
    "attention_dropout": cfg.dropout,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "gelu",
    "hidden_size": cfg.embed_dim,
    "initializer_range": 0.02,
    "intermediate_size": cfg.ff_hidden_dim,
    "max_position_embeddings": cfg.block_size,
    "model_type": "mistral",
    "num_attention_heads": cfg.num_heads,
    "num_hidden_layers": cfg.num_layers,
    "num_key_value_heads": cfg.num_heads,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.36.0",
    "use_cache": True,
    "vocab_size": cfg.vocab_size,
}
with open(HF_OUTPUT / 'config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)
print(f"✅ Saved config.json")

# 3. Convert to GGUF
print("\n🔄 Converting to GGUF (this may take a few minutes)...")
import subprocess
result = subprocess.run(
    [
        sys.executable, 
        '-m', 'llama_cpp.llama_cpp_cli',
        '--input-model-path', str(HF_OUTPUT),
        '--model-name', 'french-llm-v3',
        '--output-model-path', str(GGUF_OUTPUT),
    ],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print(f"✅ GGUF conversion successful: {GGUF_OUTPUT}")
else:
    print(f"⚠️  llama_cpp CLI failed, trying alternative conversion...")
    # Fallback: use llama.cpp's convert.py if available
    import os
    os.chdir(str(HF_OUTPUT))
    result = subprocess.run(
        ['python3', '/path/to/llama.cpp/convert.py', '.', '--outtype', 'f16'],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"⚠️  Conversion failed. Install llama.cpp and try manual conversion:")
        print(f"    cd {HF_OUTPUT}")
        print(f"    python3 /path/to/llama.cpp/convert.py . --outtype f16")
    else:
        print(f"✅ GGUF created via convert.py")

print(f"\n✅ Export complete!")
print(f"   HF Model: {HF_OUTPUT}")
print(f"   GGUF File: {GGUF_OUTPUT}")
print(f"\n📖 Next steps:")
print(f"   1. Copy {GGUF_OUTPUT} to LM Studio models folder")
print(f"   2. Or load via: llama-cli -m {GGUF_OUTPUT} -p 'Bonjour, je suis '")
