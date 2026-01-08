#!/usr/bin/env python3
"""API d'inférence pour le modèle français LLM - Checkpoint 300k."""
import sys
sys.path.insert(0, '/home/vincent/code/repo/french-llm-from-scratch')

import torch
from pathlib import Path
from scripts.train_subtitles_transformer import Config, SubtitleTrainer
from transformers import AutoTokenizer
import json
from datetime import datetime

# Configuration du checkpoint le plus récent (300k steps)
# Utilise le checkpoint principal écrasé à la fin de l'entraînement
CHECKPOINT_PATH = Path('/home/vincent/code/repo/french-llm-from-scratch/trained_models/tiny_subtitles_transformer.pt')
TOKENIZER_PATH = Path('data_clean/mistral_tokenizer')

print("[API] Démarrage de l'API d'inférence...")
print(f"[API] Checkpoint: {CHECKPOINT_PATH}")
print(f"[API] Tokenizer: {TOKENIZER_PATH}")

# Vérifier le checkpoint
if not CHECKPOINT_PATH.exists():
    print(f"❌ Checkpoint non trouvé: {CHECKPOINT_PATH}")
    sys.exit(1)

print(f"✓ Checkpoint trouvé: {CHECKPOINT_PATH.stat().st_size / 1e9:.2f} GB")

# Charger le checkpoint
try:
    import pathlib as _pl
    torch.serialization.add_safe_globals([_pl.PosixPath])
except Exception:
    pass

print("[API] Chargement du modèle...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

# Récupérer la config du checkpoint
ck_cfg = checkpoint.get('config', {})
ck_step = checkpoint.get('step', 'unknown')
ck_loss = checkpoint.get('val_loss', 'unknown')

print(f"[API] Étape d'entraînement: {ck_step}")
print(f"[API] Val loss: {ck_loss}")

# Configuration du modèle
cfg = Config()
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.tokenizer_path = TOKENIZER_PATH

# Appliquer la config du checkpoint
if isinstance(ck_cfg, dict):
    cfg.embed_dim = ck_cfg.get('embed_dim', cfg.embed_dim)
    cfg.num_layers = ck_cfg.get('num_layers', cfg.num_layers)
    cfg.num_heads = ck_cfg.get('num_heads', cfg.num_heads)
    cfg.ff_hidden_dim = ck_cfg.get('ff_hidden_dim', cfg.ff_hidden_dim)
    cfg.vocab_size = ck_cfg.get('vocab_size', cfg.vocab_size)
    cfg.block_size = ck_cfg.get('block_size', cfg.block_size)

# Disable auto-resume to avoid loading the wrong checkpoint
cfg.resume_checkpoint = None
cfg.resume_run_dir = None

print(f"[API] Architecture: {cfg.num_layers}L {cfg.num_heads}H {cfg.embed_dim}E {cfg.vocab_size}V")
print(f"[API] Device: {cfg.device}")

# Initialiser le trainer et le modèle
try:
    trainer = SubtitleTrainer(cfg)
    model = trainer.model.to(cfg.device)
    
    # Charger les poids
    model_state = checkpoint.get('model_state') or checkpoint.get('model')
    if model_state is None:
        print("❌ Pas de model_state dans le checkpoint")
        sys.exit(1)
    
    model.load_state_dict(model_state, strict=False)
    model.eval()
    print(f"✓ Modèle chargé avec succès")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Charger le tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    print(f"✓ Tokenizer chargé: vocab_size={tokenizer.vocab_size}")
except Exception as e:
    print(f"❌ Erreur tokenizer: {e}")
    sys.exit(1)

# API Flask simple
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': RUN_NAME,
        'checkpoint_step': ck_step,
        'device': cfg.device,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Générer du texte basé sur un prompt."""
    try:
        data = request.json or {}
        prompt = data.get('prompt', 'Bonjour, je suis')
        max_tokens = min(int(data.get('max_tokens', 100)), 200)
        temperature = float(data.get('temperature', 0.7))
        top_p = float(data.get('top_p', 0.95))
        
        # Tokenizer
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(cfg.device)
        input_ids = inputs
        
        print(f"[GENERATE] Prompt: {prompt}")
        print(f"[GENERATE] Input shape: {input_ids.shape}")
        
        # Générer
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id or 2
            )
        
        # Décoder
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        return jsonify({
            'prompt': prompt,
            'generated_text': output_text,
            'tokens_generated': len(generated[0]) - len(input_ids[0]),
            'model': RUN_NAME,
            'checkpoint_step': ck_step,
            'device': cfg.device,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'model_name': RUN_NAME,
        'checkpoint_step': ck_step,
        'val_loss': ck_loss,
        'architecture': {
            'num_layers': cfg.num_layers,
            'num_heads': cfg.num_heads,
            'embed_dim': cfg.embed_dim,
            'ff_hidden_dim': cfg.ff_hidden_dim,
            'vocab_size': cfg.vocab_size,
            'block_size': cfg.block_size
        },
        'device': cfg.device,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🚀 API PRÊTE SUR http://127.0.0.1:5000")
    print(f"   Modèle: {RUN_NAME} (step {ck_step}, loss {ck_loss})")
    print(f"   Device: {cfg.device}")
    print(f"{'='*60}\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
