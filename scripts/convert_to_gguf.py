#!/usr/bin/env python3
"""
Convertit le modèle Hugging Face en format GGUF pour LM Studio
Nécessite: pip install gguf
"""
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convertir modèle HF vers GGUF pour LM Studio")
    parser.add_argument(
        '--model-dir',
        type=str,
        default='trained_models/huggingface/french-llm-from-scratch',
        help='Répertoire du modèle Hugging Face'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='trained_models/gguf',
        help='Répertoire de sortie pour les fichiers GGUF'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='vincent-pro-ai/french-llm-from-scratch',
        help='Repo ID Hugging Face (pour téléchargement si besoin)'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Télécharger depuis HF au lieu d\'utiliser le dossier local'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔄 CONVERSION VERS FORMAT GGUF (LM Studio)")
    print("=" * 70)
    
    # Vérifier les dépendances
    try:
        import gguf
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("\n📦 Installation requise:")
        print("   pip install gguf transformers torch")
        sys.exit(1)
    
    # Déterminer le chemin du modèle
    if args.download:
        print(f"📥 Téléchargement depuis Hugging Face: {args.repo_id}")
        model_path = args.repo_id
    else:
        model_path = Path(args.model_dir)
        if not model_path.exists():
            print(f"❌ Modèle local introuvable: {model_path}")
            print(f"💡 Utilisez --download pour télécharger depuis HF")
            sys.exit(1)
        print(f"📂 Chargement local: {model_path}")
    
    # Créer le dossier de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger le modèle et tokenizer
    print("⏳ Chargement du modèle...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    
    vocab_size = len(tokenizer)
    print(f"✅ Modèle chargé: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    print(f"✅ Vocab size: {vocab_size:,}")
    
    # Nom du fichier de sortie
    output_file = output_dir / "french-llm-from-scratch-f32.gguf"
    
    print(f"\n📝 Création du fichier GGUF: {output_file}")
    
    # Initialiser le writer GGUF
    gguf_writer = gguf.GGUFWriter(str(output_file), "gpt2")
    
    # Ajouter les métadonnées
    gguf_writer.add_name("french-llm-from-scratch")
    gguf_writer.add_description("French LLM trained from scratch - 260M params")
    gguf_writer.add_author("vincent-pro-ai")
    gguf_writer.add_url("https://huggingface.co/vincent-pro-ai/french-llm-from-scratch")
    gguf_writer.add_license("MIT")
    
    # Architecture GPT-2
    config = model.config
    gguf_writer.add_block_count(config.n_layer)
    gguf_writer.add_context_length(config.n_positions)
    gguf_writer.add_embedding_length(config.n_embd)
    gguf_writer.add_feed_forward_length(config.n_inner)
    gguf_writer.add_head_count(config.n_head)
    gguf_writer.add_layer_norm_eps(config.layer_norm_epsilon)
    
    # Tokenizer
    print("📝 Ajout du vocabulaire et des merges BPE...")
    tokens = []
    scores = []
    for i in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(i)
        tokens.append(token.encode('utf-8') if isinstance(token, str) else token)
        scores.append(0.0)  # Score par défaut pour BPE

    gguf_writer.add_tokenizer_model("gpt2")
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)

    # Ajouter les merges BPE depuis tokenizer.json (nécessaire pour GPT2/ByteLevel BPE)
    merges_path = Path(model_path) / "tokenizer.json" if isinstance(model_path, Path) else None
    merges = None
    try:
        if merges_path and merges_path.exists():
            import json as _json
            with open(merges_path, 'r', encoding='utf-8') as f:
                tok_json = _json.load(f)
            model_section = tok_json.get('model', {})
            merges = model_section.get('merges')
        else:
            # Essayer via backend_tokenizer si disponible
            backend = getattr(tokenizer, 'backend_tokenizer', None)
            if backend is not None and hasattr(backend, 'model') and hasattr(backend.model, 'merges'):
                merges = backend.model.merges
        if merges:
            # merges peut être une liste de paires ["A","B"] -> convertir en "A B"
            merges_str = []
            for m in merges:
                if isinstance(m, str):
                    merges_str.append(m)
                elif isinstance(m, (list, tuple)) and len(m) == 2:
                    a = m[0] if isinstance(m[0], str) else str(m[0])
                    b = m[1] if isinstance(m[1], str) else str(m[1])
                    merges_str.append(f"{a} {b}")
                else:
                    merges_str.append(str(m))
            gguf_writer.add_token_merges(merges_str)
            print(f"✅ Merges BPE ajoutés: {len(merges_str)}")
        else:
            print("⚠️  Merges BPE introuvables – certains chargeurs peuvent échouer")
    except Exception as e:
        print(f"⚠️  Impossible d'ajouter les merges BPE: {e}")
    
    # Tokens spéciaux
    gguf_writer.add_bos_token_id(tokenizer.bos_token_id or 1)
    gguf_writer.add_eos_token_id(tokenizer.eos_token_id or 2)
    gguf_writer.add_pad_token_id(tokenizer.pad_token_id or 0)
    gguf_writer.add_unk_token_id(tokenizer.unk_token_id or 0)
    
    # Convertir les poids
    print("⏳ Conversion des poids du modèle...")
    state_dict = model.state_dict()
    
    tensor_map = {
        # Embeddings (wpe handled via default mapping; wte handled manually for shape/padding)
        "transformer.wpe.weight": "position_embd.weight",
        
        # Output (lm_head handled manuellement for shape)
        
        # Final norm
        "transformer.ln_f.weight": "output_norm.weight",
        "transformer.ln_f.bias": "output_norm.bias",
    }
    
    # Layers
    for i in range(config.n_layer):
        layer_map = {
            f"transformer.h.{i}.ln_1.weight": f"blk.{i}.attn_norm.weight",
            f"transformer.h.{i}.ln_1.bias": f"blk.{i}.attn_norm.bias",
            f"transformer.h.{i}.attn.c_attn.weight": f"blk.{i}.attn_qkv.weight",
            f"transformer.h.{i}.attn.c_attn.bias": f"blk.{i}.attn_qkv.bias",
            f"transformer.h.{i}.attn.c_proj.weight": f"blk.{i}.attn_output.weight",
            f"transformer.h.{i}.attn.c_proj.bias": f"blk.{i}.attn_output.bias",
            f"transformer.h.{i}.ln_2.weight": f"blk.{i}.ffn_norm.weight",
            f"transformer.h.{i}.ln_2.bias": f"blk.{i}.ffn_norm.bias",
            f"transformer.h.{i}.mlp.c_fc.weight": f"blk.{i}.ffn_up.weight",
            f"transformer.h.{i}.mlp.c_fc.bias": f"blk.{i}.ffn_up.bias",
            f"transformer.h.{i}.mlp.c_proj.weight": f"blk.{i}.ffn_down.weight",
            f"transformer.h.{i}.mlp.c_proj.bias": f"blk.{i}.ffn_down.bias",
        }
        tensor_map.update(layer_map)
    
    # Ajouter embeddings token (padded + transposés pour GGUF: [emb, vocab])
    if "transformer.wte.weight" in state_dict:
        wte = state_dict["transformer.wte.weight"].cpu()
        vocab_model, emb = wte.shape
        vocab_tok = vocab_size
        if vocab_model != vocab_tok:
            # Padding pour aligner le vocab du modèle au tokenizer (ajouter ligne zéro pour pad/unk)
            import torch
            if vocab_model < vocab_tok:
                pad_rows = vocab_tok - vocab_model
                wte = torch.cat([wte, torch.zeros(pad_rows, emb)], dim=0)
                print(f"ℹ️  wte: padding {pad_rows} rows (model {vocab_model} -> tok {vocab_tok})")
            else:
                wte = wte[:vocab_tok, :]
                print(f"ℹ️  wte: truncation (model {vocab_model} -> tok {vocab_tok})")
        gguf_writer.add_tensor(
            "token_embd.weight",
            wte.t().contiguous().cpu().numpy(),
            raw_shape=(emb, vocab_tok, 1, 1)
        )
    
    # Ajouter lm_head (transposé pour GGUF: [emb, vocab]) si présent
    if "lm_head.weight" in state_dict:
        lm = state_dict["lm_head.weight"].cpu()
        vocab_model, emb = lm.shape
        vocab_tok = vocab_size
        if vocab_model != vocab_tok:
            import torch
            if vocab_model < vocab_tok:
                pad_rows = vocab_tok - vocab_model
                lm = torch.cat([lm, torch.zeros(pad_rows, emb)], dim=0)
                print(f"ℹ️  lm_head: padding {pad_rows} rows (model {vocab_model} -> tok {vocab_tok})")
            else:
                lm = lm[:vocab_tok, :]
                print(f"ℹ️  lm_head: truncation (model {vocab_model} -> tok {vocab_tok})")
        gguf_writer.add_tensor(
            "output.weight",
            lm.t().contiguous().cpu().numpy(),
            raw_shape=(emb, vocab_tok, 1, 1)
        )

    # Ajouter les autres tensors
    count = 0
    for hf_name, gguf_name in tensor_map.items():
        if hf_name in state_dict:
            tensor = state_dict[hf_name].cpu().numpy()
            gguf_writer.add_tensor(gguf_name, tensor)
            count += 1
            if count % 10 == 0:
                print(f"   ... {count} tensors ajoutés")
    
    print(f"✅ Total: {count} tensors convertis")
    
    # Écrire le fichier
    print("💾 Écriture du fichier GGUF...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    
    file_size_mb = output_file.stat().st_size / (1024**2)
    print(f"✅ GGUF créé: {output_file} ({file_size_mb:.1f} MB)")
    
    print("\n" + "=" * 70)
    print("✅ CONVERSION TERMINÉE")
    print("=" * 70)
    print(f"\n📂 Fichier GGUF (FP32): {output_file}")
    print(f"📊 Taille: {file_size_mb:.1f} MB")
    
    print("\n🔧 PROCHAINES ÉTAPES:")
    print("\n1️⃣  Quantization (réduire la taille):")
    print("   Installez llama.cpp pour créer des versions quantisées:")
    print("   git clone https://github.com/ggerganov/llama.cpp")
    print("   cd llama.cpp && make")
    print()
    print(f"   # Q4_K_M (recommandé, ~70% plus petit)")
    print(f"   ./llama.cpp/llama-quantize {output_file} \\")
    print(f"       {output_dir}/french-llm-from-scratch-Q4_K_M.gguf Q4_K_M")
    print()
    print(f"   # Q5_K_M (meilleure qualité)")
    print(f"   ./llama.cpp/llama-quantize {output_file} \\")
    print(f"       {output_dir}/french-llm-from-scratch-Q5_K_M.gguf Q5_K_M")
    print()
    print(f"   # Q8_0 (haute qualité)")
    print(f"   ./llama.cpp/llama-quantize {output_file} \\")
    print(f"       {output_dir}/french-llm-from-scratch-Q8_0.gguf Q8_0")
    
    print("\n2️⃣  Utiliser avec LM Studio:")
    print(f"   - Ouvrir LM Studio")
    print(f"   - Aller dans 'Local Models'")
    print(f"   - Cliquer 'Import' et sélectionner le fichier .gguf")
    print(f"   - Ou copier directement dans: ~/.cache/lm-studio/models/")
    
    print("\n3️⃣  Tester avec llama.cpp:")
    print(f"   ./llama.cpp/llama-cli -m {output_dir}/french-llm-from-scratch-Q4_K_M.gguf \\")
    print(f"       -p \"Explique l'intelligence artificielle:\" -n 100")
    
    print("\n4️⃣  Publier sur Hugging Face (optionnel):")
    print("   python3 scripts/upload_to_huggingface.py \\")
    print(f"       --local-dir {output_dir} \\")
    print(f"       --repo-id vincent-pro-ai/french-llm-from-scratch-GGUF")


if __name__ == "__main__":
    main()
