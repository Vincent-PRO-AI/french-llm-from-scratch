#!/usr/bin/env python3
"""
Convertit le modèle HF en GGUF "mixte" (embeddings + head FP32) avec padding vocab = 32001
Objectif: compatibilité LM Studio sans erreurs de shape (token_embd/output 32001)

Usage:
  python3 scripts/convert_to_gguf_mixed.py \
      --model-dir trained_models/huggingface/french-llm-from-scratch \
      --output trained_models/gguf/french-llm-from-scratch-Q4_K_M-mixed.gguf
"""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Convertir modèle HF vers GGUF mixte (FP32 embeddings/head)")
    parser.add_argument('--model-dir', type=str, default='trained_models/huggingface/french-llm-from-scratch')
    parser.add_argument('--output', type=str, default='trained_models/gguf/french-llm-from-scratch-Q4_K_M-mixed.gguf')
    args = parser.parse_args()

    try:
        import gguf
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        print("❌ Dépendance manquante:", e)
        print("📦 Installez: pip install gguf transformers torch safetensors")
        sys.exit(1)

    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"❌ Modèle local introuvable: {model_path}")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🔄 CRÉATION GGUF MIXTE (FP32 embeddings/head) – padding 32001")
    print("=" * 70)

    print("📥 Chargement modèle/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()

    vocab_tok = len(tokenizer)
    cfg = model.config
    print(f"✅ Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M | Vocab tok: {vocab_tok}")

    writer = gguf.GGUFWriter(str(out_path), "gpt2")
    # Métadonnées
    writer.add_name("french-llm-from-scratch-mixed")
    writer.add_description("French LLM 260M – Mixed (FP32 emb/head), padded to 32001")
    writer.add_author("vincent-pro-ai")
    writer.add_url("https://huggingface.co/vincent-pro-ai/french-llm-from-scratch")
    writer.add_license("MIT")

    # Architecture
    writer.add_block_count(cfg.n_layer)
    writer.add_context_length(cfg.n_positions)
    writer.add_embedding_length(cfg.n_embd)
    writer.add_feed_forward_length(cfg.n_inner)
    writer.add_head_count(cfg.n_head)
    writer.add_layer_norm_eps(cfg.layer_norm_epsilon)

    # Tokenizer + merges
    print("📝 Ajout vocab + merges BPE...")
    tokens = []
    scores = []
    for i in range(vocab_tok):
        tok = tokenizer.convert_ids_to_tokens(i)
        tokens.append(tok.encode('utf-8') if isinstance(tok, str) else tok)
        scores.append(0.0)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)

    # merges depuis tokenizer.json
    merges_str = []
    try:
        import json
        tj = json.loads((model_path / 'tokenizer.json').read_text())
        merges_list = tj.get('model', {}).get('merges', [])
        for m in merges_list:
            if isinstance(m, str):
                merges_str.append(m)
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                merges_str.append(f"{m[0]} {m[1]}")
    except Exception as e:
        print(f"⚠️  Merges introuvables/illisibles: {e}")
    if merges_str:
        writer.add_token_merges(merges_str)
        print(f"✅ Merges: {len(merges_str)}")

    # Tokens spéciaux
    writer.add_bos_token_id(tokenizer.bos_token_id or 1)
    writer.add_eos_token_id(tokenizer.eos_token_id or 2)
    writer.add_pad_token_id(tokenizer.pad_token_id or 0)
    writer.add_unk_token_id(tokenizer.unk_token_id or 0)

    sd = model.state_dict()

    # Embeddings token (padding => vocab_tok, transposé => [emb, vocab])
    wte = sd["transformer.wte.weight"].cpu()
    vm, emb = wte.shape
    if vm != vocab_tok:
        if vm < vocab_tok:
            pad = vocab_tok - vm
            wte = torch.cat([wte, torch.zeros(pad, emb)], dim=0)
            print(f"ℹ️  wte: padding {pad} rows (model {vm} -> tok {vocab_tok})")
        else:
            wte = wte[:vocab_tok, :]
            print(f"ℹ️  wte: truncation (model {vm} -> tok {vocab_tok})")
    writer.add_tensor("token_embd.weight", wte.t().contiguous().numpy(), raw_shape=(emb, vocab_tok, 1, 1))

    # Position embeddings (transposé pour cohérence gpt2 gguf)
    if "transformer.wpe.weight" in sd:
        wpe = sd["transformer.wpe.weight"].cpu().numpy()
        writer.add_tensor("position_embd.weight", wpe.T)

    # lm_head (padding + transposé)
    lm = sd["lm_head.weight"].cpu()
    vm2, emb2 = lm.shape
    if vm2 != vocab_tok:
        if vm2 < vocab_tok:
            pad = vocab_tok - vm2
            lm = torch.cat([lm, torch.zeros(pad, emb2)], dim=0)
            print(f"ℹ️  lm_head: padding {pad} rows (model {vm2} -> tok {vocab_tok})")
        else:
            lm = lm[:vocab_tok, :]
            print(f"ℹ️  lm_head: truncation (model {vm2} -> tok {vocab_tok})")
    writer.add_tensor("output.weight", lm.t().contiguous().numpy(), raw_shape=(emb2, vocab_tok, 1, 1))

    # Final norm
    if "transformer.ln_f.weight" in sd:
        writer.add_tensor("output_norm.weight", sd["transformer.ln_f.weight"].cpu().numpy())
    if "transformer.ln_f.bias" in sd:
        writer.add_tensor("output_norm.bias", sd["transformer.ln_f.bias"].cpu().numpy())

    # Couches (FP32) – pas de transposition additionnelle ici, on suit le mapping du script principal
    added = 0
    for i in range(cfg.n_layer):
        p_old = f"transformer.h.{i}"
        p_new = f"blk.{i}"
        mapping = {
            f"{p_old}.ln_1.weight": f"{p_new}.attn_norm.weight",
            f"{p_old}.ln_1.bias": f"{p_new}.attn_norm.bias",
            f"{p_old}.attn.c_attn.weight": f"{p_new}.attn_qkv.weight",
            f"{p_old}.attn.c_attn.bias": f"{p_new}.attn_qkv.bias",
            f"{p_old}.attn.c_proj.weight": f"{p_new}.attn_output.weight",
            f"{p_old}.attn.c_proj.bias": f"{p_new}.attn_output.bias",
            f"{p_old}.ln_2.weight": f"{p_new}.ffn_norm.weight",
            f"{p_old}.ln_2.bias": f"{p_new}.ffn_norm.bias",
            f"{p_old}.mlp.c_fc.weight": f"{p_new}.ffn_up.weight",
            f"{p_old}.mlp.c_fc.bias": f"{p_new}.ffn_up.bias",
            f"{p_old}.mlp.c_proj.weight": f"{p_new}.ffn_down.weight",
            f"{p_old}.mlp.c_proj.bias": f"{p_new}.ffn_down.bias",
        }
        for k, v in mapping.items():
            if k in sd:
                writer.add_tensor(v, sd[k].cpu().numpy())
                added += 1
        if (i + 1) % 6 == 0:
            print(f"   ... {i+1}/{cfg.n_layer} couches ajoutées")

    print(f"✅ Tensors blocs ajoutés: {added}")

    print("💾 Écriture...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out_path.stat().st_size / (1024**2)
    print(f"✅ GGUF mixte écrit: {out_path} ({size_mb:.1f} MB)")
    print("⚠️  Remarque: fichier en FP32 complet; quantization sélective nécessite llama-quantize avancé.")


if __name__ == "__main__":
    main()
