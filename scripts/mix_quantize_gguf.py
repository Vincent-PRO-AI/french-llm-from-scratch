#!/usr/bin/env python3
"""
Créer une version GGUF "mixte": toutes les couches quantisées (depuis Q4_K_M) mais garder la tête de sortie en FP32.
- Source FP32: trained_models/gguf/french-llm-from-scratch-f32.gguf
- Source Q4_K_M: trained_models/gguf/french-llm-from-scratch-Q4_K_M.gguf
- Sortie mixte: trained_models/gguf/french-llm-from-scratch-mixed-Q4-F32head.gguf
"""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Assembler GGUF mixte (corps quantisé, tête FP32)")
    parser.add_argument('--fp32', type=str, default='trained_models/gguf/french-llm-from-scratch-f32.gguf')
    parser.add_argument('--q4', type=str, default='trained_models/gguf/french-llm-from-scratch-Q4_K_M.gguf')
    parser.add_argument('--output', type=str, default='trained_models/gguf/french-llm-from-scratch-mixed-Q4-F32head.gguf')
    args = parser.parse_args()

    try:
        import gguf
        import numpy as np
    except Exception as e:
        print("❌ gguf non installé. Faites: pip install gguf")
        print(e)
        sys.exit(1)

    fp32_path = Path(args.fp32)
    q4_path = Path(args.q4)
    out_path = Path(args.output)

    if not fp32_path.exists() or not q4_path.exists():
        print("❌ Fichiers source introuvables:")
        print("   FP32:", fp32_path)
        print("   Q4_K_M:", q4_path)
        sys.exit(1)

    print("📥 Lecture FP32 et Q4_K_M...")
    fp32 = gguf.GGUFReader(fp32_path)
    q4 = gguf.GGUFReader(q4_path)

    # Copie des métadonnées (KV) de Q4 pour cohérence, mais on mettra output.weight de FP32
    print("📝 Préparation writer mixte...")
    writer = gguf.GGUFWriter(str(out_path), q4.get_architecture())

    # Copier KV depuis Q4 (incluant tokenizer et merges)
    for key, val in q4.get_kv_data().items():
        vtype = val.type
        sub = val.sub_type
        writer.add_key_value(key, val.value, vtype, sub)

    # Copier tensors: par défaut ceux de Q4, sauf output.weight et token_embd.weight si besoin
    tensors_q4 = {name: q4.get_tensor(name) for name in q4.get_tensor_names()}
    tensors_fp32 = {name: fp32.get_tensor(name) for name in fp32.get_tensor_names()}

    # Helper pour ajouter un tensor numpy avec même shape/dtype que source
    def add_tensor_from(reader_tensor, name):
        arr = reader_tensor.data
        # Shape pour GGUF: reader donne déjà les dims
        shape = tuple(reader_tensor.shape)
        dtype = reader_tensor.data.dtype
        writer.add_tensor_info(name, shape, dtype, arr.nbytes)
        # On écrit ensuite après TI
        writer.tensors[-1][name].tensor = arr

    # D'abord ajouter tous les tensors Q4
    for name, t in tensors_q4.items():
        add_tensor_from(t, name)

    # Remplacer output.weight par FP32
    if 'output.weight' in tensors_fp32:
        print("🔁 Remplacement output.weight par FP32")
        writer.tensors[-1]['output.weight'].tensor = tensors_fp32['output.weight'].data
        # Assurer que nbytes correspond
        writer.tensors[-1]['output.weight'].nbytes = tensors_fp32['output.weight'].data.nbytes
    else:
        print("⚠️ output.weight absent dans FP32, on conserve Q4")

    # Remplacer token_embd.weight par FP32 (plus stable pour LM Studio)
    if 'token_embd.weight' in tensors_fp32:
        print("🔁 Remplacement token_embd.weight par FP32")
        writer.tensors[-1]['token_embd.weight'].tensor = tensors_fp32['token_embd.weight'].data
        writer.tensors[-1]['token_embd.weight'].nbytes = tensors_fp32['token_embd.weight'].data.nbytes

    print("💾 Écriture du fichier mixte...")
    writer.write_header_to_file(out_path)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"✅ GGUF mixte créé: {out_path}")

if __name__ == '__main__':
    main()
