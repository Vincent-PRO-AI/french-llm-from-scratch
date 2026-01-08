import torch
import numpy as np
from pathlib import Path

# Chemins
pt_path = Path("data_clean/french_large_filtered_mistral_tokenized.pt")
bin_path = Path("data_clean/french_large_filtered.bin")

print(f"🔄 Chargement de {pt_path}...")
data = torch.load(pt_path, map_location="cpu")

print(f"📊 Analyse: {len(data):,} tokens, dtype={data.dtype}")
# On utilise uint16 car le vocab est de 32000 (0..65535 suffit)
print(f"💾 Conversion en uint16 et sauvegarde vers {bin_path}...")
mmap = np.memmap(bin_path, dtype=np.uint16, mode='w+', shape=(len(data),))
mmap[:] = data.numpy().astype(np.uint16)
mmap.flush()

print("✅ Conversion terminée !")
print(f"Taille finale: {bin_path.stat().st_size / 1e9:.2f} GB")
