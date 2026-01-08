
import torch
from pathlib import Path
from transformers import AutoTokenizer

def inspect_data():
    data_path = "data_clean/massive_french_clean.pt"
    tokenizer_path = "data_clean/mistral_tokenizer"
    
    if not Path(data_path).exists():
        print(f"Fichier {data_path} non trouvé.")
        return
        
    print(f"Chargement de {data_path}...")
    data = torch.load(data_path)
    
    print(f"Type des données: {type(data)}")
    if isinstance(data, torch.Tensor):
        print(f"Shape du tenseur: {data.shape}")
        print(f"Dtype: {data.dtype}")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Afficher les 500 premiers tokens
        sample_ids = data[:500].tolist()
        text = tokenizer.decode(sample_ids)
        print("\n--- Échantillon (Premiers 500 tokens) ---")
        print(text)
        
        # Afficher un échantillon au milieu
        mid = len(data) // 2
        sample_ids_mid = data[mid:mid+500].tolist()
        text_mid = tokenizer.decode(sample_ids_mid)
        print("\n--- Échantillon (Milieu du fichier) ---")
        print(text_mid)

    elif isinstance(data, dict):
        print(f"Clés du dictionnaire: {data.keys()}")
        if 'train' in data:
            print(f"Taille train: {len(data['train'])}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            sample_ids = data['train'][:500].tolist()
            print("\n--- Échantillon train ---")
            print(tokenizer.decode(sample_ids))

if __name__ == "__main__":
    inspect_data()
