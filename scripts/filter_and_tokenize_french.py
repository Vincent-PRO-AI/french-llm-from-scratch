#!/usr/bin/env python3
"""
Filtre et tokenise les fichiers texte de french_large/ pour créer un dataset français pur.
Utilise une détection de langue et des heuristiques pour filtrer le contenu anglais/code.
"""
import re
import torch
from pathlib import Path
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import Counter

# Mots français très fréquents pour la détection
FRENCH_STOPWORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'mais',
    'dans', 'sur', 'pour', 'avec', 'sans', 'par', 'ces', 'ses', 'son',
    'est', 'sont', 'être', 'avoir', 'faire', 'dire', 'peut', 'tout',
    'nous', 'vous', 'ils', 'elles', 'je', 'tu', 'il', 'elle', 'qui', 'que',
    'où', 'quand', 'comment', 'pourquoi', 'donc', 'car', 'comme', 'aussi'
}

ENGLISH_STOPWORDS = {
    'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
    'might', 'can', 'this', 'that', 'these', 'those', 'with', 'from', 'about'
}


def detect_language(text: str) -> str:
    """Détecte si un texte est français, anglais ou autre."""
    if not text or len(text) < 50:
        return "unknown"
    
    # Normaliser et extraire les mots
    words = re.findall(r'\b[a-zàâäéèêëïîôùûüÿæœç]+\b', text.lower())
    
    if not words:
        return "unknown"
    
    # Compter les mots français vs anglais
    french_count = sum(1 for w in words if w in FRENCH_STOPWORDS)
    english_count = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    
    total_words = len(words)
    french_ratio = french_count / total_words if total_words > 0 else 0
    english_ratio = english_count / total_words if total_words > 0 else 0
    
    # Détecter les accents français
    has_french_chars = bool(re.search(r'[àâäéèêëïîôùûüÿæœç]', text))
    
    # Décision
    if french_ratio > 0.08 or has_french_chars:
        if english_ratio < french_ratio * 0.7:
            return "french"
    
    if english_ratio > 0.10:
        return "english"
    
    return "unknown"


def is_good_french_text(text: str, min_length: int = 100) -> bool:
    """Vérifie si un texte est du français de qualité."""
    if not text or len(text) < min_length:
        return False
    
    # Rejeter si trop de caractères non-français
    alpha_chars = len(re.findall(r'[a-zA-ZàâäéèêëïîôùûüÿæœçÀÂÄÉÈÊËÏÎÔÙÛÜŸÆŒÇ]', text))
    total_chars = len(text)
    
    if total_chars > 0 and alpha_chars / total_chars < 0.5:
        return False
    
    # Rejeter si trop de lignes de code
    code_indicators = ['def ', 'class ', 'import ', 'function ', 'return ', '==', '!=', '&&', '||']
    code_count = sum(text.count(ind) for ind in code_indicators)
    
    if code_count > 5:
        return False
    
    # Vérifier la langue
    lang = detect_language(text)
    
    return lang == "french"


def filter_and_tokenize_directory(
    input_dir: Path,
    output_path: Path,
    tokenizer_path: Path,
    max_files: int = None,
    chunk_size: int = 10000
) -> None:
    """Filtre et tokenise tous les fichiers .txt d'un dossier."""
    
    print(f"🔍 Chargement du tokenizer depuis {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Trouver tous les fichiers .txt
    txt_files = sorted(input_dir.glob("*.txt"))
    
    if max_files:
        txt_files = txt_files[:max_files]
    
    print(f"📂 Trouvé {len(txt_files)} fichiers .txt")
    
    all_tokens = []
    french_files = 0
    rejected_files = 0
    total_chars_processed = 0
    
    for file_path in tqdm(txt_files, desc="Traitement des fichiers"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            total_chars_processed += len(content)
            
            # Filtrer le contenu
            if not is_good_french_text(content):
                rejected_files += 1
                continue
            
            # Tokeniser
            tokens = tokenizer.encode(content, add_special_tokens=False)
            all_tokens.extend(tokens)
            french_files += 1
            
            # Sauvegarder par chunks pour économiser la RAM
            if len(all_tokens) > chunk_size * 1000:
                print(f"\n💾 Sauvegarde intermédiaire: {len(all_tokens):,} tokens")
                temp_tensor = torch.tensor(all_tokens, dtype=torch.long)
                
                # Si c'est la première sauvegarde, créer le fichier
                if not output_path.exists():
                    torch.save(temp_tensor, output_path)
                else:
                    # Sinon, fusionner avec l'existant
                    existing = torch.load(output_path)
                    merged = torch.cat([existing, temp_tensor])
                    torch.save(merged, output_path)
                
                all_tokens = []
        
        except Exception as e:
            print(f"\n⚠️  Erreur sur {file_path.name}: {e}")
            continue
    
    # Sauvegarder les tokens restants
    if all_tokens:
        print(f"\n💾 Sauvegarde finale: {len(all_tokens):,} tokens")
        temp_tensor = torch.tensor(all_tokens, dtype=torch.long)
        
        if output_path.exists():
            existing = torch.load(output_path)
            merged = torch.cat([existing, temp_tensor])
            torch.save(merged, output_path)
        else:
            torch.save(temp_tensor, output_path)
    
    # Statistiques finales
    final_data = torch.load(output_path) if output_path.exists() else torch.tensor([])
    
    print("\n" + "="*80)
    print("✅ TOKENISATION TERMINÉE")
    print("="*80)
    print(f"📊 Fichiers traités: {len(txt_files)}")
    print(f"✅ Fichiers français acceptés: {french_files}")
    print(f"❌ Fichiers rejetés: {rejected_files}")
    print(f"📝 Caractères traités: {total_chars_processed:,}")
    print(f"🎯 Tokens générés: {len(final_data):,} ({len(final_data)/1e6:.1f}M)")
    print(f"💾 Fichier de sortie: {output_path}")
    print(f"   Taille: {output_path.stat().st_size / (1024**2):.1f} MB")
    
    # Échantillon de vérification
    if len(final_data) > 0:
        print("\n📖 Échantillon du dataset généré:")
        sample = final_data[:500].tolist()
        text = tokenizer.decode(sample)
        print(f"   {text[:300]}...")


def main():
    # Configuration
    input_dir = Path("data_clean/french_large")
    output_path = Path("data_clean/french_large_filtered_mistral_tokenized.pt")
    tokenizer_path = Path("data_clean/mistral_tokenizer")
    
    # Vérifier que les dossiers existent
    if not input_dir.exists():
        print(f"❌ Erreur: {input_dir} n'existe pas")
        return
    
    if not tokenizer_path.exists():
        print(f"❌ Erreur: {tokenizer_path} n'existe pas")
        return
    
    print("🚀 FILTRAGE ET TOKENISATION DES DONNÉES FRANÇAISES")
    print("="*80)
    print(f"📂 Source: {input_dir}")
    print(f"💾 Destination: {output_path}")
    print(f"🔤 Tokenizer: {tokenizer_path}")
    print("="*80)
    print()
    
    # Lancer le traitement
    filter_and_tokenize_directory(
        input_dir=input_dir,
        output_path=output_path,
        tokenizer_path=tokenizer_path,
        max_files=None  # Traiter tous les fichiers
    )


if __name__ == "__main__":
    main()
