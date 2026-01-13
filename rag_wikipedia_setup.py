#!/usr/bin/env python3
"""
Setup RAG avec Wikipedia FR pour le modèle 300k
Télécharge et prépare les données Wikipedia françaises
"""

import json
import requests
from pathlib import Path
from tqdm import tqdm
import bz2
import xml.etree.ElementTree as ET

# Configuration
MAX_ARTICLES = 100_000  # objectif demandé
DATA_DIR = Path("data_clean/wikipedia_rag")
WIKI_PARSED = DATA_DIR / "frwiki_articles.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_wikipedia():
    """Télécharge Wikipedia FR via HuggingFace Datasets (streaming)"""
    print("📥 Téléchargement de Wikipedia FR via HuggingFace Datasets...")
    print(f"⚙️  Cible: {MAX_ARTICLES} articles (streaming, peut prendre plusieurs minutes)")

    from datasets import load_dataset

    dataset = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)

    count = 0
    with open(WIKI_PARSED, 'w', encoding='utf-8') as f:
        for sample in dataset:
            f.write(json.dumps({
                'title': sample['title'],
                'text': sample['text'][:2000],  # Limiter la longueur
                'source': 'wikipedia_fr'
            }, ensure_ascii=False) + '\n')

            count += 1
            if count % 1000 == 0:
                print(f"   → {count} articles sauvegardés...")
            if count >= MAX_ARTICLES:
                break

    print(f"✅ {count} articles téléchargés et sauvegardés dans {WIKI_PARSED}")

def parse_wikipedia_xml():
    """Parse le XML Wikipedia et extrait les articles"""
    if WIKI_PARSED.exists():
        print(f"✅ Articles déjà parsés: {WIKI_PARSED}")
        return
    
    print("📖 Parsing des articles Wikipedia...")
    
    articles = []
    with bz2.open(WIKI_RAW, 'rt', encoding='utf-8') as f:
        content = f.read(50 * 1024 * 1024)  # Lire 50MB max pour le prototype
    
    # Parser XML simplifié (pour prototype)
    # En production, utiliser mwxml ou WikiExtractor
    import re
    
    # Extraire les pages avec regex simple
    page_pattern = re.compile(r'<page>(.*?)</page>', re.DOTALL)
    title_pattern = re.compile(r'<title>(.*?)</title>')
    text_pattern = re.compile(r'<text[^>]*>(.*?)</text>', re.DOTALL)
    
    pages = page_pattern.findall(content)
    
    for page in tqdm(pages[:1000], desc="Extraction des articles"):  # Limiter à 1000 articles pour démarrer
        title_match = title_pattern.search(page)
        text_match = text_pattern.search(page)
        
        if title_match and text_match:
            title = title_match.group(1)
            text = text_match.group(1)
            
            # Filtrer les pages spéciales
            if ':' in title or title.startswith('Wikipédia:'):
                continue
            
            # Nettoyer le texte (enlever la syntaxe wiki basique)
            text = re.sub(r'\{\{.*?\}\}', '', text)  # Templates
            text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', text)  # Links
            text = re.sub(r'==+\s*(.*?)\s*==+', r'\n\1\n', text)  # Headings
            text = re.sub(r"'{2,}", '', text)  # Bold/italic
            text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines
            text = text.strip()
            
            if len(text) > 200:  # Garder seulement les articles substantiels
                articles.append({
                    'title': title,
                    'text': text[:2000],  # Limiter à 2000 chars par article
                    'source': 'wikipedia_fr'
                })
    
    # Sauvegarder en JSONL
    with open(WIKI_PARSED, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')
    
    print(f"✅ {len(articles)} articles parsés et sauvegardés dans {WIKI_PARSED}")

def main():
    print("🚀 Setup RAG Wikipedia FR")
    print("=" * 50)

    if WIKI_PARSED.exists():
        print(f"✅ Articles déjà disponibles: {WIKI_PARSED}")
        with open(WIKI_PARSED, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print(f"   → {count} articles trouvés")
        if count < MAX_ARTICLES:
            print(f"   ⚠️  Seulement {count}/{MAX_ARTICLES}. Relance du téléchargement...")
            download_wikipedia()
    else:
        download_wikipedia()

    print("\n✅ Setup terminé!")
    print(f"📁 Articles disponibles dans: {WIKI_PARSED}")
    print("\n💡 Prochaine étape: Lancer 'python rag_index_wikipedia.py' pour indexer les articles")

if __name__ == "__main__":
    main()
