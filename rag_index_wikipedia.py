#!/usr/bin/env python3
"""
Indexe les articles Wikipedia FR dans Chroma DB
"""

import json
from pathlib import Path
import json
from pathlib import Path
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
DATA_DIR = Path("data_clean/wikipedia_rag")
WIKI_PARSED = DATA_DIR / "frwiki_articles.jsonl"
CHROMA_DIR = Path("chroma_db")


def iter_articles(batch_size: int = 32):
    """Itère sur les articles JSONL en lots pour limiter la mémoire"""
    batch = []
    with open(WIKI_PARSED, 'r', encoding='utf-8') as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def index_wikipedia():
    """Indexe les articles dans Chroma DB"""
    print("🔍 Chargement du modèle d'embeddings...")
    embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    print("💾 Initialisation de Chroma DB...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        chroma_client.delete_collection("wikipedia_fr")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="wikipedia_fr",
        metadata={"hnsw:space": "cosine"}
    )

    print("🚀 Indexation des articles...")
    batch_size = 64
    total = 0
    for batch in tqdm(iter_articles(batch_size=batch_size), desc="Indexation", unit="batch"):
        idx_start = total
        texts = [f"{art['title']}\n\n{art['text']}" for art in batch]
        ids = [f"wiki_{idx_start + j}" for j in range(len(batch))]
        metadatas = [{'title': art['title'], 'source': art['source']} for art in batch]

        embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        total += len(batch)

    print(f"\n✅ Indexation terminée!")
    print(f"   → {total} articles indexés dans {CHROMA_DIR}")
    print("\n💡 Prochaine étape: Lancer 'python rag_api.py' pour démarrer l'API RAG")


def test_retrieval():
    """Test rapide de récupération"""
    print("\n🧪 Test de récupération...")

    embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection("wikipedia_fr")

    query = "Quelle est la capitale de la France ?"
    query_embedding = embed_model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print(f"\n📝 Question: {query}")
    print("\n📚 Top 3 résultats:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"\n{i}. {meta['title']}")
        print(f"   {doc[:200]}...")


if __name__ == "__main__":
    index_wikipedia()
    test_retrieval()
