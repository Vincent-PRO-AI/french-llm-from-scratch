#!/usr/bin/env python3
"""
API RAG avec Wikipedia FR + Modèle 300k
"""

import sys
sys.path.insert(0, '/home/vincent/code/repo/french-llm-from-scratch')

import torch
from pathlib import Path
from flask import Flask, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Configuration
MODEL_PATH = Path('/home/vincent/code/repo/french-llm-from-scratch/trained_models/tiny_subtitles_transformer.pt')
TOKENIZER_PATH = Path('data_clean/mistral_tokenizer')
CHROMA_DIR = Path("chroma_db")

app = Flask(__name__)

# Charger le modèle et les outils
print("🔧 Chargement du modèle 300k...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model_state = checkpoint.get('model_state')
config = checkpoint.get('config', {})

print(f"   → Config: {config.get('n_layer', '?')} layers, {config.get('n_embd', '?')} embed_dim")

# Charger le tokenizer
print("🔧 Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))

# Charger l'embedding model
print("🔧 Chargement du modèle d'embeddings...")
embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Charger Chroma DB
print("💾 Connexion à Chroma DB...")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection("wikipedia_fr")

print("✅ API RAG prête!\n")

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Récupère le contexte pertinent depuis Wikipedia"""
    query_embedding = embed_model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Combiner les résultats
    context_parts = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        context_parts.append(f"**{meta['title']}**\n{doc[:500]}")
    
    return "\n\n---\n\n".join(context_parts)

def generate_answer(prompt: str, max_length: int = 200) -> str:
    """Génère une réponse avec le modèle (simplifié pour l'instant)"""
    # Pour l'instant, retourner un placeholder
    # En production, intégrer la génération complète avec le modèle
    return f"[Génération avec modèle 300k - À implémenter]\n\nPrompt reçu:\n{prompt[:500]}..."

@app.route('/rag', methods=['POST'])
def rag_query():
    """Endpoint principal RAG"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Question manquante'}), 400
    
    # 1. Récupérer le contexte
    context = retrieve_context(question, top_k=3)
    
    # 2. Construire le prompt
    prompt = f"""Contexte tiré de Wikipedia:

{context}

Question: {question}

Réponse:"""
    
    # 3. Générer la réponse
    answer = generate_answer(prompt)
    
    return jsonify({
        'question': question,
        'context': context,
        'prompt': prompt,
        'answer': answer
    })

@app.route('/search', methods=['POST'])
def search_only():
    """Endpoint de recherche sémantique pure (sans génération)"""
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({'error': 'Query manquante'}), 400
    
    query_embedding = embed_model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    formatted_results = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        formatted_results.append({
            'title': meta['title'],
            'content': doc,
            'source': meta['source']
        })
    
    return jsonify({
        'query': query,
        'results': formatted_results
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model': str(MODEL_PATH),
        'chroma_db': str(CHROMA_DIR),
        'num_articles': collection.count()
    })

if __name__ == '__main__':
    print("🚀 Démarrage de l'API RAG sur http://0.0.0.0:5000")
    print("\nEndpoints disponibles:")
    print("  POST /rag       - RAG complet (récupération + génération)")
    print("  POST /search    - Recherche sémantique uniquement")
    print("  GET  /health    - Status de l'API")
    print("\n📝 Exemple de requête:")
    print('  curl -X POST http://localhost:5000/search -H "Content-Type: application/json" -d \'{"query": "Tour Eiffel"}\'')
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
