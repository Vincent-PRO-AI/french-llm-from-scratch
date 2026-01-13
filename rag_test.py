#!/usr/bin/env python3
"""
Test du système RAG
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Test du endpoint health"""
    print("🏥 Test Health Check...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_search(query: str):
    """Test de recherche sémantique"""
    print(f"🔍 Test Recherche: '{query}'")
    response = requests.post(
        f"{API_URL}/search",
        json={"query": query, "top_k": 3}
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"   → {len(results['results'])} résultats trouvés\n")
        
        for i, result in enumerate(results['results'], 1):
            print(f"   {i}. {result['title']}")
            print(f"      {result['content'][:150]}...\n")
    else:
        print(f"   ❌ Erreur: {response.status_code}")
    print()

def test_rag(question: str):
    """Test RAG complet"""
    print(f"💬 Test RAG: '{question}'")
    response = requests.post(
        f"{API_URL}/rag",
        json={"question": question}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n📚 Contexte récupéré:")
        print(result['context'][:500])
        print("\n🤖 Réponse générée:")
        print(result['answer'])
    else:
        print(f"   ❌ Erreur: {response.status_code}")
    print()

if __name__ == "__main__":
    print("🧪 Tests du système RAG Wikipedia FR")
    print("=" * 60)
    print()
    
    # Test 1: Health check
    try:
        test_health()
    except Exception as e:
        print(f"❌ Erreur health check: {e}")
        print("💡 Assurez-vous que l'API est lancée avec: python rag_api.py")
        exit(1)
    
    # Test 2: Recherches sémantiques
    test_search("Quelle est la capitale de la France ?")
    test_search("Tour Eiffel")
    test_search("Histoire de Paris")
    
    # Test 3: RAG complet
    test_rag("Où se trouve la Tour Eiffel ?")
    
    print("✅ Tests terminés!")
