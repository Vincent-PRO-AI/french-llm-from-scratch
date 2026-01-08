#!/usr/bin/env python3
"""
Test du tokenizer V2 pour vérifier l'absence d'artefacts BPE (Ġ, Ċ)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Exemples de textes français
TEXTES_TEST = [
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Le français est une belle langue avec des accents : é, è, ê, à, ù.",
    "L'intelligence artificielle révolutionne notre monde.",
    "Je m'appelle Vincent et j'adore programmer en Python.",
    "Les réseaux de neurones transformers sont très puissants.",
    "Voici un texte avec plusieurs phrases.\nEt voici une nouvelle ligne.",
    "Question : Quelle est la capitale de la France ?\nRéponse : Paris.",
]

def test_sentencepiece_tokenizer():
    """Test avec SentencePiece (tokenizer.model)"""
    try:
        import sentencepiece as spm
        
        tokenizer_path = ROOT / "trained_models/tokenizers/vincent_tokenizer_FR_v2/tokenizer.model"
        
        if not tokenizer_path.exists():
            print(f"❌ Tokenizer non trouvé: {tokenizer_path}")
            return False
        
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_path))
        
        print("\n" + "="*80)
        print("🔍 TEST TOKENIZER V2 - SentencePiece")
        print("="*80)
        print(f"\n📍 Tokenizer: {tokenizer_path}")
        print(f"📊 Vocab Size: {sp.vocab_size()}")
        print(f"🔢 PAD ID: {sp.pad_id()}")
        print(f"🔢 BOS ID: {sp.bos_id()}")
        print(f"🔢 EOS ID: {sp.eos_id()}")
        print(f"🔢 UNK ID: {sp.unk_id()}")
        
        print("\n" + "="*80)
        print("🧪 TESTS D'ENCODAGE/DÉCODAGE")
        print("="*80)
        
        has_artifacts = False
        
        for i, texte in enumerate(TEXTES_TEST, 1):
            print(f"\n📝 Test {i}:")
            print(f"   Input: {texte[:80]}...")
            
            # Encoder
            tokens = sp.encode(texte)
            token_pieces = sp.encode(texte, out_type=str)
            
            print(f"   Tokens: {len(tokens)} tokens")
            print(f"   Pieces: {' | '.join(token_pieces[:10])}{'...' if len(token_pieces) > 10 else ''}")
            
            # Décoder
            decoded = sp.decode(tokens)
            print(f"   Output: {decoded[:80]}...")
            
            # Vérifier artefacts BPE
            artifacts = []
            if 'Ġ' in decoded:
                artifacts.append('Ġ (espace BPE)')
            if 'Ċ' in decoded:
                artifacts.append('Ċ (newline BPE)')
            if '▁' in decoded:  # SentencePiece utilise ▁ pour les espaces
                # C'est normal pour SentencePiece, mais on vérifie qu'ils sont bien décodés
                if '▁' in decoded:  # Si visible dans le décodage final = problème
                    artifacts.append('▁ (espace SentencePiece non décodé)')
            
            # Vérifier correspondance
            match = (texte.strip() == decoded.strip())
            
            if artifacts:
                print(f"   ⚠️  Artefacts détectés: {', '.join(artifacts)}")
                has_artifacts = True
            elif not match:
                print(f"   ⚠️  Décodage différent de l'input")
                has_artifacts = True
            else:
                print(f"   ✅ Pas d'artefacts, décodage correct")
        
        print("\n" + "="*80)
        print("📊 ÉCHANTILLON DU VOCABULAIRE")
        print("="*80)
        
        # Afficher quelques tokens du vocab
        print("\nPremiers 20 tokens:")
        for i in range(min(20, sp.vocab_size())):
            piece = sp.id_to_piece(i)
            print(f"  {i:5d}: {piece}")
        
        print("\nTokens français courants (si présents):")
        french_words = ["le", "la", "de", "et", "est", "que", "pour", "dans", "avec", "bonjour"]
        for word in french_words:
            tokens = sp.encode(word, out_type=str)
            print(f"  '{word}' → {tokens}")
        
        print("\n" + "="*80)
        if has_artifacts:
            print("❌ RÉSULTAT: Artefacts BPE détectés!")
            print("   → Le tokenizer nécessite un post-processing")
            return False
        else:
            print("✅ RÉSULTAT: Tokenizer V2 propre, sans artefacts BPE!")
            print("   → Prêt pour l'entraînement et la publication")
            return True
        
    except ImportError:
        print("❌ Module sentencepiece non installé")
        print("   Installation: pip install sentencepiece")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transformers_tokenizer():
    """Test avec Transformers (si disponible)"""
    try:
        from transformers import AutoTokenizer
        
        tokenizer_path = ROOT / "trained_models/tokenizers/vincent_tokenizer_FR_v2"
        
        print("\n" + "="*80)
        print("🔍 TEST TOKENIZER V2 - HuggingFace Transformers")
        print("="*80)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            print("✅ Tokenizer chargé avec Transformers")
            print(f"📊 Vocab Size: {len(tokenizer)}")
            
            for texte in TEXTES_TEST[:3]:
                encoded = tokenizer.encode(texte)
                decoded = tokenizer.decode(encoded)
                print(f"\n   Input:  {texte[:60]}...")
                print(f"   Output: {decoded[:60]}...")
                
        except Exception as e:
            print(f"⚠️  Tokenizer pas compatible Transformers (normal pour SentencePiece brut)")
            print(f"   Utiliser SentencePiece directement")
            
    except ImportError:
        print("ℹ️  Transformers non installé (optionnel)")

def main():
    print("\n╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║  🧪 TEST TOKENIZER V2 - Vérification Artefacts BPE                       ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    
    # Test principal avec SentencePiece
    success = test_sentencepiece_tokenizer()
    
    # Test optionnel avec Transformers
    test_transformers_tokenizer()
    
    print("\n╔═══════════════════════════════════════════════════════════════════════════╗")
    if success:
        print("║  ✅ TOKENIZER V2 VALIDÉ - Prêt pour entraînement et publication          ║")
    else:
        print("║  ❌ TOKENIZER V2 À CORRIGER - Artefacts détectés                         ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
