#!/usr/bin/env python3
"""
Crée un dataset conversations 100% français validé pour Phase 2.
Sources: OASST2 FR + Dolly FR (conversations vérifiées manuellement).
Évite la contamination anglaise qui a détruit les checkpoints précédents.
"""

import json
import re
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

ROOT = Path(__file__).resolve().parent.parent

# Mots anglais courants pour détecter contamination
ENGLISH_WORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
    'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how',
    'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did',
    'its', 'let', 'put', 'say', 'she', 'too', 'use', 'what', 'when', 'which'
}

def detect_language(text: str) -> Tuple[str, float]:
    """
    Détecte la langue d'un texte.
    Retourne: ('fr' ou 'en', score_confiance)
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    if not words:
        return 'unknown', 0.0
    
    # Compter mots anglais
    english_count = sum(1 for w in words if w in ENGLISH_WORDS)
    english_ratio = english_count / len(words)
    
    # Mots français typiques
    french_markers = ['le', 'la', 'les', 'un', 'une', 'des', 'est', 'sont', 
                      'pour', 'dans', 'avec', 'que', 'qui', 'sur', 'ou']
    french_count = sum(1 for w in words if w in french_markers)
    french_ratio = french_count / len(words)
    
    # Accents français
    has_accents = bool(re.search(r'[éèêëàâäùûüôöîïç]', text_lower))
    
    if french_ratio > english_ratio or has_accents:
        confidence = min(1.0, french_ratio + (0.2 if has_accents else 0))
        return 'fr', confidence
    elif english_ratio > 0.15:
        return 'en', english_ratio
    else:
        return 'unknown', 0.5

def validate_conversation(user_msg: str, assistant_msg: str) -> Dict[str, any]:
    """
    Valide qu'une conversation est en français et de qualité.
    """
    # Vérifier langue utilisateur
    user_lang, user_conf = detect_language(user_msg)
    
    # Vérifier langue assistant
    asst_lang, asst_conf = detect_language(assistant_msg)
    
    # Critères de validation
    is_valid = True
    reasons = []
    
    # 1. Langue doit être français
    if user_lang != 'fr' or asst_lang != 'fr':
        is_valid = False
        reasons.append(f"Langue incorrecte: user={user_lang}, asst={asst_lang}")
    
    # 2. Confiance minimale
    if user_conf < 0.6 or asst_conf < 0.6:
        is_valid = False
        reasons.append(f"Confiance faible: user={user_conf:.2f}, asst={asst_conf:.2f}")
    
    # 3. Longueur raisonnable
    if len(user_msg) < 5 or len(assistant_msg) < 5:
        is_valid = False
        reasons.append("Messages trop courts")
    
    if len(assistant_msg) > 1000:
        is_valid = False
        reasons.append("Réponse assistant trop longue (>1000 chars)")
    
    # 4. Pas de caractères bizarres
    if re.search(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', user_msg + assistant_msg):
        is_valid = False
        reasons.append("Caractères de contrôle détectés")
    
    return {
        'valid': is_valid,
        'reasons': reasons,
        'user_lang': user_lang,
        'asst_lang': asst_lang,
        'user_conf': user_conf,
        'asst_conf': asst_conf
    }

def load_oasst2_french(data_dir: Path) -> List[Dict]:
    """Charge OASST2 conversations françaises."""
    conversations = []
    
    oasst_file = data_dir / "conversations_hf" / "oasst2_conversations.jsonl"
    
    if not oasst_file.exists():
        print(f"⚠️  OASST2 non trouvé: {oasst_file}")
        return conversations
    
    print(f"📂 Chargement OASST2: {oasst_file}")
    
    with open(oasst_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                # Extraire conversation
                if 'conversations' in data:
                    for conv in data['conversations']:
                        if len(conv) >= 2:
                            user_msg = conv[0].get('value', '')
                            asst_msg = conv[1].get('value', '')
                            
                            if user_msg and asst_msg:
                                conversations.append({
                                    'user': user_msg,
                                    'assistant': asst_msg,
                                    'source': 'oasst2'
                                })
            except Exception as e:
                if line_num % 1000 == 0:
                    print(f"   Erreur ligne {line_num}: {e}")
    
    print(f"   ✓ {len(conversations)} conversations OASST2 chargées")
    return conversations

def load_dolly_french(data_dir: Path) -> List[Dict]:
    """Charge Dolly FR conversations."""
    conversations = []
    
    dolly_file = data_dir / "dolly_fr_tokenized.pt"
    dolly_txt = data_dir / "dolly_fr.txt"
    
    # Essayer version texte d'abord
    if dolly_txt.exists():
        print(f"📂 Chargement Dolly FR: {dolly_txt}")
        
        with open(dolly_txt, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Parser format "Utilisateur: ... Assistant: ..."
            pattern = r'Utilisateur:\s*(.+?)\s*Assistant:\s*(.+?)(?=Utilisateur:|$)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for user_msg, asst_msg in matches:
                user_msg = user_msg.strip()
                asst_msg = asst_msg.strip()
                
                if user_msg and asst_msg:
                    conversations.append({
                        'user': user_msg,
                        'assistant': asst_msg,
                        'source': 'dolly_fr'
                    })
    
    print(f"   ✓ {len(conversations)} conversations Dolly FR chargées")
    return conversations

def format_conversation(user: str, assistant: str) -> str:
    """Formate une conversation au format attendu."""
    return f"Utilisateur: {user}\nAssistant: {assistant}"

def create_validated_dataset(
    sources: List[str],
    output_path: Path,
    tokenizer_path: Path,
    max_tokens: int = 50_000_000,
    validation_threshold: float = 0.7
):
    """
    Crée le dataset validé.
    """
    print("\n" + "="*80)
    print("🚀 CRÉATION DATASET CONVERSATIONS VALIDÉ FR")
    print("="*80)
    
    data_dir = ROOT / "data_clean"
    
    # 1. Charger toutes les conversations
    all_conversations = []
    
    if 'oasst2' in sources:
        all_conversations.extend(load_oasst2_french(data_dir))
    
    if 'dolly_fr' in sources:
        all_conversations.extend(load_dolly_french(data_dir))
    
    print(f"\n📊 Total conversations chargées: {len(all_conversations)}")
    
    # 2. Validation
    print("\n🔍 VALIDATION DES CONVERSATIONS...")
    
    valid_conversations = []
    invalid_count = 0
    stats = {'en': 0, 'fr': 0, 'short': 0, 'long': 0, 'other': 0}
    
    for i, conv in enumerate(all_conversations):
        if i % 1000 == 0 and i > 0:
            print(f"   Traité: {i}/{len(all_conversations)} ({len(valid_conversations)} valides)")
        
        validation = validate_conversation(conv['user'], conv['assistant'])
        
        if validation['valid']:
            valid_conversations.append(conv)
        else:
            invalid_count += 1
            # Compter raisons
            for reason in validation['reasons']:
                if 'Langue' in reason:
                    if 'en' in reason:
                        stats['en'] += 1
                elif 'courts' in reason:
                    stats['short'] += 1
                elif 'longue' in reason:
                    stats['long'] += 1
                else:
                    stats['other'] += 1
    
    print(f"\n✅ Conversations valides: {len(valid_conversations)}")
    print(f"❌ Conversations rejetées: {invalid_count}")
    print(f"   Raisons:")
    print(f"   - Anglais: {stats['en']}")
    print(f"   - Trop court: {stats['short']}")
    print(f"   - Trop long: {stats['long']}")
    print(f"   - Autre: {stats['other']}")
    
    # 3. Formater
    print("\n📝 FORMATAGE DES CONVERSATIONS...")
    
    formatted_texts = []
    for conv in valid_conversations:
        text = format_conversation(conv['user'], conv['assistant'])
        formatted_texts.append(text)
    
    # 4. Tokenizer
    print(f"\n🔧 TOKENISATION avec {tokenizer_path}...")
    
    try:
        # Utiliser SentencePiece directement
        import sentencepiece as spm
        
        tokenizer_model = tokenizer_path / "tokenizer.model"
        if not tokenizer_model.exists():
            print(f"❌ Tokenizer model non trouvé: {tokenizer_model}")
            return False
        
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_model))
        
        print(f"   ✓ Tokenizer chargé: vocab_size={sp.vocab_size()}")
        
        all_tokens = []
        total_tokens = 0
        
        for i, text in enumerate(formatted_texts):
            if total_tokens >= max_tokens:
                print(f"   ⚠️  Limite {max_tokens} tokens atteinte")
                break
            
            tokens = sp.encode(text, out_type=int)
            all_tokens.extend(tokens)
            total_tokens += len(tokens)
            
            if i % 500 == 0 and i > 0:
                print(f"   Tokenisé: {i}/{len(formatted_texts)} ({total_tokens:,} tokens)")
        
        # Sauvegarder
        print(f"\n💾 SAUVEGARDE: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'tokens': torch.tensor(all_tokens, dtype=torch.long),
            'num_conversations': len(valid_conversations),
            'num_tokens': total_tokens,
            'sources': sources,
            'validation_threshold': validation_threshold,
        }, output_path)
        
        print(f"   ✓ Sauvegardé: {total_tokens:,} tokens, {len(valid_conversations)} conversations")
        
        # Stats finales
        size_mb = output_path.stat().st_size / 1_000_000
        print(f"   ✓ Taille: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Erreur tokenisation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ DATASET VALIDÉ CRÉÉ AVEC SUCCÈS!")
    print("="*80)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Créer dataset conversations FR validé")
    parser.add_argument('--sources', type=str, default='oasst2,dolly_fr',
                       help='Sources séparées par virgule')
    parser.add_argument('--max-tokens', type=int, default=50_000_000,
                       help='Nombre max de tokens')
    parser.add_argument('--tokenizer-path', type=str,
                       default='trained_models/tokenizers/fineweb-32k',
                       help='Chemin vers le tokenizer')
    parser.add_argument('--output', type=str,
                       default='data_clean/conversations_validated_fr.pt',
                       help='Fichier de sortie')
    parser.add_argument('--validation-threshold', type=float, default=0.7,
                       help='Seuil de confiance pour validation')
    
    args = parser.parse_args()
    
    sources = args.sources.split(',')
    output_path = ROOT / args.output
    tokenizer_path = ROOT / args.tokenizer_path
    
    success = create_validated_dataset(
        sources=sources,
        output_path=output_path,
        tokenizer_path=tokenizer_path,
        max_tokens=args.max_tokens,
        validation_threshold=args.validation_threshold
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
