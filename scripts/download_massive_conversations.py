#!/usr/bin/env python3
"""
Télécharge massivement des conversations françaises depuis HuggingFace
Objectif: 100M+ tokens pour améliorer significativement le modèle
"""
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = Path("data_clean/conversations_massive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def count_tokens_approximate(text: str) -> int:
    """Estimation rapide: ~1.3 tokens par mot en français"""
    return int(len(text.split()) * 1.3)

def format_conversation(messages: list) -> str:
    """Formate une conversation au format Utilisateur/Assistant"""
    formatted = []
    for msg in messages:
        role = msg.get('role', msg.get('from', 'unknown'))
        content = msg.get('content', msg.get('value', ''))
        
        if role in ['user', 'human', 'utilisateur']:
            formatted.append(f"Utilisateur: {content}")
        elif role in ['assistant', 'gpt', 'bot']:
            formatted.append(f"Assistant: {content}")
    
    return "\n".join(formatted)

def download_oasst2_french():
    """
    OpenAssistant Conversations Dataset v2
    ~160k conversations, multilingual (filtrer FR)
    Objectif: ~30-40M tokens français
    """
    print("\n📥 Téléchargement OASST2 (conversations complètes)...")
    
    try:
        dataset = load_dataset("OpenAssistant/oasst2", split="train")
        
        output_file = OUTPUT_DIR / "oasst2_fr.txt"
        meta_file = OUTPUT_DIR / "oasst2_meta.json"
        
        conversations = []
        total_tokens = 0
        french_count = 0
        
        # Grouper par conversation (message_tree_id)
        conversations_dict = {}
        for item in tqdm(dataset, desc="Parsing OASST2"):
            tree_id = item.get('message_tree_id')
            lang = item.get('lang', '')
            
            if lang in ['fr', 'fr-FR', 'fr-CA']:
                if tree_id not in conversations_dict:
                    conversations_dict[tree_id] = []
                conversations_dict[tree_id].append({
                    'role': item.get('role'),
                    'content': item.get('text', '')
                })
        
        # Formater et écrire
        with open(output_file, 'w', encoding='utf-8') as f:
            for tree_id, messages in tqdm(conversations_dict.items(), desc="Writing OASST2"):
                formatted = format_conversation(messages)
                if formatted:
                    f.write(formatted + "\n\n")
                    tokens = count_tokens_approximate(formatted)
                    total_tokens += tokens
                    french_count += 1
        
        meta = {
            "source": "OpenAssistant/oasst2",
            "conversations": french_count,
            "tokens_estimate": total_tokens,
            "language": "fr"
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✅ OASST2: {french_count:,} conversations FR (~{total_tokens/1e6:.1f}M tokens)")
        return total_tokens
        
    except Exception as e:
        print(f"❌ Erreur OASST2: {e}")
        return 0

def download_ultrachat_french():
    """
    UltraChat Filtered
    ~200k conversations, high quality
    Filtrer pour FR uniquement
    Objectif: ~40-50M tokens
    """
    print("\n📥 Téléchargement UltraChat (filtered, FR only)...")
    
    try:
        # Utiliser la version 200k
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        
        output_file = OUTPUT_DIR / "ultrachat_fr.txt"
        meta_file = OUTPUT_DIR / "ultrachat_meta.json"
        
        total_tokens = 0
        french_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Processing UltraChat"):
                messages = item.get('messages', [])
                
                # Détection simple français: chercher mots français communs
                text_sample = " ".join([m.get('content', '')[:200] for m in messages[:2]])
                french_indicators = ['le ', 'la ', 'les ', 'de ', 'et ', 'est ', 'que ', 'pour ', 'dans ']
                
                if any(indicator in text_sample.lower() for indicator in french_indicators):
                    formatted = format_conversation(messages)
                    if formatted and len(formatted) > 100:
                        f.write(formatted + "\n\n")
                        tokens = count_tokens_approximate(formatted)
                        total_tokens += tokens
                        french_count += 1
                        
                        # Limiter pour ne pas tout télécharger
                        if french_count >= 30000:  # ~40M tokens max
                            break
        
        meta = {
            "source": "HuggingFaceH4/ultrachat_200k",
            "conversations": french_count,
            "tokens_estimate": total_tokens,
            "language": "fr (filtered)"
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✅ UltraChat: {french_count:,} conversations FR (~{total_tokens/1e6:.1f}M tokens)")
        return total_tokens
        
    except Exception as e:
        print(f"❌ Erreur UltraChat: {e}")
        return 0

def download_sharegpt_french():
    """
    ShareGPT conversations (filtré FR)
    ~90k conversations totales
    Objectif: ~15-20M tokens FR
    """
    print("\n📥 Téléchargement ShareGPT (conversations FR)...")
    
    try:
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
        
        output_file = OUTPUT_DIR / "sharegpt_fr.txt"
        meta_file = OUTPUT_DIR / "sharegpt_meta.json"
        
        total_tokens = 0
        french_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Processing ShareGPT"):
                conversations = item.get('conversations', [])
                
                # Vérifier si français
                text_sample = " ".join([c.get('value', '')[:200] for c in conversations[:2]])
                french_indicators = ['bonjour', 'merci', 'comment', 'pouvez', 'est-ce', 'français']
                
                if any(indicator in text_sample.lower() for indicator in french_indicators):
                    formatted = format_conversation(conversations)
                    if formatted and len(formatted) > 100:
                        f.write(formatted + "\n\n")
                        tokens = count_tokens_approximate(formatted)
                        total_tokens += tokens
                        french_count += 1
        
        meta = {
            "source": "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "conversations": french_count,
            "tokens_estimate": total_tokens,
            "language": "fr (filtered)"
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✅ ShareGPT: {french_count:,} conversations FR (~{total_tokens/1e6:.1f}M tokens)")
        return total_tokens
        
    except Exception as e:
        print(f"❌ Erreur ShareGPT: {e}")
        return 0

def download_camembert_conversations():
    """
    Camembert French Conversations
    Dataset spécifiquement français
    Objectif: ~10-15M tokens
    """
    print("\n📥 Téléchargement CamemBERT conversations...")
    
    try:
        # Chercher datasets français conversationnels
        dataset = load_dataset("openchat/openchat_sharegpt4_dataset", split="train")
        
        output_file = OUTPUT_DIR / "openchat_fr.txt"
        meta_file = OUTPUT_DIR / "openchat_meta.json"
        
        total_tokens = 0
        french_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Processing OpenChat"):
                items = item.get('items', [])
                
                # Filtrer français
                text_sample = " ".join([i.get('value', '')[:150] for i in items[:2]])
                if any(word in text_sample.lower() for word in ['le ', 'la ', 'de ', 'je ', 'vous ', 'est ']):
                    formatted = format_conversation(items)
                    if formatted and len(formatted) > 100:
                        f.write(formatted + "\n\n")
                        tokens = count_tokens_approximate(formatted)
                        total_tokens += tokens
                        french_count += 1
                        
                        if french_count >= 20000:  # Limiter
                            break
        
        meta = {
            "source": "openchat/openchat_sharegpt4_dataset",
            "conversations": french_count,
            "tokens_estimate": total_tokens,
            "language": "fr (filtered)"
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✅ OpenChat: {french_count:,} conversations FR (~{total_tokens/1e6:.1f}M tokens)")
        return total_tokens
        
    except Exception as e:
        print(f"❌ Erreur OpenChat: {e}")
        return 0

def download_french_instruct():
    """
    Vigogne Instruct French
    Dataset d'instructions en français
    Objectif: ~5-10M tokens
    """
    print("\n📥 Téléchargement Vigogne Instruct (instructions FR)...")
    
    try:
        dataset = load_dataset("bofenghuang/vigogne-instruct", split="train")
        
        output_file = OUTPUT_DIR / "vigogne_instruct_fr.txt"
        meta_file = OUTPUT_DIR / "vigogne_meta.json"
        
        total_tokens = 0
        french_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Processing Vigogne"):
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                # Formater comme conversation
                if instruction:
                    formatted = f"Utilisateur: {instruction}"
                    if input_text:
                        formatted += f"\n{input_text}"
                    formatted += f"\nAssistant: {output_text}"
                    
                    f.write(formatted + "\n\n")
                    tokens = count_tokens_approximate(formatted)
                    total_tokens += tokens
                    french_count += 1
        
        meta = {
            "source": "bofenghuang/vigogne-instruct",
            "conversations": french_count,
            "tokens_estimate": total_tokens,
            "language": "fr"
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✅ Vigogne Instruct: {french_count:,} instructions FR (~{total_tokens/1e6:.1f}M tokens)")
        return total_tokens
        
    except Exception as e:
        print(f"❌ Erreur Vigogne: {e}")
        return 0

def main():
    print("=" * 80)
    print("🚀 TÉLÉCHARGEMENT MASSIF DE CONVERSATIONS FRANÇAISES")
    print("=" * 80)
    print(f"\n📂 Répertoire de sortie: {OUTPUT_DIR}")
    print("\nObjectif: 100M+ tokens pour amélioration significative du modèle")
    print("=" * 80)
    
    total_tokens = 0
    
    # Télécharger tous les datasets
    total_tokens += download_oasst2_french()
    total_tokens += download_ultrachat_french()
    total_tokens += download_sharegpt_french()
    total_tokens += download_openchat_french()
    total_tokens += download_french_instruct()
    
    # Résumé final
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ FINAL")
    print("=" * 80)
    print(f"📂 Fichiers créés dans: {OUTPUT_DIR}")
    print(f"📈 Total estimé: ~{total_tokens/1e6:.1f}M tokens")
    print(f"🎯 Objectif atteint: {'✅ OUI' if total_tokens >= 100e6 else '⚠️  Proche'}")
    print("\n💡 Prochaines étapes:")
    print("  1. Combiner avec dataset existant (15M tokens)")
    print(f"  2. Total après combinaison: ~{(total_tokens + 15e6)/1e6:.1f}M tokens")
    print("  3. Tokeniser: bash scripts/tokenize_massive_conversations.sh")
    print("  4. Fine-tuner: 115k → 180k+ steps")
    print("  5. Amélioration attendue: +50-70% qualité")
    print("=" * 80)
    
    # Créer fichier de métadonnées global
    global_meta = {
        "total_tokens_estimate": total_tokens,
        "total_tokens_millions": round(total_tokens / 1e6, 1),
        "sources": [
            "OpenAssistant/oasst2",
            "HuggingFaceH4/ultrachat_200k",
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "openchat/openchat_sharegpt4_dataset",
            "bofenghuang/vigogne-instruct"
        ],
        "language": "fr",
        "date": "2025-11-17"
    }
    
    with open(OUTPUT_DIR / "global_meta.json", 'w', encoding='utf-8') as f:
        json.dump(global_meta, f, indent=2)

def download_openchat_french():
    """Wrapper pour éviter erreur de nom"""
    return download_camembert_conversations()

if __name__ == "__main__":
    main()
