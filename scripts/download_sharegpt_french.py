#!/usr/bin/env python3
"""
Download ShareGPT French conversations for fine-tuning.
Tries multiple sources and filters for French content.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from langdetect import detect, LangDetectException
from tqdm import tqdm


def detect_language(text: str) -> str:
    """Detect language using langdetect library."""
    if not text or len(text.strip()) < 20:
        return "other"
    
    try:
        lang = detect(text[:1000])  # Check first 1000 chars
        return lang
    except LangDetectException:
        return "other"


def format_conversation(conv: Dict[str, Any]) -> Optional[str]:
    """Format conversation in Utilisateur/Assistant format."""
    if "conversations" not in conv:
        return None
    
    messages = conv["conversations"]
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    
    formatted_lines = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        role = msg.get("from", "").lower()
        value = msg.get("value", "").strip()
        
        if not value:
            continue
        
        if role in {"human", "user"}:
            formatted_lines.append(f"Utilisateur: {value}")
        elif role in {"gpt", "assistant"}:
            formatted_lines.append(f"Assistant: {value}")
    
    if len(formatted_lines) < 2:
        return None
    
    return "\n".join(formatted_lines) + "\n\n"


def download_openchat_sharegpt() -> List[str]:
    """Try OpenChat ShareGPT dataset."""
    print("📥 Tentative: openchat/openchat_sharegpt4_dataset...")
    try:
        ds = load_dataset("openchat/openchat_sharegpt4_dataset", split="train")
        print(f"✅ Chargé: {len(ds)} conversations")
        
        french_convs = []
        for item in tqdm(ds, desc="Filtrage français"):
            if isinstance(item, dict) and "conversations" in item:
                # Check first message for French
                first_msg = item["conversations"][0] if item["conversations"] else {}
                text = first_msg.get("value", "")
                if detect_language(text) == "fr":
                    formatted = format_conversation(item)
                    if formatted:
                        french_convs.append(formatted)
        
        print(f"✅ Conversations françaises trouvées: {len(french_convs)}")
        return french_convs
    except Exception as e:
        print(f"❌ Échec: {e}")
        return []


def download_anon8231_sharegpt() -> List[str]:
    """Try Anon8231 ShareGPT cleaned dataset."""
    print("📥 Tentative: anon8231489123/ShareGPT_Vicuna_unfiltered...")
    try:
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
        print(f"✅ Chargé: {len(ds)} conversations")
        
        french_convs = []
        for item in tqdm(ds, desc="Filtrage français"):
            if isinstance(item, dict) and "conversations" in item:
                first_msg = item["conversations"][0] if item["conversations"] else {}
                text = first_msg.get("value", "")
                if detect_language(text) == "fr":
                    formatted = format_conversation(item)
                    if formatted:
                        french_convs.append(formatted)
        
        print(f"✅ Conversations françaises trouvées: {len(french_convs)}")
        return french_convs
    except Exception as e:
        print(f"❌ Échec: {e}")
        return []


def download_french_instruct() -> List[str]:
    """Try French instruction datasets - using native language fields."""
    print("📥 Tentative: datasets français d'instructions...")
    french_convs = []
    
    # OpenAssistant/oasst2 - has 'lang' field!
    try:
        print(f"  📥 OpenAssistant/oasst2 (filtrage par lang='fr')...")
        ds = load_dataset("OpenAssistant/oasst2", split="train")
        
        # Group messages by message_tree_id to reconstruct conversations
        conversations_by_tree = {}
        for item in tqdm(ds, desc="Regroupement par conversation"):
            if item.get("lang") == "fr":  # Direct filter on lang field!
                tree_id = item.get("message_tree_id")
                if tree_id not in conversations_by_tree:
                    conversations_by_tree[tree_id] = []
                conversations_by_tree[tree_id].append(item)
        
        print(f"  ✅ {len(conversations_by_tree)} arbres de conversation français trouvés")
        
        # Format conversations
        for tree_id, messages in tqdm(conversations_by_tree.items(), desc="Formatage conversations"):
            # Sort by created_date or use parent structure
            messages.sort(key=lambda x: x.get("created_date", ""))
            
            conv_lines = []
            for msg in messages:
                role = msg.get("role", "")
                text = msg.get("text", "").strip()
                
                if not text:
                    continue
                
                if role == "prompter":
                    conv_lines.append(f"Utilisateur: {text}")
                elif role == "assistant":
                    conv_lines.append(f"Assistant: {text}")
            
            if len(conv_lines) >= 2:  # At least one exchange
                french_convs.append("\n".join(conv_lines) + "\n\n")
        
        print(f"  ✅ {len(french_convs)} conversations formatées")
        
    except Exception as e:
        print(f"  ❌ Échec OpenAssistant: {e}")
    
    # Try Databricks Dolly (fewer French, but good quality)
    try:
        print(f"  📥 Databricks Dolly-15k (détection langue)...")
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        count_before = len(french_convs)
        for item in tqdm(ds, desc="Filtrage Dolly"):
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            
            if instruction and response and detect_language(instruction) == "fr":
                conv = f"Utilisateur: {instruction}\nAssistant: {response}\n\n"
                french_convs.append(conv)
        
        print(f"  ✅ {len(french_convs) - count_before} conversations Dolly extraites")
    except Exception as e:
        print(f"  ❌ Échec Dolly: {e}")
    
    return french_convs


def main():
    parser = argparse.ArgumentParser(description="Download French ShareGPT conversations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_clean/sharegpt_french"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=10000,
        help="Maximum conversations to keep",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum conversation length in characters",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try multiple sources
    all_conversations = []
    
    print("\n🔍 Recherche de conversations françaises ShareGPT...\n")
    
    # Try OpenChat first (usually good quality)
    convs = download_openchat_sharegpt()
    all_conversations.extend(convs)
    
    # If not enough, try other sources
    if len(all_conversations) < args.max_conversations:
        convs = download_anon8231_sharegpt()
        all_conversations.extend(convs)
    
    # Add French instruction datasets
    if len(all_conversations) < args.max_conversations:
        convs = download_french_instruct()
        all_conversations.extend(convs)
    
    # Filter by length and deduplicate
    print(f"\n🧹 Nettoyage et filtrage...")
    unique_convs = list(set(all_conversations))
    filtered_convs = [c for c in unique_convs if len(c) >= args.min_length]
    
    print(f"✅ Total unique: {len(unique_convs)}")
    print(f"✅ Après filtrage longueur: {len(filtered_convs)}")
    
    # Limit to max
    final_convs = filtered_convs[:args.max_conversations]
    
    # Save
    output_file = args.output_dir / "sharegpt_french.txt"
    output_file.write_text("".join(final_convs), encoding="utf-8")
    
    total_chars = sum(len(c) for c in final_convs)
    total_tokens = total_chars // 4  # Rough estimate
    
    print(f"\n✅ Sauvegardé: {output_file}")
    print(f"📊 Stats:")
    print(f"   - Conversations: {len(final_convs)}")
    print(f"   - Caractères: {total_chars:,}")
    print(f"   - Tokens estimés: {total_tokens:,}")
    print(f"   - Taille fichier: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save metadata
    meta = {
        "conversations": len(final_convs),
        "total_chars": total_chars,
        "estimated_tokens": total_tokens,
        "min_length": args.min_length,
        "output_path": str(output_file),
    }
    meta_file = args.output_dir / "sharegpt_french_meta.json"
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
