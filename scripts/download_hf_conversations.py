#!/usr/bin/env python3
"""
Download multiple French conversation datasets from HuggingFace.
"""
import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset
from langdetect import detect, LangDetectException
from tqdm import tqdm


def detect_french(text: str) -> bool:
    """Check if text is in French."""
    if not text or len(text.strip()) < 20:
        return False
    try:
        return detect(text[:1000]) == "fr"
    except LangDetectException:
        return False


def download_ultrachat() -> List[str]:
    """Download UltraChat French conversations."""
    print("\n📥 UltraChat French...")
    conversations = []
    
    try:
        # Try UltraChat with French filter
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        print(f"   Loaded {len(ds)} conversations")
        
        for item in tqdm(ds.select(range(min(50000, len(ds)))), desc="Filtering French"):
            messages = item.get("messages", [])
            if len(messages) < 2:
                continue
            
            # Check first message for French
            first_text = messages[0].get("content", "")
            if detect_french(first_text):
                conv_lines = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "").strip()
                    
                    if role == "user":
                        conv_lines.append(f"Utilisateur: {content}")
                    elif role == "assistant":
                        conv_lines.append(f"Assistant: {content}")
                
                if len(conv_lines) >= 2:
                    conversations.append("\n".join(conv_lines) + "\n\n")
        
        print(f"   ✅ {len(conversations)} French conversations")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    return conversations


def download_wildchat() -> List[str]:
    """Download WildChat conversations (multilingual)."""
    print("\n📥 WildChat (multilingual)...")
    conversations = []
    
    try:
        ds = load_dataset("allenai/WildChat", split="train")
        print(f"   Loaded {len(ds)} conversations")
        
        for item in tqdm(ds.select(range(min(100000, len(ds)))), desc="Filtering French"):
            lang = item.get("language", "")
            if lang != "French":
                continue
            
            conversation = item.get("conversation", [])
            if len(conversation) < 2:
                continue
            
            conv_lines = []
            for msg in conversation:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                
                if role == "user":
                    conv_lines.append(f"Utilisateur: {content}")
                elif role == "assistant":
                    conv_lines.append(f"Assistant: {content}")
            
            if len(conv_lines) >= 2:
                conversations.append("\n".join(conv_lines) + "\n\n")
        
        print(f"   ✅ {len(conversations)} French conversations")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    return conversations


def download_lmsys_chat() -> List[str]:
    """Download LMSYS Chat Arena conversations."""
    print("\n📥 LMSYS Chat Arena...")
    conversations = []
    
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
        print(f"   Loaded {len(ds)} conversations")
        
        for item in tqdm(ds.select(range(min(100000, len(ds)))), desc="Filtering French"):
            lang = item.get("language", "")
            if lang not in ["French", "fr"]:
                # Try detection
                conv = item.get("conversation", [])
                if not conv or not detect_french(conv[0].get("content", "")):
                    continue
            
            conversation = item.get("conversation", [])
            if len(conversation) < 2:
                continue
            
            conv_lines = []
            for msg in conversation:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                
                if role == "user":
                    conv_lines.append(f"Utilisateur: {content}")
                elif role == "assistant":
                    conv_lines.append(f"Assistant: {content}")
            
            if len(conv_lines) >= 2:
                conversations.append("\n".join(conv_lines) + "\n\n")
        
        print(f"   ✅ {len(conversations)} French conversations")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    return conversations


def download_french_instruct_datasets() -> List[str]:
    """Download pure French instruction datasets."""
    print("\n📥 French instruction datasets...")
    conversations = []
    
    datasets_to_try = [
        "fblgit/simple-instructions-2023-08-09-fr",
        "bigscience/xP3",  # Multilingual with French
        "Anthropic/hh-rlhf",  # Has some French
    ]
    
    for dataset_name in datasets_to_try:
        try:
            print(f"\n   📥 {dataset_name}...")
            
            if dataset_name == "bigscience/xP3":
                # xP3 has language subsets
                ds = load_dataset(dataset_name, "fr", split="train")
            else:
                ds = load_dataset(dataset_name, split="train")
            
            print(f"      Loaded {len(ds)} items")
            
            count_before = len(conversations)
            for item in tqdm(ds.select(range(min(50000, len(ds)))), desc=f"   Processing"):
                # Try different field structures
                instruction = item.get("instruction") or item.get("inputs") or item.get("prompt", "")
                response = item.get("response") or item.get("targets") or item.get("chosen", "")
                
                if not instruction or not response:
                    continue
                
                # Check language
                if dataset_name != "bigscience/xP3":  # Already filtered
                    if not detect_french(instruction):
                        continue
                
                conv = f"Utilisateur: {instruction.strip()}\nAssistant: {response.strip()}\n\n"
                conversations.append(conv)
            
            added = len(conversations) - count_before
            print(f"      ✅ {added} conversations added")
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
    
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Download French conversation datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_clean/conversations_hf"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=50000,
        help="Maximum total conversations",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Downloading French conversation datasets from HuggingFace...\n")
    
    all_conversations = []
    
    # Try WildChat first (has language field)
    convs = download_wildchat()
    all_conversations.extend(convs)
    print(f"\n📊 Total so far: {len(all_conversations)}")
    
    # Try LMSYS if needed
    if len(all_conversations) < args.max_total:
        convs = download_lmsys_chat()
        all_conversations.extend(convs)
        print(f"\n📊 Total so far: {len(all_conversations)}")
    
    # Try UltraChat if needed
    if len(all_conversations) < args.max_total:
        convs = download_ultrachat()
        all_conversations.extend(convs)
        print(f"\n📊 Total so far: {len(all_conversations)}")
    
    # Add French instruction datasets
    if len(all_conversations) < args.max_total:
        convs = download_french_instruct_datasets()
        all_conversations.extend(convs)
        print(f"\n📊 Total so far: {len(all_conversations)}")
    
    # Deduplicate and limit
    print(f"\n🧹 Deduplicating...")
    unique_convs = list(set(all_conversations))
    print(f"   ✅ {len(unique_convs)} unique conversations")
    
    final_convs = unique_convs[:args.max_total]
    
    # Save
    output_file = args.output_dir / "conversations_hf_fr.txt"
    output_file.write_text("".join(final_convs), encoding="utf-8")
    
    total_chars = sum(len(c) for c in final_convs)
    total_tokens = total_chars // 4
    
    print(f"\n✅ Sauvegardé: {output_file}")
    print(f"\n📊 Stats finales:")
    print(f"   - Conversations: {len(final_convs):,}")
    print(f"   - Caractères: {total_chars:,}")
    print(f"   - Tokens estimés: {total_tokens:,}")
    print(f"   - Taille: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Metadata
    meta = {
        "conversations": len(final_convs),
        "total_chars": total_chars,
        "estimated_tokens": total_tokens,
        "output_path": str(output_file),
    }
    meta_file = args.output_dir / "meta.json"
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
