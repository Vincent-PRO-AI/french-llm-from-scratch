#!/usr/bin/env python3
"""
Télécharge les meilleurs datasets de conversations françaises depuis HuggingFace.
Optimisé pour obtenir 50-100k conversations de haute qualité.
"""
import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm


def download_wildchat(max_samples: int = 20000) -> List[str]:
    """
    WildChat - Conversations ChatGPT réelles (meilleure qualité).
    Dataset: allenai/WildChat
    """
    print("\n📥 1. WildChat (conversations ChatGPT réelles)...")
    conversations = []
    
    try:
        # Télécharger en mode non-streaming avec limite
        print(f"   🔄 Chargement de {max_samples * 10} exemples...")
        ds = load_dataset("allenai/WildChat", split=f"train[:{max_samples * 10}]")
        print(f"   ✅ Chargé {len(ds):,} exemples")
        
        count = 0
        for item in tqdm(ds, desc="   Filtering French"):
            if count >= max_samples:
                break
                
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
                
                if not content:
                    continue
                    
                if role == "user":
                    conv_lines.append(f"Utilisateur: {content}")
                elif role == "assistant":
                    conv_lines.append(f"Assistant: {content}")
            
            if len(conv_lines) >= 2:
                conversations.append("\n".join(conv_lines) + "\n\n")
                count += 1
        
        print(f"   ✅ {len(conversations):,} conversations françaises")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    return conversations


def download_lmsys_chat(max_samples: int = 30000) -> List[str]:
    """
    LMSYS Chat-1M - Arena battles entre modèles.
    Dataset: lmsys/lmsys-chat-1m
    """
    print("\n📥 2. LMSYS Chat-1M (arena battles)...")
    conversations = []
    
    try:
        # Télécharger un sous-ensemble
        print(f"   🔄 Chargement de {max_samples * 10} exemples...")
        ds = load_dataset("lmsys/lmsys-chat-1m", split=f"train[:{max_samples * 10}]")
        print(f"   ✅ Chargé {len(ds):,} exemples")
        
        count = 0
        for item in tqdm(ds, desc="   Filtering French"):
            if count >= max_samples:
                break
            
            # Check language
            lang = item.get("language", "")
            if lang not in ["French", "fr", "fra"]:
                continue
            
            conversation = item.get("conversation", [])
            if len(conversation) < 2:
                continue
            
            conv_lines = []
            for msg in conversation:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                
                if not content:
                    continue
                
                if role == "user":
                    conv_lines.append(f"Utilisateur: {content}")
                elif role == "assistant":
                    conv_lines.append(f"Assistant: {content}")
            
            if len(conv_lines) >= 2:
                conversations.append("\n".join(conv_lines) + "\n\n")
                count += 1
        
        print(f"   ✅ {len(conversations):,} conversations françaises")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    return conversations


def download_openhermes(max_samples: int = 20000) -> List[str]:
    """
    OpenHermes 2.5 - Instructions GPT-4.
    Dataset: teknium/OpenHermes-2.5
    """
    print("\n📥 3. OpenHermes 2.5 (GPT-4 instructions)...")
    conversations = []
    
    try:
        # Télécharger un sous-ensemble
        print(f"   🔄 Chargement de {max_samples * 10} exemples...")
        ds = load_dataset("teknium/OpenHermes-2.5", split=f"train[:{max_samples * 10}]")
        print(f"   ✅ Chargé {len(ds):,} exemples")
        
        count = 0
        for item in tqdm(ds, desc="   Filtering French"):
            if count >= max_samples:
                break
            
            # Check language field or detect
            conversations_list = item.get("conversations", [])
            
            if not conversations_list or len(conversations_list) < 2:
                continue
            
            # Simple heuristic: check if first message has French words
            first_msg = conversations_list[0].get("value", "")
            french_indicators = ["le", "la", "de", "que", "est", "dans", "pour", "à", "quel"]
            if not any(word in first_msg.lower().split()[:20] for word in french_indicators):
                continue
            
            conv_lines = []
            for msg in conversations_list:
                role = msg.get("from", "")
                content = msg.get("value", "").strip()
                
                if not content:
                    continue
                
                if role in ["human", "user"]:
                    conv_lines.append(f"Utilisateur: {content}")
                elif role in ["gpt", "assistant"]:
                    conv_lines.append(f"Assistant: {content}")
            
            if len(conv_lines) >= 2:
                conversations.append("\n".join(conv_lines) + "\n\n")
                count += 1
        
        print(f"   ✅ {len(conversations):,} conversations françaises")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    return conversations


def download_vigogne(max_samples: int = 10000) -> List[str]:
    """
    Vigogne - Dataset français pur.
    Dataset: bofenghuang/vigogne-instruction-following-v2
    """
    print("\n📥 4. Vigogne (français pur)...")
    conversations = []
    
    try:
        ds = load_dataset("bofenghuang/vigogne-instruction-following-v2", split="train")
        print(f"   ✅ Chargé {len(ds):,} exemples")
        
        for item in tqdm(ds.select(range(min(max_samples, len(ds)))), desc="   Processing"):
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            output = item.get("output", "").strip()
            
            if not instruction or not output:
                continue
            
            # Combine instruction + input if present
            if input_text:
                user_msg = f"{instruction}\n{input_text}"
            else:
                user_msg = instruction
            
            conv = f"Utilisateur: {user_msg}\nAssistant: {output}\n\n"
            conversations.append(conv)
        
        print(f"   ✅ {len(conversations):,} conversations")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Télécharge les meilleurs datasets FR")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_clean/conversations_extra"),
        help="Dossier de sortie",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=80000,
        help="Nombre maximum de conversations",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🚀 Téléchargement des meilleurs datasets de conversations FR")
    print("=" * 70)
    
    all_conversations = []
    
    # 1. WildChat (20k)
    convs = download_wildchat(max_samples=20000)
    all_conversations.extend(convs)
    print(f"\n📊 Total: {len(all_conversations):,}")
    
    # 2. LMSYS (30k)
    if len(all_conversations) < args.max_total:
        convs = download_lmsys_chat(max_samples=30000)
        all_conversations.extend(convs)
        print(f"\n📊 Total: {len(all_conversations):,}")
    
    # 3. OpenHermes (20k)
    if len(all_conversations) < args.max_total:
        convs = download_openhermes(max_samples=20000)
        all_conversations.extend(convs)
        print(f"\n📊 Total: {len(all_conversations):,}")
    
    # 4. Vigogne (10k)
    if len(all_conversations) < args.max_total:
        convs = download_vigogne(max_samples=10000)
        all_conversations.extend(convs)
        print(f"\n📊 Total: {len(all_conversations):,}")
    
    # Deduplicate
    print(f"\n🧹 Déduplication...")
    unique_convs = list(set(all_conversations))
    print(f"   ✅ {len(unique_convs):,} conversations uniques")
    
    final_convs = unique_convs[:args.max_total]
    
    # Save
    output_file = args.output_dir / "conversations_extra_fr.txt"
    output_file.write_text("".join(final_convs), encoding="utf-8")
    
    total_chars = sum(len(c) for c in final_convs)
    total_tokens = total_chars // 4
    
    print(f"\n" + "=" * 70)
    print(f"✅ Sauvegardé: {output_file}")
    print(f"\n📊 Statistiques finales:")
    print(f"   - Conversations: {len(final_convs):,}")
    print(f"   - Caractères: {total_chars:,}")
    print(f"   - Tokens estimés: {total_tokens:,}")
    print(f"   - Taille fichier: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 70)
    
    # Metadata
    meta = {
        "conversations": len(final_convs),
        "total_chars": total_chars,
        "estimated_tokens": total_tokens,
        "output_path": str(output_file),
        "sources": ["WildChat", "LMSYS-Chat-1M", "OpenHermes-2.5", "Vigogne"],
    }
    meta_file = args.output_dir / "meta.json"
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    print(f"\n💡 Prochaine étape: tokeniser avec pretokenize_corpus.py")


if __name__ == "__main__":
    main()
