#!/usr/bin/env python3
"""
Télécharge des datasets de conversations en français pour fine-tuning.
Datasets ciblés:
- OpenAssistant conversations (multilingue avec filtre français)
-Daily Dialog adapté en français
"""

import sys
from pathlib import Path
from typing import Optional

def download_conversations(
    output_dir: str = "data_clean/conversations",
    max_conversations: int = 10000,
    max_size_mb: Optional[float] = 100.0
) -> None:
    """
    Télécharge et formate des conversations en français.
    
    Args:
        output_dir: Répertoire de sortie
        max_conversations: Nombre max de conversations
        max_size_mb: Taille maximale en Mo (None = pas de limite)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets not installed. Run: pip install datasets")
        sys.exit(1)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "conversations_fr.txt"
    
    print(f"Téléchargement des conversations françaises...")
    print(f"Output: {output_file}")
    
    collected = 0
    total_bytes = 0
    max_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else float('inf')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Dataset 1: OpenAssistant Conversations (OASST1)
        try:
            print("\n[1/2] Chargement OpenAssistant conversations...")
            ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
            
            conversations = {}
            for item in ds:
                if collected >= max_conversations or total_bytes >= max_bytes:
                    break
                
                # Filtrer uniquement le français
                if item.get('lang') != 'fr':
                    continue
                
                message_id = item.get('message_id')
                parent_id = item.get('parent_id')
                text = item.get('text', '').strip()
                role = item.get('role')  # 'prompter' ou 'assistant'
                
                if not text:
                    continue
                
                # Construire la conversation
                if parent_id is None or parent_id not in conversations:
                    # Nouvelle conversation
                    conversations[message_id] = {
                        'messages': [{'role': role, 'text': text}],
                        'current_id': message_id
                    }
                else:
                    # Continuer une conversation existante
                    parent_conv = conversations[parent_id]
                    parent_conv['messages'].append({'role': role, 'text': text})
                    conversations[message_id] = parent_conv
                    conversations[message_id]['current_id'] = message_id
            
            # Écrire les conversations complètes
            written_convs = set()
            for conv_id, conv_data in conversations.items():
                if conv_id in written_convs:
                    continue
                
                messages = conv_data['messages']
                if len(messages) < 2:  # Besoin d'au moins un échange
                    continue
                
                # Format: User: ... \n Assistant: ... \n User: ... etc.
                conversation_text = ""
                for msg in messages:
                    role_label = "Utilisateur" if msg['role'] == 'prompter' else "Assistant"
                    conversation_text += f"{role_label}: {msg['text']}\n"
                
                conversation_text += "\n"  # Séparateur entre conversations
                
                text_bytes = conversation_text.encode('utf-8')
                if total_bytes + len(text_bytes) > max_bytes:
                    break
                
                f.write(conversation_text)
                total_bytes += len(text_bytes)
                collected += 1
                written_convs.add(conv_id)
                
                if collected % 100 == 0:
                    print(f"  Collecté: {collected} conversations, {total_bytes / 1024 / 1024:.1f} MB")
        
        except Exception as e:
            print(f"Erreur OpenAssistant: {e}")
        
        # Dataset 2: French daily conversations (synthétiques si disponibles)
        if collected < max_conversations and total_bytes < max_bytes:
            try:
                print(f"\n[2/2] Ajout de conversations Daily Dialog (traduites)...")
                # Note: On pourrait utiliser un dataset traduit ou synthétique
                # Pour l'instant, on continue avec OASST1 seulement
                print("  (Skip - utiliser seulement OASST1 pour ce run)")
            except Exception as e:
                print(f"Erreur Daily Dialog: {e}")
    
    print(f"\n✅ Terminé!")
    print(f"   Conversations: {collected}")
    print(f"   Taille: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"   Fichier: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Télécharge des conversations françaises")
    parser.add_argument("--output-dir", default="data_clean/conversations",
                       help="Répertoire de sortie")
    parser.add_argument("--max-conversations", type=int, default=10000,
                       help="Nombre max de conversations")
    parser.add_argument("--max-size-mb", type=float, default=100.0,
                       help="Taille max en MB (None=illimité)")
    
    args = parser.parse_args()
    download_conversations(
        output_dir=args.output_dir,
        max_conversations=args.max_conversations,
        max_size_mb=args.max_size_mb
    )
