#!/usr/bin/env python3
"""
Wrapper pour rendre SentencePiece compatible avec l'interface tokenizers attendue.
"""
import sentencepiece as spm
from typing import List, Union

class SentencePieceWrapper:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.bos_token_id = self.sp.bos_id() if self.sp.bos_id() != -1 else 0
        self.eos_token_id = self.sp.eos_id() if self.sp.eos_id() != -1 else 1
        self.pad_token_id = self.sp.pad_id() if self.sp.pad_id() != -1 else 0

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids."""
        if add_special_tokens and self.bos_token_id != -1:
            return [self.bos_token_id] + self.sp.encode_as_ids(text)
        return self.sp.encode_as_ids(text)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text."""
        if skip_special_tokens:
            # Remove BOS/EOS tokens if present
            filtered_ids = [id for id in ids if id not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]]
            return self.sp.decode_ids(filtered_ids)
        return self.sp.decode_ids(ids)

    def token_to_id(self, token: str) -> int:
        """Convert token to id."""
        return self.sp.piece_to_id(token)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.sp.get_piece_size()

# Factory function to create tokenizer
def create_tokenizer(tokenizer_path: str):
    """Create appropriate tokenizer based on file extension."""
    if tokenizer_path.endswith('.model'):
        return SentencePieceWrapper(tokenizer_path)
    else:
        # Fallback to tokenizers library
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tokenizer_path)