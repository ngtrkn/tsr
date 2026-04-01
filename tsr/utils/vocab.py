"""
Vocabulary utilities for saving and loading vocabularies
"""
import json
from pathlib import Path
from typing import Dict


def save_vocab(vocab: Dict[str, int], path: str):
    """Save vocabulary to JSON file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Vocabulary saved to {path}")


def load_vocab(path: str) -> Dict[str, int]:
    """Load vocabulary from JSON file"""
    with open(path, 'r') as f:
        vocab = json.load(f)
    
    # Convert string keys to proper types if needed
    return {str(k): int(v) for k, v in vocab.items()}


def save_vocab_txt(vocab: Dict[str, int], path: str):
    """Save vocabulary to text file (one token per line, tab-separated id).
    
    Tokens are sorted by id so line order matches the id assignment.
    Special characters (newline, tab, carriage return) are escaped.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])

    with open(path, 'w', encoding='utf-8') as f:
        for token, token_id in sorted_tokens:
            escaped = token.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
            f.write(f"{escaped}\t{token_id}\n")

    print(f"Vocabulary ({len(vocab)} tokens) saved to {path}")


def load_vocab_txt(path: str) -> Dict[str, int]:
    """Load vocabulary from text file produced by save_vocab_txt."""
    vocab: Dict[str, int] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            # Split on last tab to handle tokens that may contain tabs (escaped)
            parts = line.rsplit('\t', 1)
            if len(parts) != 2:
                continue
            escaped_token, token_id_str = parts
            token = escaped_token.replace('\\r', '\r').replace('\\t', '\t').replace('\\n', '\n').replace('\\\\', '\\')
            vocab[token] = int(token_id_str)
    return vocab


def load_vocab_auto(path: str) -> Dict[str, int]:
    """Load vocabulary from either JSON or TXT format based on file extension."""
    path = Path(path)
    if path.suffix == '.json':
        return load_vocab(str(path))
    else:
        return load_vocab_txt(str(path))


def get_id_to_token(vocab: Dict[str, int]) -> Dict[int, str]:
    """Convert token->id mapping to id->token mapping"""
    return {v: k for k, v in vocab.items()}


