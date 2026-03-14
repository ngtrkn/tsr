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


def get_id_to_token(vocab: Dict[str, int]) -> Dict[int, str]:
    """Convert token->id mapping to id->token mapping"""
    return {v: k for k, v in vocab.items()}


