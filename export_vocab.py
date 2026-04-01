#!/usr/bin/env python3
"""
Export vocabulary from a model checkpoint to a text file.

Usage:
    python export_vocab.py --checkpoint path/to/epoch_10.pth --output vocab.txt
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from tsr.utils.vocab import save_vocab_txt, save_vocab


def main():
    parser = argparse.ArgumentParser(description="Export vocabulary from a checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vocab.txt",
        help="Output path (default: vocab.txt). Use .json extension for JSON format.",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    vocab = checkpoint.get("vocab")
    if vocab is None:
        print("Error: checkpoint does not contain a 'vocab' key.")
        print(f"Available keys: {list(checkpoint.keys())}")
        sys.exit(1)

    print(f"Vocabulary size: {len(vocab)} tokens")

    output_path = Path(args.output)
    if output_path.suffix == ".json":
        save_vocab(vocab, str(output_path))
    else:
        save_vocab_txt(vocab, str(output_path))


if __name__ == "__main__":
    main()
