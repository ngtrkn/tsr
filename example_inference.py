#!/usr/bin/env python3
"""
Example inference script using saved checkpoint
"""
import argparse
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import json
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tsr.models.model import TableRecognitionModel
from tsr.data.serialization import SequenceSerializer, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model checkpoint with config and vocabulary"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config_dict = checkpoint["config"]
    
    # Create model with saved config
    vocab = checkpoint.get("vocab", {})
    vocab_size = len(vocab) if vocab else config_dict.get("vocab_size", 10000)
    
    model = TableRecognitionModel(
        vocab_size=vocab_size,
        encoder_backbone=config_dict.get("encoder_backbone", "convstem"),
        embed_dim=config_dict.get("embed_dim", 384),
        decoder_layers=config_dict.get("decoder_layers", 3),
        decoder_heads=config_dict.get("decoder_heads", 6),
        ffn_dim=config_dict.get("ffn_dim", 1536),
        dropout=config_dict.get("dropout", 0.1),
        use_html_refiner=config_dict.get("use_html_refiner", False),
        use_gc_attention=config_dict.get("use_gc_attention", False),
        token_compression=config_dict.get("token_compression", None),
        use_hybrid_regression=config_dict.get("use_hybrid_regression", False),
        use_parallel_decoder=config_dict.get("use_parallel_decoder", False),
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {config_dict.get('name', 'unknown')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    if checkpoint.get('val_loss') is not None:
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model, vocab, config_dict


def preprocess_image(image_path: str, image_size=(384, 512)):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (width, height)
    
    # Resize
    image = image.resize(image_size, Image.BILINEAR)
    
    # Convert to tensor and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # (C, H, W)
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0), orig_size  # (1, C, H, W)


def ids_to_tokens(token_ids: torch.Tensor, vocab: dict):
    """Convert token IDs to tokens using vocabulary"""
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = []
    for id in token_ids.cpu().numpy():
        token = id_to_token.get(int(id), PAD_TOKEN)
        if token == EOS_TOKEN:
            break
        tokens.append(token)
    return tokens


def parse_sequence_to_table(tokens: list, serializer: SequenceSerializer, 
                           image_width: int, image_height: int):
    """Parse generated token sequence back to table structure"""
    from tsr.data.serialization import (
        TABLE_START, TABLE_END, ROW_START, ROW_END,
        CELL_START, CELL_END, HEADER_START, HEADER_END,
        XMIN_TOKEN, YMIN_TOKEN, XMAX_TOKEN, YMAX_TOKEN,
        SEP_TOKEN, EOS_TOKEN
    )
    
    table = {
        "cells": [],
        "image_width": image_width,
        "image_height": image_height
    }
    
    i = 0
    current_row = []
    
    while i < len(tokens):
        token = tokens[i]
        
        if token == TABLE_START:
            i += 1
            continue
        
        if token == TABLE_END or token == EOS_TOKEN:
            if current_row:
                table["cells"].extend(current_row)
            break
        
        if token == ROW_START:
            if current_row:
                table["cells"].extend(current_row)
                current_row = []
            i += 1
            continue
        
        if token == ROW_END:
            i += 1
            continue
        
        if token in [CELL_START, HEADER_START]:
            is_header = (token == HEADER_START)
            
            # Extract bbox tokens
            if (i + 4 < len(tokens) and
                tokens[i+1].startswith(XMIN_TOKEN) and
                tokens[i+2].startswith(YMIN_TOKEN) and
                tokens[i+3].startswith(XMAX_TOKEN) and
                tokens[i+4].startswith(YMAX_TOKEN)):
                
                xmin = int(tokens[i+1].replace(XMIN_TOKEN, ""))
                ymin = int(tokens[i+2].replace(YMIN_TOKEN, ""))
                xmax = int(tokens[i+3].replace(XMAX_TOKEN, ""))
                ymax = int(tokens[i+4].replace(YMAX_TOKEN, ""))
                
                # Convert back to continuous coordinates
                xmin_cont = (xmin / serializer.grid_width) * image_width
                ymin_cont = (ymin / serializer.grid_height) * image_height
                xmax_cont = (xmax / serializer.grid_width) * image_width
                ymax_cont = (ymax / serializer.grid_height) * image_height
                
                # Extract content
                content_start = i + 5
                content_end = content_start
                while (content_end < len(tokens) and 
                       tokens[content_end] not in [CELL_END, HEADER_END, SEP_TOKEN, EOS_TOKEN]):
                    content_end += 1
                
                content = "".join(tokens[content_start:content_end])
                
                cell = {
                    "content": content,
                    "bbox": [xmin_cont, ymin_cont, xmax_cont, ymax_cont],
                    "is_header": is_header
                }
                current_row.append(cell)
                
                i = content_end + 1
                continue
        
        i += 1
    
    return table


def main():
    parser = argparse.ArgumentParser(description="Inference example using saved checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    # Load checkpoint
    model, vocab, config = load_checkpoint(args.checkpoint, args.device)
    
    # Get image size from config
    image_size = tuple(config.get("image_size", (384, 512)))
    
    # Preprocess image
    print(f"\nLoading image from {args.image}...")
    image_tensor, (orig_width, orig_height) = preprocess_image(args.image, image_size)
    image_tensor = image_tensor.to(args.device)
    
    # Generate
    print("Generating table structure...")
    with torch.no_grad():
        generated_ids = model.generate(
            image_tensor,
            max_length=args.max_length,
            temperature=args.temperature,
        )
    
    # Convert to tokens
    tokens = ids_to_tokens(generated_ids[0], vocab)
    print(f"\nGenerated {len(tokens)} tokens")
    print(f"First 50 tokens: {tokens[:50]}")
    
    # Parse to table
    serializer = SequenceSerializer()
    table = parse_sequence_to_table(tokens, serializer, orig_width, orig_height)
    
    print(f"\nParsed table with {len(table['cells'])} cells")
    
    # Save output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(table, f, indent=2)
        print(f"\nOutput saved to {args.output}")
    else:
        print("\nTable structure:")
        print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()


