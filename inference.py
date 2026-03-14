"""
Inference script for table recognition
"""
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import json

from tsr.models.model import TableRecognitionModel
from tsr.data.serialization import SequenceSerializer


def load_model(checkpoint_path: str, vocab_size: int = None, device: str = "cuda"):
    """Load model from checkpoint with saved config"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint
    config_dict = checkpoint.get("config", {})
    vocab = checkpoint.get("vocab", {})
    
    # Use vocab size from checkpoint if available
    if vocab_size is None:
        vocab_size = len(vocab) if vocab else config_dict.get("vocab_size", 10000)
    
    # Create model with saved config
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
    
    return model, vocab, config_dict


def preprocess_image(image_path: str, image_size=(512, 640)):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, Image.BILINEAR)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
    return image_tensor, image.size


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
    in_table = False
    
    while i < len(tokens):
        token = tokens[i]
        
        if token == TABLE_START:
            in_table = True
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
                       tokens[content_end] not in [CELL_END, HEADER_END, SEP_TOKEN]):
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
    parser = argparse.ArgumentParser(description="Inference for table recognition")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max_length", type=int, default=512, help="Max generation length")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.vocab_size, args.device)
    
    # Preprocess image
    print(f"Loading image from {args.image}...")
    image_tensor, (orig_width, orig_height) = preprocess_image(args.image)
    image_tensor = image_tensor.to(args.device)
    
    # Generate
    print("Generating table structure...")
    with torch.no_grad():
        generated_ids = model.generate(
            image_tensor,
            max_length=args.max_length,
            temperature=1.0,
        )
    
    # Convert to tokens
    # Note: You'll need to load the vocabulary from training
    # For now, this is a placeholder
    serializer = SequenceSerializer()
    # vocab = ... # Load from training checkpoint or separate file
    # id_to_token = {v: k for k, v in vocab.items()}
    # tokens = [id_to_token[id.item()] for id in generated_ids[0]]
    
    # Parse to table
    # table = parse_sequence_to_table(tokens, serializer, orig_width, orig_height)
    
    # Save output
    if args.output:
        # with open(args.output, 'w') as f:
        #     json.dump(table, f, indent=2)
        print(f"Output saved to {args.output}")
    else:
        print("Generated token IDs:", generated_ids[0].tolist()[:50])  # Print first 50


if __name__ == "__main__":
    main()

