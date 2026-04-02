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
from tsr.metrics.tsr_metrics import tokens_to_html


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


def preprocess_image(
    image_path: str,
    image_size=(384, 512),
    pad_frac: float = 0.2,
):
    """
    Load image, pad canvas by ``pad_frac`` (e.g. 0.15 → 1.15× width/height, centered on white),
    resize to ``image_size``, normalize for the encoder.

    Returns:
        tensor (1,C,H,W), original (width, height), padding info (pad_left, pad_top, padded_w, padded_h)
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    orig_size = (w, h)

    new_w = max(1, int(round(w * (1.0 + pad_frac))))
    new_h = max(1, int(round(h * (1.0 + pad_frac))))
    pad_left = (new_w - w) // 2
    pad_top = (new_h - h) // 2

    padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    padded.paste(image, (pad_left, pad_top))

    resized = padded.resize(image_size, Image.BILINEAR)
    image_array = np.array(resized).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    pad_info = (pad_left, pad_top, new_w, new_h)
    return image_tensor.unsqueeze(0), orig_size, pad_info


def trim_table_bboxes_to_original(
    table: dict,
    pad_left: int,
    pad_top: int,
    orig_w: int,
    orig_h: int,
) -> None:
    """Map cell bboxes from padded-canvas coordinates to the original image; update table size fields."""
    table["image_width"] = orig_w
    table["image_height"] = orig_h
    for cell in table.get("cells", []):
        x0, y0, x1, y1 = cell["bbox"]
        cell["bbox"] = [
            max(0.0, min(float(orig_w), x0 - pad_left)),
            max(0.0, min(float(orig_h), y0 - pad_top)),
            max(0.0, min(float(orig_w), x1 - pad_left)),
            max(0.0, min(float(orig_h), y1 - pad_top)),
        ]


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


def save_table_visualization(image_path: str, table: dict, output_path: Path) -> None:
    """Draw predicted cell boxes and labels on the original image and save."""
    from PIL import ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font_small = ImageFont.load_default()

    for cell in table.get("cells", []):
        bbox = cell["bbox"]
        xmin, ymin, xmax, ymax = bbox
        outline = (255, 0, 0) if cell.get("is_header") else (0, 255, 0)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=outline, width=2)
        content = cell.get("content") or ""
        if content:
            text = content[:30] + ("..." if len(content) > 30 else "")
            try:
                tb = draw.textbbox((xmin + 2, ymin + 2), text, font=font_small)
                draw.rectangle(tb, fill=(255, 255, 255, 200))
            except Exception:
                pass
            draw.text((xmin + 2, ymin + 2), text, fill=(0, 0, 0), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


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
    parser.add_argument(
        "--visualize",
        type=str,
        nargs="?",
        const="__auto__",
        default=None,
        help="Save bbox overlay on the input image; optional path (default: <output-stem>.viz.png if --output set, else <image-stem>_viz.png)",
    )
    
    args = parser.parse_args()
    
    # Load checkpoint
    model, vocab, config = load_checkpoint(args.checkpoint, args.device)
    
    # Get image size from config
    image_size = tuple(config.get("image_size", (512, 640)))
    
    # Preprocess image
    print(f"\nLoading image from {args.image}...")
    image_tensor, (orig_width, orig_height), (pad_left, pad_top, padded_w, padded_h) = preprocess_image(
        args.image, image_size
    )
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
    
    # Parse to table (grid → padded canvas px); then map bboxes back to original image
    serializer = SequenceSerializer()
    table = parse_sequence_to_table(tokens, serializer, padded_w, padded_h)
    trim_table_bboxes_to_original(table, pad_left, pad_top, orig_width, orig_height)
    
    print(f"\nParsed table with {len(table['cells'])} cells")
    
    markdown_html = tokens_to_html(tokens, vocab)
    
    # Save output
    if args.output:
        out_path = Path(args.output)
        with open(out_path, 'w') as f:
            json.dump(table, f, indent=2)
        md_html_path = out_path.with_name(out_path.stem + ".markdown.html")
        with open(md_html_path, 'w', encoding='utf-8') as f:
            f.write(markdown_html)
        print(f"\nOutput saved to {out_path}")
        print(f"Markdown HTML saved to {md_html_path}")
    else:
        print("\nTable structure:")
        print(json.dumps(table, indent=2))
        print("\nMarkdown HTML:\n```html")
        print(markdown_html)
        print("```")
    
    if args.visualize is not None:
        if args.visualize == "__auto__":
            if args.output:
                viz_path = Path(args.output).with_name(Path(args.output).stem + ".viz.png")
            else:
                ip = Path(args.image)
                viz_path = ip.with_name(ip.stem + "_viz.png")
        else:
            viz_path = Path(args.visualize)
        save_table_visualization(args.image, table, viz_path)
        print(f"\nVisualization saved to {viz_path}")


if __name__ == "__main__":
    main()


