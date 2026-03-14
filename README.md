# Table and Document Recognition System

A high-performance table recognition system using an End-to-End Multi-Task Learning (MTL) architecture that transitions from an "all-token" generative foundation to a precision-optimized hybrid model.

## Features

### Phase 1: Foundational "All-Token" E2E Implementation
- **Unified Sequence Paradigm**: Serializes document images into autoregressive sequences `y={c,b,t,<Sep>}` where:
  - `c` = cell content (character-level tokens)
  - `b` = bounding box tokens (discrete spatial coordinates)
  - `t` = structural HTML tags (`<table>`, `<tr>`, `<td>`, etc.)
- **Coordinate Discretization**: Scales all coordinates to a fixed 1024 × 1280 grid
- **Token Quartet**: Represents cell bounding boxes as discrete spatial tokens: `{<Xmin>, <Ymin>, <Xmax>, <Ymax>}`
- **Right-Shifted Tokens**: Synchronized training via right-shifted target tokens
- **Unified Cross-Entropy Loss**: Treats spatial tokens and text characters with equal priority

### Phase 2: Precision and Efficiency Improvements

#### Initiative A: Spatial Precision (Hybrid Regression)
- **Auxiliary Regression Head**: Linear layer with Sigmoid activation for continuous normalized coordinates (x, y, w, h)
- **L1 + IoU Loss**: Optimizes spatial heads using L1 loss for coordinate distance and IoU loss for overlap precision
- **Column Consistency Loss**: Minimizes prediction variance across tokens in the same logical column

#### Initiative B: Accelerating Inference
- **DREAM Parallel Decoding**: Uses N element queries in a feature aggregator to generate sequences for multiple elements simultaneously
- **Multi-Token Inference**: Decoder predicts n tokens simultaneously via additional linear layers
- **Multi-Cell Parallelism**: For tables, concatenates all cell contents and predicts next-tokens for every cell simultaneously

#### Initiative C: Architectural Optimization
- **ConvStem**: Convolutional stem using stride-2, 3x3 convolutions to balance receptive field and sequence length
- **NoPE (No Positional Encoding)**: Removes explicit 1D positional embeddings; relies on causal attention mask for relative positioning
- **Token Compression**: Pixel-shuffle and compression to reduce vision token length by up to 20%

#### Initiative D: Structural Refinement
- **HTML Refiner**: Non-causal attention module between structure and content decoders allowing cells to share dense structural features
- **B-I-IB Tagging**: Beginning-Inside-InsideBelow tagging for semantic continuity (ready for implementation)
- **Global Context Attention (GCAttention)**: Multi-aspect global context attention after residual blocks in encoder

## Architecture

- **Encoder**: Swin-B, ResNet-31, or ConvStem backbone with optional GCAttention
- **Decoder**: Transformer decoder with NoPE, HTML refiner, and optional parallel decoding
- **Loss Function**: `L = λ₁ CE_struc + λ₂ CE_cont + λ₃ L1_bbox + λ₄ IoU + λ₅ Consistency`

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

The system expects data in JSON format:

```json
{
  "image_path": "path/to/image.jpg",
  "table": {
    "cells": [
      {
        "content": "Cell text",
        "bbox": [xmin, ymin, xmax, ymax],
        "is_header": false
      }
    ],
    "image_width": 1024,
    "image_height": 1280
  }
}
```

## Training

1. Prepare your data in the expected JSON format
2. Update `config.yaml` with your data paths and hyperparameters
3. Run training:

```bash
python train.py --config config.yaml
```

To resume from a checkpoint:

```bash
python train.py --config config.yaml --resume checkpoints/latest.pth
```

## Configuration

Key configuration options in `config.yaml`:

- `encoder_backbone`: Choose between "swin_b", "resnet31", or "convstem"
- `use_hybrid_regression`: Enable hybrid regression heads for continuous coordinates
- `use_parallel_decoder`: Enable DREAM parallel decoding for faster inference
- `token_compression`: Set to 0.8 for 20% token reduction
- Loss weights: Adjust `lambda_*` values to balance different loss components

## Model Components

### Core Modules

- `tsr.data.serialization`: Sequence serialization and coordinate discretization
- `tsr.data.dataset`: Dataset classes for loading table data
- `tsr.models.encoder`: Visual encoders (Swin-B, ResNet-31, ConvStem)
- `tsr.models.decoder`: Transformer decoder with NoPE and parallel decoding
- `tsr.models.model`: Main E2E MTL model
- `tsr.losses.losses`: Multi-task loss functions
- `tsr.training.trainer`: Training utilities

## References

- Architecture: Swin-B or ResNet-31 Encoder + Transformer Decoder (NoPE)
- Trigger Mechanism: MTL-TabNet
- Parallelization: DREAM Parallel Decoder
- Loss Weights: `L = λ₁ CE_struc + λ₂ CE_cont + λ₃ L1_bbox + λ₄ IoU + λ₅ Consistency`

## License

[Add your license here]


