# TSR: Table Structure Recognition

End-to-end multi-task learning system for table structure recognition from document images. Serializes tables into unified autoregressive sequences and decodes structure, content, and spatial coordinates jointly.

## Installation

```bash
git clone <repo-url> && cd tsr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.0+, CUDA-capable GPU (12 GB+ VRAM recommended).

## Quick Start

### 1. Prepare Data

**From PubTables-1M** (XML + words):

```bash
python create_dummy_dataset.py \
    --xml_dir /path/to/pub1m/xml \
    --words_dir /path/to/pub1m/words \
    --output_dir ./dummy_dataset \
    --num_train 500 --num_val 100 --num_test 100
```

This produces `dataset_list.json` files per split, each containing `[image_path, label_path]` pairs.

**Label JSON format** (one per table):

```json
{
  "image_path": "path/to/image.jpg",
  "table": {
    "cells": [
      {"content": "Cell text", "bbox": [xmin, ymin, xmax, ymax], "is_header": false}
    ],
    "image_width": 1024,
    "image_height": 1280
  }
}
```

### 2. Train

**Basic training** via `config.yaml`:

```bash
python train.py --config config.yaml
```

**Experiment scripts** (recommended — include mixed precision, gradient accumulation, metrics):

```bash
# Foundation baseline
python experiments/exp_foundation_basic.py \
    --data_path ./dummy_dataset/train/dataset_list.json \
    --val_path ./dummy_dataset/val/dataset_list.json \
    --batch_size 4 --num_epochs 10

# With hybrid regression
python experiments/exp_improvement_hybrid_regression.py \
    --data_path ./dummy_dataset/train/dataset_list.json \
    --val_path ./dummy_dataset/val/dataset_list.json \
    --batch_size 4 --num_epochs 10
```

Available experiments: `exp_foundation_basic`, `exp_improvement_hybrid_regression`, `exp_improvement_html_refiner`, `exp_improvement_gc_attention`, `exp_improvement_token_compression`, `exp_improvement_all_combined`.

### 3. Resume / Validate

```bash
# Resume training from checkpoint
python experiments/exp_foundation_basic.py \
    --data_path ... --val_path ... --resume experiment_results/checkpoints/Foundation_Basic/latest.pth

# Validation only
python experiments/exp_foundation_basic.py \
    --val_path ... --validate_only --resume .../best.pth
```

### 4. Export & Reuse Vocabulary

```bash
# Export vocab from a trained checkpoint
python export_vocab.py \
    --checkpoint experiment_results/checkpoints/Foundation_Basic/epoch_10.pth \
    --output vocab.txt

# Use exported vocab for new dataset / experiment
python experiments/exp_foundation_basic.py \
    --data_path ... --val_path ... --vocab vocab.txt
```

### 5. Inference

```bash
python example_inference.py \
    --checkpoint experiment_results/checkpoints/Foundation_Basic/best.pth \
    --image path/to/table_image.jpg \
    --output result.json
```

### 6. Compare Experiments

```bash
python experiments/compare_results.py --results_dir ./experiment_results
```

## Project Structure

```
tsr/
├── tsr/                        # Core package
│   ├── data/
│   │   ├── serialization.py    # Sequence serialization & coordinate discretization
│   │   ├── dataset.py          # TableDataset (simplified & legacy formats)
│   │   └── pub1m_parser.py     # PubTables-1M XML/words parser
│   ├── models/
│   │   ├── encoder.py          # Swin-B / ResNet-31 / ConvStem + GCAttention
│   │   ├── decoder.py          # Transformer decoder (NoPE, HTML refiner, parallel)
│   │   └── model.py            # End-to-end model
│   ├── losses/
│   │   └── losses.py           # Multi-task loss (CE + L1 + IoU + consistency)
│   ├── training/
│   │   └── trainer.py          # Training loop
│   ├── metrics/
│   │   └── tsr_metrics.py      # TEDS, token/structure/content accuracy
│   └── utils/
│       └── vocab.py            # Vocabulary save/load (txt & json)
├── experiments/                # Per-experiment scripts & framework
├── train.py                    # Config-driven training entry point
├── export_vocab.py             # Export vocab from checkpoint
├── create_dummy_dataset.py     # Dataset creation from PubTables-1M
├── example_inference.py        # Inference example
├── config.yaml                 # Default training config
└── requirements.txt
```

## Configuration

Key options in `config.yaml` and experiment scripts:

| Option | Description | Default |
|---|---|---|
| `encoder_backbone` | `"swin_b"`, `"resnet31"`, `"convstem"` | `resnet31` |
| `embed_dim` | Embedding dimension | 384–768 |
| `use_hybrid_regression` | Auxiliary bbox regression head | `false` |
| `use_html_refiner` | Non-causal structural refinement | `false` |
| `use_gc_attention` | Global context attention in encoder | `false` |
| `token_compression` | Vision token reduction ratio (e.g. `0.8`) | `null` |
| `use_mixed_precision` | FP16 training | `true` |
| `gradient_checkpointing` | Trade compute for memory | `true` |

## Memory Optimization

For 12 GB VRAM GPUs, experiments default to reduced model dimensions, FP16, gradient checkpointing, and gradient accumulation. If OOM persists:

- Reduce `batch_size` to 1 with higher `gradient_accumulation_steps`
- Reduce `image_size` (e.g. `(384, 512)`)
- Use `encoder_backbone="convstem"` (lightest encoder)
- Lower `embed_dim` / `decoder_layers`

See `docs/technical_report.md` for detailed configuration profiles.

## Documentation

- **This README**: Installation, quick-start use cases, project layout
- **[Technical Report](docs/technical_report.md)**: Problem statement, related work, architecture design, experiment setup, results discussion, and memory optimization details

## License

[Add your license here]
