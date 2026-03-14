# Project Structure

```
tsr/
├── tsr/                          # Main package
│   ├── __init__.py
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── serialization.py      # Sequence serialization & coordinate discretization
│   │   └── dataset.py            # Dataset classes
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── encoder.py            # Visual encoders (Swin-B, ResNet-31, ConvStem, GCAttention)
│   │   ├── decoder.py            # Transformer decoder (NoPE, HTML Refiner, Parallel Decoder)
│   │   └── model.py              # Main E2E MTL model
│   ├── losses/                   # Loss functions
│   │   ├── __init__.py
│   │   └── losses.py             # Multi-task loss (CE, L1, IoU, Consistency)
│   ├── training/                 # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py            # Training loop and checkpointing
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── vocab.py              # Vocabulary management
├── train.py                      # Training script
├── inference.py                  # Inference script
├── example_usage.py              # Example usage demonstrations
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore rules
```

## Key Components

### Phase 1: Foundational Implementation
- ✅ **Unified Sequence Serialization** (`tsr/data/serialization.py`)
  - Coordinate discretization to 1024×1280 grid
  - Token quartet representation for bounding boxes
  - Sequence format: `y={c,b,t,<Sep>}`

- ✅ **Encoder-Decoder Architecture** (`tsr/models/`)
  - Visual encoder with multiple backbone options
  - Transformer decoder with NoPE
  - Right-shifted token mechanism

- ✅ **Unified Cross-Entropy Loss** (`tsr/losses/losses.py`)
  - Treats all tokens (spatial + text) equally

### Phase 2: Precision and Efficiency

- ✅ **Hybrid Regression** (`tsr/models/model.py`)
  - Continuous coordinate prediction (x, y, w, h)
  - L1 + IoU loss for spatial precision
  - Column consistency loss

- ✅ **Parallel Decoding** (`tsr/models/decoder.py`)
  - DREAM-style parallel decoder
  - Multi-token inference support

- ✅ **Architectural Optimizations**
  - ConvStem backbone option
  - NoPE (No Positional Encoding)
  - Token compression support
  - GCAttention for global context

- ✅ **Structural Refinement**
  - HTML Refiner (non-causal attention)
  - Ready for B-I-IB tagging extension

## Usage Flow

1. **Data Preparation**: Format data as JSON with table structure
2. **Training**: Run `python train.py --config config.yaml`
3. **Inference**: Run `python inference.py --checkpoint <path> --image <path>`

## Configuration

All hyperparameters and model settings are in `config.yaml`:
- Model architecture choices
- Training parameters
- Loss weights (λ₁ through λ₅)


