# Checkpoint Usage Guide

## Overview

Model checkpoints are automatically saved during training experiments. Each checkpoint contains:
- Model state dict (weights)
- Optimizer state dict
- Training configuration
- Vocabulary
- Training/validation losses
- Epoch number

## Checkpoint Locations

Checkpoints are saved in:
```
experiment_results/
  checkpoints/
    {experiment_name}/
      best.pth          # Best model (lowest validation loss)
      latest.pth        # Most recent checkpoint
      epoch_{N}.pth     # Checkpoint at epoch N (every 5 epochs + final)
```

## Using Checkpoints for Inference

### Method 1: Using example_inference.py (Recommended)

```bash
python example_inference.py \
    --checkpoint experiment_results/checkpoints/Foundation_Basic/best.pth \
    --image path/to/table_image.jpg \
    --output output_table.json \
    --device cuda
```

### Method 2: Programmatic Usage

```python
from example_inference import load_checkpoint, preprocess_image, ids_to_tokens, parse_sequence_to_table
from tsr.data.serialization import SequenceSerializer
import torch

# Load checkpoint
model, vocab, config = load_checkpoint("checkpoint.pth", device="cuda")

# Preprocess image
image_tensor, (orig_width, orig_height) = preprocess_image(
    "image.jpg", 
    image_size=tuple(config.get("image_size", (384, 512)))
)

# Generate
with torch.no_grad():
    generated_ids = model.generate(image_tensor.to("cuda"), max_length=512)

# Convert to tokens
tokens = ids_to_tokens(generated_ids[0], vocab)

# Parse to table
serializer = SequenceSerializer()
table = parse_sequence_to_table(tokens, serializer, orig_width, orig_height)
```

## Checkpoint Contents

Each checkpoint (.pth file) contains:

```python
{
    "epoch": int,                    # Training epoch
    "model_state_dict": dict,        # Model weights
    "optimizer_state_dict": dict,    # Optimizer state
    "config": dict,                  # ExperimentConfig as dict
    "train_loss": float,             # Training loss at this epoch
    "val_loss": float,               # Validation loss (if available)
    "vocab": dict,                   # Vocabulary {token: id}
}
```

## Loading Checkpoints

### For Inference

```python
from tsr.models.model import TableRecognitionModel
import torch

checkpoint = torch.load("checkpoint.pth", map_location="cuda")
config = checkpoint["config"]
vocab = checkpoint["vocab"]

# Create model with saved config
model = TableRecognitionModel(
    vocab_size=len(vocab),
    encoder_backbone=config["encoder_backbone"],
    embed_dim=config["embed_dim"],
    decoder_layers=config["decoder_layers"],
    decoder_heads=config["decoder_heads"],
    ffn_dim=config["ffn_dim"],
    # ... other config parameters
)

# Load weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### For Resuming Training

```python
# Load checkpoint
checkpoint = torch.load("checkpoint.pth")

# Create model and optimizer (same as training)
model = create_model(config, vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Load states
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Resume from epoch
start_epoch = checkpoint["epoch"] + 1
```

## Example: Complete Inference Pipeline

```python
#!/usr/bin/env python3
import torch
from example_inference import (
    load_checkpoint, preprocess_image, ids_to_tokens, 
    parse_sequence_to_table
)
from tsr.data.serialization import SequenceSerializer

# 1. Load checkpoint
model, vocab, config = load_checkpoint(
    "experiment_results/checkpoints/Foundation_Basic/best.pth",
    device="cuda"
)

# 2. Preprocess image
image_tensor, (orig_width, orig_height) = preprocess_image(
    "test_image.jpg",
    image_size=tuple(config.get("image_size", (384, 512)))
)

# 3. Generate table structure
with torch.no_grad():
    generated_ids = model.generate(
        image_tensor.to("cuda"),
        max_length=512,
        temperature=1.0
    )

# 4. Convert to tokens
tokens = ids_to_tokens(generated_ids[0], vocab)

# 5. Parse to table structure
serializer = SequenceSerializer()
table = parse_sequence_to_table(
    tokens, serializer, orig_width, orig_height
)

# 6. Use table structure
print(f"Found {len(table['cells'])} cells")
for cell in table['cells']:
    print(f"  {cell['content']} at {cell['bbox']}")
```

## Best Practices

1. **Use best.pth for inference**: This contains the model with lowest validation loss
2. **Save vocab separately**: Vocabulary is included in checkpoint, but you can also save it separately for convenience
3. **Check config before loading**: Make sure the model architecture matches your checkpoint
4. **Use same image preprocessing**: Use the same image_size as training
5. **Set model to eval mode**: Always call `model.eval()` before inference

## Troubleshooting

### Error: "Unexpected key(s) in state_dict"
- Make sure model architecture matches checkpoint config
- Check that all config parameters are correctly loaded

### Error: "Missing key(s) in state_dict"
- Model architecture may have changed
- Try loading with `strict=False` (not recommended)

### Error: Vocabulary mismatch
- Make sure you're using the vocabulary from the checkpoint
- Check that token IDs match between training and inference

## File Sizes

Typical checkpoint sizes:
- Model only: ~50-100 MB (FP32) or ~25-50 MB (FP16)
- With optimizer state: ~100-200 MB
- Full checkpoint (with vocab): ~100-250 MB


