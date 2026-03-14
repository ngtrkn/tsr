# Memory Optimization for 12GB VRAM

## Overview

The model architecture has been optimized to train on 12GB VRAM GPUs. Key optimizations include:

1. **Reduced Model Size**
   - `embed_dim`: 768 → 512 (33% reduction)
   - `decoder_layers`: 6 → 4 (33% reduction)
   - `ffn_dim`: 3072 → 2048 (33% reduction)
   - Total parameter reduction: ~40-50%

2. **Gradient Accumulation**
   - Default batch size: 2
   - Gradient accumulation steps: 4
   - Effective batch size: 2 × 4 = 8
   - Maintains training stability while reducing memory usage

3. **Mixed Precision Training (FP16)**
   - Enabled by default
   - Reduces memory usage by ~50%
   - Minimal accuracy impact
   - Automatic loss scaling

4. **Gradient Checkpointing** (Optional)
   - Can be enabled for even more memory savings
   - Trades compute time for memory
   - Reduces memory by ~30-40% at cost of ~20% slower training

## Memory Usage Estimates

### Without Optimizations (Original)
- Model parameters: ~200-300M parameters
- Batch size 4: ~8-10GB VRAM
- Batch size 8: ~16-20GB VRAM (OOM on 12GB)

### With Optimizations (Current)
- Model parameters: ~100-150M parameters
- Batch size 2 + FP16: ~4-6GB VRAM
- Batch size 2 + FP16 + Gradient Checkpointing: ~3-4GB VRAM
- Effective batch size 8 via accumulation: Same training dynamics as batch size 8

## Configuration

All experiment scripts now use memory-efficient defaults:

```python
ExperimentConfig(
    embed_dim=512,  # Reduced from 768
    decoder_layers=4,  # Reduced from 6
    ffn_dim=2048,  # Reduced from 3072
    batch_size=2,  # Reduced from 4
    gradient_accumulation_steps=4,  # Effective batch = 8
    use_mixed_precision=True,  # FP16
    gradient_checkpointing=False,  # Can enable if needed
)
```

## Usage

### Standard Training (Recommended)
```bash
python experiments/exp_foundation_basic.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --device cuda
```

### Maximum Memory Savings
If you still encounter OOM errors, enable gradient checkpointing:
```python
# In experiment script, change:
gradient_checkpointing=True
```

Or reduce batch size further:
```bash
python experiments/exp_foundation_basic.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --batch_size 1 \
    --device cuda
```

## Performance Impact

- **Training Speed**: ~10-15% slower due to smaller batch size and gradient accumulation
- **Model Accuracy**: Minimal impact (<2% typically) due to maintained effective batch size
- **Memory Savings**: ~50-60% reduction in peak VRAM usage

## Monitoring Memory Usage

To monitor GPU memory during training:
```bash
watch -n 1 nvidia-smi
```

Or add to training script:
```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

## Further Optimizations (If Needed)

1. **Reduce embed_dim further**: 512 → 384 or 256
2. **Reduce decoder_layers**: 4 → 3
3. **Reduce ffn_dim**: 2048 → 1536 or 1024
4. **Enable token compression**: `token_compression=0.8` (20% reduction)
5. **Use smaller encoder**: `encoder_backbone="resnet31"` (already default)

## Notes

- Mixed precision training requires CUDA and compatible GPU (compute capability >= 7.0)
- Gradient checkpointing works best with PyTorch 1.9+
- Effective batch size should remain constant for fair comparisons between experiments


