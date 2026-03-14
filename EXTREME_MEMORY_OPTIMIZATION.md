# Extreme Memory Optimization for 12GB VRAM

## Overview

For cases where even batch_size=1 is not enough, we've implemented extreme memory optimizations:

### 1. Ultra-Reduced Model Size
- `embed_dim`: 512 → **384** (25% further reduction)
- `decoder_layers`: 4 → **3** (25% further reduction)
- `decoder_heads`: 8 → **6** (25% reduction)
- `ffn_dim`: 2048 → **1536** (25% further reduction)
- Total parameter reduction: ~60-70% from original

### 2. Minimal Encoder
- Changed from `resnet31` to `convstem` (smallest encoder)
- ConvStem is much lighter than ResNet31 or Swin-B

### 3. Gradient Checkpointing Enabled by Default
- Enabled for both encoder and decoder
- Reduces memory by ~30-40%
- Trade-off: ~20% slower training

### 4. Token Compression Enabled
- `token_compression=0.8` (20% token reduction)
- Reduces sequence length memory usage

### 5. Reduced Image Size
- Image size: (512, 640) → **(384, 512)**
- Reduces encoder memory by ~40%

### 6. Batch Size = 1 with Large Accumulation
- `batch_size=1` (minimum possible)
- `gradient_accumulation_steps=8`
- Effective batch size: 1 × 8 = 8

## Memory Usage Estimates

### Extreme Optimizations
- Model parameters: ~20-30M parameters
- Batch size 1 + FP16 + Checkpointing: ~2-3GB VRAM
- With reduced image size: ~1.5-2.5GB VRAM
- Total savings: ~75-80% from original

## Configuration

```python
ExperimentConfig(
    encoder_backbone="convstem",  # Smallest encoder
    embed_dim=384,  # Ultra-reduced
    decoder_layers=3,  # Ultra-reduced
    decoder_heads=6,  # Reduced
    ffn_dim=1536,  # Ultra-reduced
    batch_size=1,  # Minimum
    gradient_accumulation_steps=8,  # Maintain effective batch size
    use_mixed_precision=True,  # FP16
    gradient_checkpointing=True,  # Enabled by default
    token_compression=0.8,  # 20% reduction
    image_size=(384, 512),  # Reduced image size
)
```

## Usage

### Extreme Memory Mode (Default Now)
```bash
python experiments/exp_foundation_basic.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --device cuda
```

### If Still OOM, Try CPU Offloading
If you still encounter OOM, you can:
1. Use CPU for some operations (slower but uses less GPU memory)
2. Further reduce image size to (256, 384)
3. Reduce embed_dim to 256
4. Reduce decoder_layers to 2

## Performance Impact

- **Training Speed**: ~30-40% slower due to:
  - Gradient checkpointing
  - Smaller batch size
  - Reduced image size (less parallelization)
- **Model Accuracy**: May have ~3-5% accuracy impact due to smaller model
- **Memory Savings**: ~75-80% reduction in peak VRAM usage

## Monitoring

Monitor GPU memory:
```bash
watch -n 1 nvidia-smi
```

Or in Python:
```python
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"GPU Memory Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

## Additional Tips

1. **Clear cache between epochs**:
   ```python
   torch.cuda.empty_cache()
   ```

2. **Use gradient clipping** (already enabled):
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **Disable features you don't need**:
   - Set `use_html_refiner=False`
   - Set `use_gc_attention=False`
   - Set `use_hybrid_regression=False` (if not needed)

4. **Use smaller vocabulary** if possible (reduce vocab size)

5. **Consider using DeepSpeed ZeRO** for even more aggressive memory savings (future enhancement)

## Troubleshooting

If you still get OOM errors:

1. Check actual memory usage with `nvidia-smi`
2. Reduce image size further: `image_size=(256, 384)`
3. Reduce embed_dim: `embed_dim=256`
4. Reduce decoder_layers: `decoder_layers=2`
5. Disable token compression: `token_compression=None`
6. Use CPU for validation: `device="cpu"` for validation only

## Model Size Comparison

| Configuration | Parameters | FP32 Size | FP16 Size | Est. VRAM (batch=1) |
|--------------|------------|-----------|-----------|---------------------|
| Original | ~200M | ~800MB | ~400MB | ~8-10GB |
| Standard Optimized | ~42M | ~168MB | ~84MB | ~4-6GB |
| Extreme Optimized | ~20-30M | ~80-120MB | ~40-60MB | ~1.5-2.5GB |


