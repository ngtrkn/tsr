# Experiments Guide: Foundation to Improvement Phase

This guide explains how to run experiments that validate improvements from the Foundation Phase to the Improvement Phase.

## Experiment Structure

### Phase 1: Foundation Experiments

**Foundation_Basic**
- Unified sequence paradigm (y={c,b,t,<Sep>})
- Right-shifted tokens for synchronized training
- Unified Cross-Entropy loss treating all tokens equally
- Baseline for comparison

### Phase 2: Improvement Experiments

Each experiment adds one or more improvements on top of the foundation:

1. **Improvement_HybridRegression** (Initiative A)
   - Adds hybrid regression heads
   - L1 + IoU loss for spatial precision
   - Column consistency loss
   - **Expected**: Better bbox accuracy, lower spatial loss

2. **Improvement_HTMLRefiner** (Initiative D)
   - Adds HTML refiner (non-causal attention)
   - Allows cells to share structural features
   - **Expected**: Better structure understanding, lower structure loss

3. **Improvement_GCAttention** (Initiative C)
   - Adds Global Context Attention
   - Multi-aspect global context modeling
   - **Expected**: Better global understanding, overall loss improvement

4. **Improvement_TokenCompression** (Initiative C)
   - Adds token compression (20% reduction)
   - Reduces vision token length
   - **Expected**: ~20% faster inference, similar accuracy

5. **Improvement_AllCombined**
   - All improvements together
   - **Expected**: Cumulative benefits from all improvements

## Running Experiments

Each experiment is now an independent script for clarity and easier execution.

### Prerequisites

1. Prepare your data in JSON format (see data format in README)
2. Ensure you have training and validation data

### Individual Experiment Scripts

#### Phase 1: Foundation

```bash
# Foundation Basic (baseline)
python experiments/exp_foundation_basic.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10
```

#### Phase 2: Improvements

```bash
# Hybrid Regression
python experiments/exp_improvement_hybrid_regression.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10

# HTML Refiner
python experiments/exp_improvement_html_refiner.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10

# GCAttention
python experiments/exp_improvement_gc_attention.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10

# Token Compression
python experiments/exp_improvement_token_compression.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10 \
    --compression_ratio 0.8

# All Combined
python experiments/exp_improvement_all_combined.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10 \
    --compression_ratio 0.8
```

### Comparing Results

After running experiments, compare all results:

```bash
python experiments/compare_results.py \
    --results_dir ./experiment_results \
    --output ./experiment_results/comparison_report.md
```

### Quick Test

Test the framework with dummy data:

```bash
python experiments/quick_test.py
```

## Understanding Results

### Metrics Tracked

1. **Training Loss**: Lower is better
   - Foundation: Should establish baseline
   - Improvements: Should show reduction

2. **Validation Loss**: Lower is better
   - Measures generalization
   - Should track with training loss

3. **Training Time**: Total time for all epochs
   - Baseline for comparison
   - Some improvements may add overhead

4. **Inference Time**: Average per batch (lower is better)
   - Critical for deployment
   - Token compression should improve this

5. **Model Size**: In MB (lower is better for deployment)
   - Some improvements add parameters
   - Trade-off with accuracy

6. **Parameters**: Total model parameters
   - Indicates model complexity

### Expected Improvements

Based on the implementation guide:

| Improvement | Expected Benefit | Metric |
|------------|------------------|--------|
| Hybrid Regression | Better spatial precision | Lower bbox loss, better IoU |
| HTML Refiner | Better structure understanding | Lower structure loss |
| GCAttention | Better global context | Lower overall loss |
| Token Compression | Faster inference | ~20% reduction in inference time |
| All Combined | Cumulative benefits | Best overall performance |

### Interpreting Results

The comparison report will show:

1. **Loss Improvement**: `(baseline - improved) / baseline * 100`
   - Positive = improvement
   - Negative = degradation (investigate)

2. **Speed Improvement**: `(baseline - improved) / baseline * 100`
   - Positive = faster
   - Negative = slower (may be acceptable for accuracy gain)

3. **Size Change**: `(improved - baseline) / baseline * 100`
   - Positive = larger model
   - Negative = smaller model

## Example Workflow

### Step 1: Run Foundation Baseline

```bash
python experiments/exp_foundation_basic.py \
    --data_path data/train \
    --val_path data/val \
    --batch_size 8 \
    --num_epochs 20
```

### Step 2: Run Individual Improvements

```bash
# Test each improvement separately
python experiments/exp_improvement_hybrid_regression.py \
    --data_path data/train --val_path data/val --batch_size 8 --num_epochs 20

python experiments/exp_improvement_html_refiner.py \
    --data_path data/train --val_path data/val --batch_size 8 --num_epochs 20

python experiments/exp_improvement_gc_attention.py \
    --data_path data/train --val_path data/val --batch_size 8 --num_epochs 20

python experiments/exp_improvement_token_compression.py \
    --data_path data/train --val_path data/val --batch_size 8 --num_epochs 20
```

### Step 3: Run Combined

```bash
python experiments/exp_improvement_all_combined.py \
    --data_path data/train \
    --val_path data/val \
    --batch_size 8 \
    --num_epochs 20 \
    --compression_ratio 0.8
```

### Step 4: Compare Results

```bash
python experiments/compare_results.py \
    --results_dir ./experiment_results
```

Check `experiment_results/comparison_report.md` for:
- Side-by-side comparisons
- Improvement percentages
- Recommendations

## Customizing Experiments

### Adding New Experiments

Edit `experiments/experiment_framework.py`:

```python
def create_custom_experiments():
    return [
        ExperimentConfig(
            name="Custom_Experiment",
            phase="improvement",
            # ... your config
        )
    ]
```

### Modifying Metrics

Edit `ExperimentRunner.run_experiment()` to add custom metrics.

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Use smaller `embed_dim` or `decoder_layers`
- Use CPU: `--device cpu`

### Slow Training

- Reduce `num_epochs` for testing
- Use smaller model (reduce `decoder_layers`, `ffn_dim`)
- Use fewer workers in DataLoader

### Inconsistent Results

- Ensure same random seed
- Use same data splits
- Check data preprocessing consistency

## Best Practices

1. **Start Small**: Run quick tests with few epochs first
2. **Baseline First**: Always establish foundation baseline
3. **Incremental**: Test improvements one at a time
4. **Reproducible**: Use fixed seeds for reproducibility
5. **Document**: Note any deviations from expected results

## Next Steps

After running experiments:

1. Review `comparison_report.md`
2. Identify best configuration for your use case
3. Train final model with best config
4. Evaluate on test set
5. Deploy optimized model

