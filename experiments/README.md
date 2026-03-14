# Experiments: Foundation to Improvement Phase

Individual experiment scripts to validate improvements from Foundation Phase to Improvement Phase.

## Experiment Scripts

Each experiment is now a standalone script that can be run independently:

### Phase 1: Foundation

#### `exp_foundation_basic.py`
**Foundation Basic** - Baseline implementation
- Unified sequence paradigm
- Right-shifted tokens
- Unified Cross-Entropy loss

```bash
python experiments/exp_foundation_basic.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10
```

### Phase 2: Improvements

#### `exp_improvement_hybrid_regression.py`
**Hybrid Regression** (Initiative A: Spatial Precision)
- Adds hybrid regression heads
- L1 + IoU loss
- Column consistency loss

```bash
python experiments/exp_improvement_hybrid_regression.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10
```

#### `exp_improvement_html_refiner.py`
**HTML Refiner** (Initiative D: Structural Refinement)
- Non-causal attention between structure and content
- Allows cells to share structural features

```bash
python experiments/exp_improvement_html_refiner.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10
```

#### `exp_improvement_gc_attention.py`
**Global Context Attention** (Initiative C: Architectural Optimization)
- Multi-aspect global context attention
- Models global relationships

```bash
python experiments/exp_improvement_gc_attention.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10
```

#### `exp_improvement_token_compression.py`
**Token Compression** (Initiative C: Architectural Optimization)
- 20% reduction in vision token length
- Expected: ~20% faster inference

```bash
python experiments/exp_improvement_token_compression.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10 \
    --compression_ratio 0.8
```

#### `exp_improvement_all_combined.py`
**All Improvements Combined**
- All improvements working together
- Cumulative benefits

```bash
python experiments/exp_improvement_all_combined.py \
    --data_path /path/to/train/data \
    --val_path /path/to/val/data \
    --output_dir ./experiment_results \
    --batch_size 4 \
    --num_epochs 10 \
    --compression_ratio 0.8
```

## Comparing Results

After running experiments, compare results:

```bash
python experiments/compare_results.py \
    --results_dir ./experiment_results \
    --output ./experiment_results/comparison_report.md
```

This generates a comparison report showing:
- Side-by-side metrics
- Improvement percentages
- Best performing configurations

## Running All Experiments

You can run all experiments sequentially:

```bash
# Foundation
python experiments/exp_foundation_basic.py --data_path /path/to/data --val_path /path/to/val

# Improvements
python experiments/exp_improvement_hybrid_regression.py --data_path /path/to/data --val_path /path/to/val
python experiments/exp_improvement_html_refiner.py --data_path /path/to/data --val_path /path/to/val
python experiments/exp_improvement_gc_attention.py --data_path /path/to/data --val_path /path/to/val
python experiments/exp_improvement_token_compression.py --data_path /path/to/data --val_path /path/to/val
python experiments/exp_improvement_all_combined.py --data_path /path/to/data --val_path /path/to/val

# Compare
python experiments/compare_results.py --results_dir ./experiment_results
```

## Output

Each experiment generates:
- `{experiment_name}_results.json`: Detailed results including losses, times, model stats

The comparison script generates:
- `comparison_report.md`: Human-readable comparison report

## Common Arguments

All experiment scripts support:
- `--data_path`: Path to training data (required)
- `--val_path`: Path to validation data (optional)
- `--output_dir`: Output directory (default: `./experiment_results`)
- `--device`: Device to use (default: `cuda`)
- `--batch_size`: Batch size (default: 4)
- `--num_epochs`: Number of epochs (default: 10)
- `--learning_rate`: Learning rate (default: 1e-4)

## Expected Improvements

| Experiment | Expected Benefit |
|------------|------------------|
| Hybrid Regression | Better spatial precision (lower bbox loss) |
| HTML Refiner | Better structure understanding (lower structure loss) |
| GCAttention | Better global context (lower overall loss) |
| Token Compression | ~20% faster inference |
| All Combined | Cumulative benefits from all improvements |

## Notes

- Each experiment is independent and can be run separately
- Results are saved individually for each experiment
- Use `compare_results.py` to generate unified comparison
- Experiments use ResNet31 backbone for variable input sizes
- All experiments share the same vocabulary from training data
