# Checkpoint Resumption Guide

## Overview

The experiment framework now supports resuming training from saved checkpoints. This allows you to:
- Continue training after interruptions
- Resume from the best model
- Continue from a specific epoch checkpoint

## Usage

### Basic Usage

Resume from the latest checkpoint (relative path):
```bash
python experiments/exp_foundation_basic.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --resume latest.pth
```

Resume from an absolute path:
```bash
python experiments/exp_foundation_basic.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --resume /home/cain/code/github/dev/tsr/experiment_results/checkpoints/Foundation_Basic/latest.pth
```

Resume from the best checkpoint:
```bash
python experiments/exp_foundation_basic.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --resume best.pth
```

Resume from a specific epoch:
```bash
python experiments/exp_foundation_basic.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --resume epoch_5.pth
```

### Full Path

You can also provide a full path to the checkpoint:
```bash
python experiments/exp_foundation_basic.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --resume /path/to/checkpoints/Foundation_Basic/epoch_10.pth
```

## What Gets Restored

When resuming from a checkpoint, the following are restored:

1. **Model State**: All model parameters
2. **Optimizer State**: Optimizer parameters and momentum
3. **Scaler State**: Mixed precision scaler state (if using FP16)
4. **Training History**:
   - Training losses for all completed epochs
   - Validation losses for all completed epochs
   - Validation metrics history (TEDS, accuracy, etc.)
5. **Best Model Info**:
   - Best validation loss
   - Best epoch number
6. **Vocabulary**: Vocabulary dictionary
7. **Configuration**: Experiment configuration from checkpoint

## Checkpoint Structure

Checkpoints are saved in:
```
experiment_results/
  checkpoints/
    {experiment_name}/
      best.pth          # Best model (lowest validation loss)
      latest.pth        # Latest epoch
      epoch_5.pth       # Every 5 epochs
      epoch_10.pth
      ...
```

## What Happens During Resume

1. **Checkpoint Loading**: The checkpoint is loaded from the specified path (supports both absolute and relative paths)
2. **State Restoration**: Model, optimizer, and scaler states are restored
3. **Epoch Check**: 
   - If checkpoint epoch < num_epochs: Training continues from `epoch + 1`
   - If checkpoint epoch >= num_epochs: Training resets to epoch 0 (fresh start with same model weights)
4. **History Preservation**: 
   - If continuing: All previous training/validation history is preserved
   - If resetting: Training history is cleared, but model weights are kept
5. **Best Model Tracking**: 
   - If continuing: Best model tracking continues from saved state
   - If resetting: Best model tracking is reset

## Example Output

### Normal Resume (epoch < num_epochs)

When resuming from an incomplete training, you'll see:
```
Loading checkpoint from /home/cain/code/github/dev/tsr/experiment_results/checkpoints/Foundation_Basic/latest.pth
  ✓ Loaded model state
  ✓ Loaded optimizer state
  ✓ Loaded scaler state
  ✓ Resuming from epoch 5
  ✓ Training losses history: 5 epochs
  ✓ Validation losses history: 5 epochs
  ✓ Best validation loss: 12.3456 (epoch 3)

Resuming with config from checkpoint:
  Epochs completed: 5
  Remaining epochs: 5

Effective batch size: 8 (batch_size=1 × accumulation=8)
Checkpoints will be saved to: experiment_results/checkpoints/Foundation_Basic
Random inference with TEDS will be displayed during validation

Epoch 6/10:
  Train Loss: 11.2345
  ...
```

### Reset Resume (epoch >= num_epochs)

When resuming from a completed training, you'll see:
```
Loading checkpoint from /home/cain/code/github/dev/tsr/experiment_results/checkpoints/Foundation_Basic/latest.pth
  ✓ Loaded model state
  ✓ Loaded optimizer state
  ✓ Loaded scaler state
  ✓ Resuming from epoch 10
  ✓ Training losses history: 10 epochs
  ✓ Validation losses history: 10 epochs
  ✓ Best validation loss: 10.1234 (epoch 8)

⚠️  Checkpoint shows training completed (epoch 10 >= 10)
   Resetting to epoch 0 to start fresh training
   Model weights will be reused, but training history will be reset

Effective batch size: 8 (batch_size=1 × accumulation=8)
Checkpoints will be saved to: experiment_results/checkpoints/Foundation_Basic
Random inference with TEDS will be displayed during validation

Epoch 1/10:
  Train Loss: 15.2345
  ...
```

## Supported Experiment Scripts

All experiment scripts support the `--resume` argument:
- `experiments/exp_foundation_basic.py`
- `experiments/exp_improvement_hybrid_regression.py`
- `experiments/exp_improvement_html_refiner.py`
- `experiments/exp_improvement_gc_attention.py`
- `experiments/exp_improvement_token_compression.py`
- `experiments/exp_improvement_all_combined.py`

## Notes

1. **Configuration**: The configuration from the checkpoint is loaded, but you can still override `num_epochs` in your command. The training will continue for the remaining epochs.

2. **Vocabulary**: The vocabulary from the checkpoint is used. Make sure your dataset uses the same vocabulary.

3. **Device**: The checkpoint is loaded to the device specified in your command (default: `cuda`).

4. **Mixed Precision**: If the checkpoint was saved with mixed precision, the scaler state is restored.

5. **Best Model**: The best model tracking continues from the checkpoint, so if you resume from `epoch_5.pth`, the best model might still be from an earlier epoch.

## Troubleshooting

### Checkpoint Not Found
```
FileNotFoundError: Checkpoint not found: ...
```
**Solution**: Make sure the checkpoint path is correct. Use relative paths like `latest.pth` or full absolute paths.

### Vocabulary Mismatch
If you get vocabulary-related errors, make sure you're using the same dataset that was used to create the checkpoint.

### Configuration Mismatch
If the model architecture changed, you may need to create a new experiment instead of resuming.

## Best Practices

1. **Use `latest.pth`** for general resumption - it's always the most recent checkpoint
2. **Use `best.pth`** if you want to continue from the best model
3. **Use `epoch_N.pth`** for specific epoch checkpoints
4. **Keep checkpoints** - they contain all necessary information to resume training
5. **Monitor disk space** - checkpoints can be large (especially with full training history)

