# Experiment Setup Verification

## Status: ✅ Ready to Run

All experiments have been tested and verified to work correctly with the conda environment `/home/cain/conda/tsr`.

## Key Fixes Applied

1. **Sequence Padding**: Added `collate_fn` to handle variable-length sequences by padding to the maximum length in each batch
2. **DataLoader Configuration**: Updated all experiment scripts to use:
   - `collate_fn=collate_fn` for proper sequence padding
   - `num_workers=0` to avoid multiprocessing issues
3. **Import Updates**: Added `collate_fn` import to all experiment scripts

## Verified Components

✅ Dataset loading with simplified format (`dataset_list.json`)
✅ Model creation and forward pass
✅ Loss computation (MultiTaskLoss)
✅ Training loop (one epoch tested)
✅ Validation loop (tested)
✅ Batch collation with variable-length sequences

## Running Experiments

### Foundation Baseline
```bash
conda activate /home/cain/conda/tsr
python experiments/exp_foundation_basic.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --batch_size 4 \
    --num_epochs 10 \
    --device cpu  # or cuda if available
```

### Improvement Experiments
```bash
# Hybrid Regression
python experiments/exp_improvement_hybrid_regression.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --batch_size 4 \
    --num_epochs 10

# HTML Refiner
python experiments/exp_improvement_html_refiner.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --batch_size 4 \
    --num_epochs 10

# GCAttention
python experiments/exp_improvement_gc_attention.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --batch_size 4 \
    --num_epochs 10

# Token Compression
python experiments/exp_improvement_token_compression.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --batch_size 4 \
    --num_epochs 10

# All Combined
python experiments/exp_improvement_all_combined.py \
    --data_path dummy_dataset/train/dataset_list.json \
    --val_path dummy_dataset/val/dataset_list.json \
    --batch_size 4 \
    --num_epochs 10
```

## Test Results

Quick test run (8 samples, 1 epoch):
- Training loss: 16.08
- Validation loss: 14.34
- ✅ No errors encountered

## Notes

- All experiments use `num_workers=0` to avoid multiprocessing issues
- Sequences are automatically padded to the maximum length in each batch
- Padding token ID is 0 (as defined in `serialization.py`)
- Image size is (512, 640) - height x width
- Model supports both CPU and CUDA devices


