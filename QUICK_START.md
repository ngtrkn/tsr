# Quick Start Guide

## Creating and Using Dummy Dataset

### Step 1: Create Dummy Dataset

```bash
python create_dummy_dataset.py \
    --xml_dir /mnt/disks/data/flax/table_data/external/pub1m/org/test/test \
    --words_dir /mnt/disks/data/flax/table_data/external/pub1m/org/words/words \
    --output_dir ./dummy_dataset \
    --num_train 500 \
    --num_val 100 \
    --num_test 100
```

Or use the quick script:
```bash
./create_dummy_dataset.sh
```

### Step 2: Run Experiments

```bash
# Foundation baseline
python experiments/exp_foundation_basic.py \
    --data_path ./dummy_dataset/train/dataset_list.json \
    --val_path ./dummy_dataset/val/dataset_list.json \
    --batch_size 4 \
    --num_epochs 10

# Improvement: Hybrid Regression
python experiments/exp_improvement_hybrid_regression.py \
    --data_path ./dummy_dataset/train/dataset_list.json \
    --val_path ./dummy_dataset/val/dataset_list.json \
    --batch_size 4 \
    --num_epochs 10

# ... and so on for other improvements
```

### Step 3: Compare Results

```bash
python experiments/compare_results.py \
    --results_dir ./experiment_results
```

## Dataset Format

The simplified dataset uses `dataset_list.json` files containing:
```json
[
  ["/path/to/image1.jpg", "/path/to/label1.json"],
  ["/path/to/image2.jpg", "/path/to/label2.json"]
]
```

Each label JSON contains the full table structure in model format.


