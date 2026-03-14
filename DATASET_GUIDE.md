# Dataset Creation Guide

Guide for creating dummy datasets from PubTables1M for experiments.

## Overview

The dataset creation script converts PubTables1M XML/words format to the model's expected format and creates train/val/test splits.

## Dataset Format

The simplified dataset format uses a list of `(image_path, label_path)` tuples:

```json
[
  [
    "/path/to/image1.jpg",
    "/path/to/label1.json"
  ],
  [
    "/path/to/image2.jpg",
    "/path/to/label2.json"
  ]
]
```

Each label JSON file contains:
```json
{
  "image_path": "/path/to/image.jpg",
  "table": {
    "cells": [
      {
        "content": "Cell text",
        "bbox": [xmin, ymin, xmax, ymax],
        "is_header": false
      }
    ],
    "image_width": 593,
    "image_height": 251
  }
}
```

## Creating Dataset

### Basic Usage

```bash
python create_dummy_dataset.py \
    --xml_dir /mnt/disks/data/flax/table_data/external/pub1m/org/test/test \
    --words_dir /mnt/disks/data/flax/table_data/external/pub1m/org/words/words \
    --output_dir ./dummy_dataset \
    --num_train 500 \
    --num_val 100 \
    --num_test 100
```

### Quick Script

```bash
./create_dummy_dataset.sh
```

### Arguments

- `--xml_dir`: Directory containing Pub1M XML annotation files
- `--words_dir`: Directory containing Pub1M words JSON files
- `--output_dir`: Output directory for created dataset (default: `./dummy_dataset`)
- `--num_train`: Number of training samples (default: 500)
- `--num_val`: Number of validation samples (default: 100)
- `--num_test`: Number of test samples (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)

## Output Structure

```
dummy_dataset/
├── train/
│   ├── labels/
│   │   ├── PMC123456_table_0.json
│   │   ├── PMC123456_table_1.json
│   │   └── ...
│   └── dataset_list.json
├── val/
│   ├── labels/
│   │   └── ...
│   └── dataset_list.json
├── test/
│   ├── labels/
│   │   └── ...
│   └── dataset_list.json
└── dataset_summary.json
```

## Using the Dataset

### With Simplified Format

```python
from tsr.data.dataset import TableDataset
from torch.utils.data import DataLoader

# Load dataset using simplified format
dataset = TableDataset(
    data_path="./dummy_dataset/train/dataset_list.json",
    image_size=(512, 640),
    use_simplified_format=True  # Important!
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### With Legacy Format

The dataset still supports the legacy format (directory of JSON files):

```python
dataset = TableDataset(
    data_path="./dummy_dataset/train/labels",  # Directory
    image_size=(512, 640),
    use_simplified_format=False  # or omit
)
```

## Features

1. **Automatic Label Conversion**: Converts Pub1M XML/words to model format
2. **Image Path Resolution**: Automatically finds images in athena_format directory
3. **No Overlap**: Ensures train/val/test splits don't overlap
4. **Spanning Cell Support**: Preserves spanning cells in converted labels
5. **Progress Tracking**: Shows progress during conversion

## Notes

- The script automatically finds images in `/mnt/disks/data/flax/table_data/external/pub1m/org/athena_format/test`
- Each split uses a different random seed to ensure diversity
- Samples are tracked to prevent overlap between splits
- Failed conversions are skipped with warnings

## Troubleshooting

### Not Enough Samples

If you request more samples than available:
- The script will use all available samples
- A warning will be printed
- Check the summary JSON for actual counts

### Image Not Found

If images can't be found:
- Check that athena_format directory exists
- Images are searched in subdirectories (test*/input/)
- Fallback uses XML filename with .jpg extension

### Memory Issues

For large datasets:
- Process in smaller batches
- Consider using fewer samples
- Check available disk space for labels


