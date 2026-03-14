# PubTables1M Parser

This parser converts PubTables1M dataset annotations (XML structure + words JSON) into the format expected by the table recognition model.

## Overview

The PubTables1M dataset provides:
- **XML files**: Structural annotations (table, rows, columns, headers) with bounding boxes
- **Words JSON files**: Word-level OCR data with text and bounding boxes

The parser combines these to create cell-level annotations with:
- Cell content (text from words)
- Cell bounding boxes (intersection of row and column)
- Header flags (based on header row detection)

## Usage

### Single File

```bash
python parse_pub1m.py \
    --xml /path/to/PMC6701984_table_0.xml \
    --words /path/to/PMC6701984_table_0_words.json \
    --output /path/to/output.json
```

### Batch Processing

```bash
python parse_pub1m.py \
    --xml /path/to/xml_directory \
    --words /path/to/words_directory \
    --output /path/to/output_directory \
    --batch
```

### With Image Directory

```bash
python parse_pub1m.py \
    --xml /path/to/xml_directory \
    --words /path/to/words_directory \
    --output /path/to/output_directory \
    --image /path/to/image_directory \
    --batch
```

## Output Format

The parser generates JSON files in the following format:

```json
{
  "image_path": "/path/to/image.jpg",
  "table": {
    "cells": [
      {
        "content": "Cell text content",
        "bbox": [xmin, ymin, xmax, ymax],
        "is_header": false
      }
    ],
    "image_width": 593,
    "image_height": 251
  }
}
```

## How It Works

1. **Parse XML Structure**: Extracts table, row, column, and header bounding boxes
2. **Parse Words JSON**: Extracts word-level OCR data with text and bounding boxes
3. **Detect Spanning Cells**: 
   - Identifies words that overlap multiple grid cells
   - Groups cells that share words (indicating spanning)
   - Calculates union bounding boxes for spanning cells
4. **Assign Words to Cells**: Matches words to cells based on:
   - IoU (Intersection over Union) threshold (default: 0.1)
   - Word center point containment
   - Word bbox overlap with grid cells (for spanning detection)
5. **Merge Spanning Cells**: 
   - Combines grid cells that are part of the same spanning cell
   - Uses union of word bounding boxes to determine true cell boundaries
   - Preserves cell content from all words in the span
6. **Combine Words**: Joins words within each cell to form cell content
7. **Detect Headers**: Identifies header rows based on overlap with header bounding box

## Algorithm Details

### Cell Bounding Box Calculation

For each row-column intersection:
```python
cell_bbox = {
    xmin: max(row.xmin, col.xmin),
    ymin: max(row.ymin, col.ymin),
    xmax: min(row.xmax, col.xmax),
    ymax: min(row.ymax, col.ymax)
}
```

### Word Assignment

Words are assigned to cells if:
- IoU between word bbox and cell bbox > threshold (default: 0.1), OR
- Word center point is inside cell bbox

Words are then sorted by reading order (top-to-bottom, left-to-right).

### Spanning Cell Detection

The parser automatically detects and handles spanning cells (cells that span multiple rows or columns):

1. **Word Assignment**: Each word is assigned to all grid cells it overlaps with
2. **Grouping**: Words that span multiple grid cells are grouped together
3. **Merging**: Grid cells that share words are merged into a single spanning cell
4. **Bounding Box**: The spanning cell's bbox is calculated as:
   - Union of all covered grid cell bounding boxes
   - Expanded to include all word bounding boxes

This ensures that:
- Spanning cells are represented as single cells (not split into multiple cells)
- Cell content is preserved correctly
- Bounding boxes accurately reflect the true cell boundaries

### Header Detection

A row is marked as a header if:
- IoU between row bbox and header bbox > 0.5

## Example

```bash
# Parse the example file
python parse_pub1m.py \
    --xml /mnt/disks/data/flax/table_data/external/pub1m/org/test/test/PMC6701984_table_0.xml \
    --words /mnt/disks/data/flax/table_data/external/pub1m/org/words/words/PMC6701984_table_0_words.json \
    --output parsed_output.json
```

This will create `parsed_output.json` with 27 cells extracted from the table.

## Integration with Training

The parsed JSON files can be directly used with the `TableDataset` class:

```python
from tsr.data.dataset import TableDataset

dataset = TableDataset(
    data_path="/path/to/parsed_output.json",  # or directory of JSON files
    image_size=(512, 640)
)
```

## Troubleshooting

### Missing Words File
If a words file is not found, the parser will skip that XML file and print a warning.

### Invalid Bounding Boxes
Cells with invalid bounding boxes (xmin >= xmax or ymin >= ymax) are automatically skipped.

### Empty Cells
Empty cells are kept in the output (with empty content string) to preserve table structure. You can filter them out during training if needed.

## Customization

You can modify the parser behavior by editing `tsr/data/pub1m_parser.py`:

- **IoU Threshold**: Change `iou_threshold` parameter in `assign_words_to_cell()` method
- **Header Detection**: Modify `is_header_row()` method for different header detection logic
- **Word Sorting**: Adjust sorting key in `assign_words_to_cell()` for different reading orders

