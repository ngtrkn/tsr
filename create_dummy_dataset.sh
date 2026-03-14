#!/bin/bash
# Quick script to create dummy dataset from PubTables1M

XML_DIR="/mnt/disks/data/flax/table_data/external/pub1m/org/train/train"
WORDS_DIR="/mnt/disks/data/flax/table_data/external/pub1m/org/words/words"
OUTPUT_DIR="./dummy_dataset"

echo "Creating dummy dataset from PubTables1M..."
echo "XML directory: $XML_DIR"
echo "Words directory: $WORDS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

python create_dummy_dataset.py \
    --xml_dir "$XML_DIR" \
    --words_dir "$WORDS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_train 100000 \
    --num_val 100 \
    --num_test 100 \
    --seed 42

echo ""
echo "Dataset creation complete!"
echo "Check $OUTPUT_DIR for results"


