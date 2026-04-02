#!/bin/bash
# Quick script to create dummy dataset from PubTables1M

XML_DIR="/mnt/hdd2/data/pub1m/train/train"
WORDS_DIR="/mnt/hdd2/data/pub1m/words/words"
OUTPUT_DIR="./dummy_dataset"

echo "Creating dummy dataset from PubTables1M..."
echo "XML directory: $XML_DIR"
echo "Words directory: $WORDS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

XML_DIR="/mnt/hdd2/data/pub1m/train/train"
python create_dummy_dataset.py \
    --xml_dir "$XML_DIR" \
    --words_dir "$WORDS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_train 500000 \
    --num_val 0 \
    --num_test 0 \
    --seed 42


XML_DIR="/mnt/hdd2/data/pub1m/val/val"
python create_dummy_dataset.py \
    --xml_dir "$XML_DIR" \
    --words_dir "$WORDS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_train 0 \
    --num_val 10000 \
    --num_test 0 \
    --seed 42

XML_DIR="/mnt/hdd2/data/pub1m/test/test"
python create_dummy_dataset.py \
    --xml_dir "$XML_DIR" \
    --words_dir "$WORDS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_train 0 \
    --num_val 0 \
    --num_test 10000 \
    --seed 42

echo ""
echo "Dataset creation complete!"
echo "Check $OUTPUT_DIR for results"
