#!/bin/bash
# Run all experiments sequentially
# Usage: ./run_all_experiments.sh <data_path> [val_path] [output_dir] [batch_size] [num_epochs]

DATA_PATH=${1:-"data/train"}
VAL_PATH=${2:-"data/val"}
OUTPUT_DIR=${3:-"./experiment_results"}
BATCH_SIZE=${4:-4}
NUM_EPOCHS=${5:-10}

echo "Running all experiments..."
echo "Data path: $DATA_PATH"
echo "Val path: $VAL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Num epochs: $NUM_EPOCHS"
echo ""

# Foundation
echo "=========================================="
echo "Running Foundation Basic..."
echo "=========================================="
python experiments/exp_foundation_basic.py \
    --data_path "$DATA_PATH" \
    --val_path "$VAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS"

# Improvements
echo ""
echo "=========================================="
echo "Running Improvement: Hybrid Regression..."
echo "=========================================="
python experiments/exp_improvement_hybrid_regression.py \
    --data_path "$DATA_PATH" \
    --val_path "$VAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS"

echo ""
echo "=========================================="
echo "Running Improvement: HTML Refiner..."
echo "=========================================="
python experiments/exp_improvement_html_refiner.py \
    --data_path "$DATA_PATH" \
    --val_path "$VAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS"

echo ""
echo "=========================================="
echo "Running Improvement: GCAttention..."
echo "=========================================="
python experiments/exp_improvement_gc_attention.py \
    --data_path "$DATA_PATH" \
    --val_path "$VAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS"

echo ""
echo "=========================================="
echo "Running Improvement: Token Compression..."
echo "=========================================="
python experiments/exp_improvement_token_compression.py \
    --data_path "$DATA_PATH" \
    --val_path "$VAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --compression_ratio 0.8

echo ""
echo "=========================================="
echo "Running Improvement: All Combined..."
echo "=========================================="
python experiments/exp_improvement_all_combined.py \
    --data_path "$DATA_PATH" \
    --val_path "$VAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --compression_ratio 0.8

# Compare results
echo ""
echo "=========================================="
echo "Generating comparison report..."
echo "=========================================="
python experiments/compare_results.py \
    --results_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/comparison_report.md"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Comparison report: $OUTPUT_DIR/comparison_report.md"
echo "=========================================="


