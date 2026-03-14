# TSR Evaluation Metrics

## Overview

The system now includes standard Table Structure Recognition (TSR) metrics for comprehensive evaluation:

## Metrics Implemented

### 1. TEDS (Tree-Edit-Distance-based Similarity)
- **Description**: Measures structural similarity between predicted and ground truth HTML tables
- **Range**: 0.0 to 1.0 (higher is better)
- **Calculation**: Based on tree edit distance between HTML structures
- **Usage**: Standard metric in table recognition research

### 2. Token-Level Accuracy
- **Description**: Percentage of correctly predicted tokens (excluding padding)
- **Range**: 0% to 100%
- **Calculation**: (correct_tokens / total_tokens) × 100

### 3. Structure Token Accuracy
- **Description**: Accuracy on structural tokens only (`<table>`, `<tr>`, `<td>`, etc.)
- **Range**: 0% to 100%
- **Calculation**: (correct_structure_tokens / total_structure_tokens) × 100

### 4. Content Token Accuracy
- **Description**: Accuracy on content tokens only (text content)
- **Range**: 0% to 100%
- **Calculation**: (correct_content_tokens / total_content_tokens) × 100

### 5. Exact Match Rate
- **Description**: Percentage of sequences that match exactly
- **Range**: 0% to 100%
- **Calculation**: (exact_match_sequences / total_sequences) × 100

### 6. Perplexity
- **Description**: Language modeling perplexity (exp(loss))
- **Range**: 1.0 to infinity (lower is better)
- **Calculation**: exp(validation_loss)

## Usage During Training

All metrics are automatically calculated and printed during validation:

```
Epoch 1/10:
  Train Loss: 15.2345
  Val Loss: 14.1234 | Perplexity: 1345678.90
  Val Token Acc: 45.67% | Structure Acc: 78.90% | Content Acc: 32.45%
  Exact Match Rate: 5.00% | TEDS: 0.8234
  → Saved best model (val_loss: 14.1234, token_acc: 45.67%, TEDS: 0.8234)
```

## Metric Details

### TEDS Calculation

TEDS compares the HTML structure of tables:
1. Converts token sequences to HTML
2. Parses HTML to XML trees
3. Calculates tree edit distance
4. Normalizes to similarity score: `1 - (edit_distance / max_tree_size)`

**Example:**
- Perfect match: TEDS = 1.0
- Structural match, different content: TEDS ≈ 0.8-0.9
- Different structure: TEDS < 0.5

### Why TEDS?

- **Standard Metric**: Widely used in table recognition papers
- **Structure-Focused**: Evaluates table structure correctness
- **Content-Aware**: Also considers cell content
- **Robust**: Handles minor structural differences gracefully

## Metrics Storage

All metrics are saved in:
- `experiment_results/{experiment_name}_results.json`
- `val_metrics_history`: Metrics for each epoch
- `final_val_metrics`: Final metrics at end of training

## Comparison with Baselines

Use these metrics to compare with:
- PubTables-1M baselines
- Other table recognition models
- Published research results

## References

- TEDS: Tree-Edit-Distance-based Similarity metric
- Standard in table structure recognition evaluation
- Used in PubTables-1M and related datasets


