# Table Recognition Result - Spanning Cell Example

**Source Image:** `PMC4631602_table_2.jpg`  
**Image Size:** 545 × 174 pixels  
**Total Cells:** 28

## Table

<table>
  <tr>
    <th>Technique</th>
    <th>Object Actual Size</th>
    <th>Camera Parameters</th>
    <th>μ in calculating distance</th>
    <th>Distance Error Ratio ρ</th>
    <th>μ in calculating size</th>
    <th>Size Error Ratio ρ</th>
  </tr>
  <tr>
    <td>PCA</td>
    <td>No</td>
    <td>No</td>
    <td>99%</td>
    <td>0.01</td>
    <td>100%</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Regression</td>
    <td>No</td>
    <td>No</td>
    <td>59%</td>
    <td>0.69</td>
    <td>2%</td>
    <td>49</td>
  </tr>
  <tr>
    <td>Oztarak et. al.</td>
    <td>Yes</td>
    <td>Yes</td>
    <td>40%</td>
    <td>1.5</td>
    <td>61%</td>
    <td>0.64</td>
  </tr>
</table>

---

## Example with Spanning Cells

When a table contains cells that span multiple rows or columns, the HTML export will automatically detect and add `rowspan` or `colspan` attributes. For example:

```html
<table>
  <tr>
    <th rowspan="2">Header spanning 2 rows</th>
    <th>Regular header</th>
  </tr>
  <tr>
    <td>Cell below spanning header</td>
  </tr>
  <tr>
    <td colspan="2">Cell spanning 2 columns</td>
  </tr>
</table>
```

### How Spanning Detection Works

1. **Grid Building**: The parser builds a 2D grid from the table structure (rows × columns)
2. **Cell Assignment**: Each parsed cell is assigned to its primary grid position
3. **Span Calculation**: For each cell, the parser calculates:
   - **rowspan**: Number of rows the cell spans (if cell height > 1.3× base row height)
   - **colspan**: Number of columns the cell spans (if cell width > 1.3× base column width)
4. **HTML Generation**: The Markdown/HTML output includes `rowspan` and `colspan` attributes when detected

### Usage

To generate this output:

```bash
python parse_pub1m.py \
    --xml /mnt/disks/data/flax/table_data/external/pub1m/org/test/test/PMC4631602_table_2.xml \
    --words /mnt/disks/data/flax/table_data/external/pub1m/org/words/words/PMC4631602_table_2_words.json \
    --output example_spanning_cell.json \
    --html
```

**Note:** This table was generated from parsed table structure. Spanning cells are detected based on bounding box overlaps.
