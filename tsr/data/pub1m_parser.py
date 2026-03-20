"""
Parser for PubTables1M dataset
Combines XML structure annotations with word-level OCR data
"""
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class BBox:
    """Bounding box representation"""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    
    def area(self) -> float:
        """Calculate area of bounding box"""
        return max(0, self.xmax - self.xmin) * max(0, self.ymax - self.ymin)
    
    def intersection(self, other: 'BBox') -> 'BBox':
        """Calculate intersection with another bounding box"""
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        return BBox(xmin, ymin, xmax, ymax)
    
    def iou(self, other: 'BBox') -> float:
        """Calculate Intersection over Union"""
        inter = self.intersection(other)
        inter_area = inter.area()
        union_area = self.area() + other.area() - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside bounding box"""
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax
    
    def center(self) -> Tuple[float, float]:
        """Get center point"""
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)


@dataclass
class Word:
    """Word with bounding box and text"""
    text: str
    bbox: BBox
    span_num: Optional[int] = None
    line_num: Optional[int] = None
    block_num: Optional[int] = None


@dataclass
class TableStructure:
    """Table structure from XML"""
    table_bbox: BBox
    rows: List[BBox]
    columns: List[BBox]
    header_bbox: Optional[BBox] = None


class Pub1MParser:
    """Parser for PubTables1M dataset"""
    
    def __init__(self, xml_path: str, words_path: str, image_path: Optional[str] = None):
        """
        Args:
            xml_path: Path to XML annotation file
            words_path: Path to words JSON file
            image_path: Optional path to image file (for validation)
        """
        self.xml_path = Path(xml_path)
        self.words_path = Path(words_path)
        self.image_path = Path(image_path) if image_path else None
        
        # Parse XML
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        
        # Parse words JSON
        with open(self.words_path, 'r') as f:
            self.words_data = json.load(f)
        
        # Extract image dimensions
        size_elem = self.root.find('size')
        self.image_width = int(size_elem.find('width').text)
        self.image_height = int(size_elem.find('height').text)
        
        # Extract filename
        filename_elem = self.root.find('filename')
        self.filename = filename_elem.text if filename_elem is not None else None
    
    def parse_structure(self) -> TableStructure:
        """Parse table structure from XML"""
        table_bbox = None
        rows = []
        columns = []
        header_bbox = None
        
        for obj in self.root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            
            if bndbox is None:
                continue
            
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            bbox = BBox(xmin, ymin, xmax, ymax)
            
            if name == 'table':
                table_bbox = bbox
            elif name == 'table row':
                rows.append(bbox)
            elif name == 'table column':
                columns.append(bbox)
            elif name == 'table column header':
                header_bbox = bbox
        
        if table_bbox is None:
            raise ValueError("No table bounding box found in XML")
        
        # Sort rows by ymin
        rows.sort(key=lambda r: r.ymin)
        
        # Sort columns by xmin
        columns.sort(key=lambda c: c.xmin)
        
        return TableStructure(
            table_bbox=table_bbox,
            rows=rows,
            columns=columns,
            header_bbox=header_bbox
        )
    
    def parse_words(self) -> List[Word]:
        """Parse words from JSON"""
        words = []
        
        for word_data in self.words_data:
            bbox_data = word_data['bbox']
            text = word_data['text']
            
            # Handle different bbox formats
            if isinstance(bbox_data, list) and len(bbox_data) == 4:
                xmin, ymin, xmax, ymax = bbox_data
            else:
                continue
            
            # Skip words with invalid coordinates
            if xmin >= xmax or ymin >= ymax:
                continue
            
            word = Word(
                text=text,
                bbox=BBox(xmin, ymin, xmax, ymax),
                span_num=word_data.get('span_num'),
                line_num=word_data.get('line_num'),
                block_num=word_data.get('block_num')
            )
            words.append(word)
        
        return words
    
    def find_cell_bbox(self, row: BBox, col: BBox) -> BBox:
        """Find bounding box for cell at row/column intersection"""
        return BBox(
            xmin=max(row.xmin, col.xmin),
            ymin=max(row.ymin, col.ymin),
            xmax=min(row.xmax, col.xmax),
            ymax=min(row.ymax, col.ymax)
        )
    
    def assign_words_to_cell(self, cell_bbox: BBox, words: List[Word], 
                           iou_threshold: float = 0.1) -> List[Word]:
        """Assign words to a cell based on bounding box overlap"""
        assigned_words = []
        
        for word in words:
            # Calculate IoU
            iou = cell_bbox.iou(word.bbox)
            
            # Also check if word center is in cell
            center_x, center_y = word.bbox.center()
            center_in_cell = cell_bbox.contains_point(center_x, center_y)
            
            if iou > iou_threshold or center_in_cell:
                assigned_words.append(word)
        
        # Sort words by reading order (left to right, top to bottom)
        assigned_words.sort(key=lambda w: (w.bbox.ymin, w.bbox.xmin))
        
        return assigned_words
    
    def is_header_row(self, row: BBox, header_bbox: Optional[BBox]) -> bool:
        """Check if row is a header row"""
        if header_bbox is None:
            return False
        
        # Check if row overlaps significantly with header bbox
        iou = row.iou(header_bbox)
        return iou > 0.5
    
    def detect_spanning_cells(self, structure: TableStructure, words: List[Word]) -> Dict[Tuple[int, int], List[Word]]:
        """
        Detect spanning cells by analyzing word assignments.
        Returns a mapping from (row_idx, col_idx) to words, where spanning cells
        will have words assigned to multiple grid positions.
        """
        word_assignments = {}  # Maps word index to list of (row_idx, col_idx) positions
        
        # First pass: assign each word to all grid cells it overlaps with
        for word_idx, word in enumerate(words):
            word_assignments[word_idx] = []
            
            for row_idx, row in enumerate(structure.rows):
                for col_idx, col in enumerate(structure.columns):
                    cell_bbox = self.find_cell_bbox(row, col)
                    
                    if cell_bbox.xmin >= cell_bbox.xmax or cell_bbox.ymin >= cell_bbox.ymax:
                        continue
                    
                    # Check if word overlaps with this cell
                    iou = cell_bbox.iou(word.bbox)
                    center_x, center_y = word.bbox.center()
                    center_in_cell = cell_bbox.contains_point(center_x, center_y)
                    
                    if iou > 0.1 or center_in_cell:
                        word_assignments[word_idx].append((row_idx, col_idx))
        
        # Group words that span multiple cells together
        # Create a graph of connected cells (cells that share words)
        cell_groups = {}  # Maps representative position to set of positions
        word_to_group = {}  # Maps word index to representative position
        
        for word_idx, positions in word_assignments.items():
            if len(positions) == 0:
                continue
            
            # If word spans multiple cells, they should be grouped
            if len(positions) > 1:
                # Find or create a group for these positions
                # Use the top-left position as representative
                positions_sorted = sorted(positions, key=lambda p: (p[0], p[1]))
                rep_pos = positions_sorted[0]
                
                # Merge all positions into the same group
                if rep_pos not in cell_groups:
                    cell_groups[rep_pos] = set([rep_pos])
                
                for pos in positions:
                    cell_groups[rep_pos].add(pos)
                    word_to_group[word_idx] = rep_pos
            else:
                # Single cell word
                pos = positions[0]
                if pos not in cell_groups:
                    cell_groups[pos] = set([pos])
                word_to_group[word_idx] = pos
        
        # Second pass: assign words to their representative cell
        cell_word_map = {}  # Maps (row_idx, col_idx) to list of word indices
        
        for word_idx, rep_pos in word_to_group.items():
            if rep_pos not in cell_word_map:
                cell_word_map[rep_pos] = []
            cell_word_map[rep_pos].append(word_idx)
        
        # Convert word indices back to Word objects
        result = {}
        for pos, word_indices in cell_word_map.items():
            result[pos] = [words[idx] for idx in word_indices]
        
        return result
    
    def merge_spanning_cells(self, cell_word_map: Dict[Tuple[int, int], List[Word]], 
                            structure: TableStructure) -> List[Dict]:
        """
        Merge spanning cells by grouping adjacent cells that share words.
        Returns a list of cell data dictionaries with proper spanning cell bboxes.
        """
        cells = []
        processed_positions = set()
        
        for (row_idx, col_idx), cell_words in cell_word_map.items():
            if (row_idx, col_idx) in processed_positions:
                continue
            
            # Get the basic cell bbox for this grid position
            row = structure.rows[row_idx]
            col = structure.columns[col_idx]
            basic_cell_bbox = self.find_cell_bbox(row, col)
            
            # Check if header
            is_header = self.is_header_row(row, structure.header_bbox)
            
            # Calculate actual cell bbox from words
            if cell_words:
                word_xmins = [w.bbox.xmin for w in cell_words]
                word_ymins = [w.bbox.ymin for w in cell_words]
                word_xmaxs = [w.bbox.xmax for w in cell_words]
                word_ymaxs = [w.bbox.ymax for w in cell_words]
                
                actual_xmin = min(word_xmins)
                actual_ymin = min(word_ymins)
                actual_xmax = max(word_xmaxs)
                actual_ymax = max(word_ymaxs)
                
                # Check if this cell spans multiple grid positions
                # by finding all grid cells that overlap with the word bbox
                word_bbox = BBox(actual_xmin, actual_ymin, actual_xmax, actual_ymax)
                
                # Find all grid cells covered by this spanning cell
                covered_positions = []
                for r_idx, r in enumerate(structure.rows):
                    for c_idx, c in enumerate(structure.columns):
                        grid_cell_bbox = self.find_cell_bbox(r, c)
                        if grid_cell_bbox.xmin >= grid_cell_bbox.xmax or grid_cell_bbox.ymin >= grid_cell_bbox.ymax:
                            continue
                        
                        # Check if this grid cell overlaps with the word bbox
                        # Use IoU or check if word bbox significantly overlaps
                        iou = grid_cell_bbox.iou(word_bbox)
                        overlap_ratio = iou if iou > 0 else 0
                        
                        # Also check if word bbox center is in grid cell
                        center_x, center_y = word_bbox.center()
                        center_in_cell = grid_cell_bbox.contains_point(center_x, center_y)
                        
                        # Consider it covered if significant overlap or center is inside
                        if overlap_ratio > 0.2 or center_in_cell:
                            covered_positions.append((r_idx, c_idx))
                
                # If this cell spans multiple grid positions, merge them
                if len(covered_positions) > 1:
                    # Calculate the union bbox of all covered grid cells
                    covered_bboxes = [self.find_cell_bbox(structure.rows[r], structure.columns[c]) 
                                     for r, c in covered_positions]
                    
                    union_xmin = min(b.xmin for b in covered_bboxes)
                    union_ymin = min(b.ymin for b in covered_bboxes)
                    union_xmax = max(b.xmax for b in covered_bboxes)
                    union_ymax = max(b.ymax for b in covered_bboxes)
                    
                    # Use the union of grid cells, but expand to include all words
                    cell_bbox = BBox(
                        min(union_xmin, actual_xmin),
                        min(union_ymin, actual_ymin),
                        max(union_xmax, actual_xmax),
                        max(union_ymax, actual_ymax)
                    )
                    
                    # Mark all covered positions as processed
                    for pos in covered_positions:
                        processed_positions.add(pos)
                else:
                    # Regular cell - use word bbox but align with grid boundaries
                    cell_bbox = BBox(
                        min(actual_xmin, basic_cell_bbox.xmin),
                        min(actual_ymin, basic_cell_bbox.ymin),
                        max(actual_xmax, basic_cell_bbox.xmax),
                        max(actual_ymax, basic_cell_bbox.ymax)
                    )
                    processed_positions.add((row_idx, col_idx))
                
                # Combine words into content
                cell_content = ' '.join([w.text for w in cell_words]).strip()
            else:
                # Empty cell - no words assigned
                cell_bbox = basic_cell_bbox
                cell_content = ""
                processed_positions.add((row_idx, col_idx))
            
            cell_data = {
                "content": cell_content,
                "bbox": [cell_bbox.xmin, cell_bbox.ymin, cell_bbox.xmax, cell_bbox.ymax],
                "is_header": is_header
            }
            
            cells.append(cell_data)
        
        # Add empty cells that weren't processed (cells with no words)
        for row_idx, row in enumerate(structure.rows):
            is_header = self.is_header_row(row, structure.header_bbox)
            for col_idx, col in enumerate(structure.columns):
                if (row_idx, col_idx) not in processed_positions:
                    cell_bbox = self.find_cell_bbox(row, col)
                    if cell_bbox.xmin < cell_bbox.xmax and cell_bbox.ymin < cell_bbox.ymax:
                        cell_data = {
                            "content": "",
                            "bbox": [cell_bbox.xmin, cell_bbox.ymin, cell_bbox.xmax, cell_bbox.ymax],
                            "is_header": is_header
                        }
                        cells.append(cell_data)
        
        # Sort cells by position (top to bottom, left to right)
        cells.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
        
        return cells
    
    def parse_to_model_format(self, image_base: Path = None) -> Dict:
        """
        Parse XML and words to model-expected format with spanning cell support
        
        Returns:
            Dictionary with:
                - image_path: Path to image
                - table: {
                    - cells: List of cell data
                    - image_width: Image width
                    - image_height: Image height
                }
        """
        # Parse structure
        structure = self.parse_structure()
        
        # Parse words
        words = self.parse_words()
        
        # Detect spanning cells and assign words
        cell_word_map = self.detect_spanning_cells(structure, words)
        
        # Merge spanning cells and create final cell list
        cells = self.merge_spanning_cells(cell_word_map, structure)
        
        # Determine image path
        if self.image_path and Path(self.image_path).exists():
            image_path = str(self.image_path)
        elif self.filename:
            # Try multiple locations for image
            image_base = Path("/mnt/hdd2/data/pub1m/images/images") if image_base is not None else image_base
            possible_paths = [
                image_base / f"{self.filename}"
            ]   # origin pub1m label
            if Path(image_base / "train").exists():     # Cinnamon convert
                for category in ["train", "test", "val"]:
                    for number in range(400):
                        possible_paths.append(image_base / category / f"{category}{number}" / "input" / f"{self.filename}")


            
            image_path = None
            for path in possible_paths:
                if path.exists():
                    image_path = str(path)
                    break
            
            if image_path is None:
                # Use XML path but change extension as fallback
                image_path = str(self.xml_path.with_suffix('.jpg'))
        else:
            # Use XML path but change extension
            image_path = str(self.xml_path.with_suffix('.jpg'))
        
        return {
            "image_path": image_path,
            "table": {
                "cells": cells,
                "image_width": self.image_width,
                "image_height": self.image_height
            }
        }
    
    def visualize_labels(self, output_image_path: Optional[str] = None) -> Optional[str]:
        """
        Visualize parsed labels on the image
        
        Args:
            output_image_path: Path to save visualized image (optional)
        Returns:
            Path to saved image if output_image_path provided, else None
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("Warning: PIL not available, skipping visualization")
            return None
        
        data = self.parse_to_model_format()
        image_path = data['image_path']
        
        if not Path(image_path).exists():
            print(f"Warning: Image not found at {image_path}, skipping visualization")
            return None
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw cells
        for cell in data['table']['cells']:
            bbox = cell['bbox']
            xmin, ymin, xmax, ymax = bbox
            
            # Choose color based on header
            if cell['is_header']:
                outline_color = (255, 0, 0)  # Red for headers
            else:
                outline_color = (0, 255, 0)  # Green for regular cells
            
            # Draw rectangle
            draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_color, width=2)
            
            # Draw text (first 30 chars) if there's content
            if cell['content']:
                text = cell['content'][:30] + ('...' if len(cell['content']) > 30 else '')
                # Draw text background
                try:
                    bbox_text = draw.textbbox((xmin + 2, ymin + 2), text, font=font_small)
                    draw.rectangle(bbox_text, fill=(255, 255, 255, 200))
                except:
                    pass
                # Draw text
                draw.text((xmin + 2, ymin + 2), text, fill=(0, 0, 0), font=font_small)
        
        # Save or return
        if output_image_path:
            output_path = Path(output_image_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_image_path)
            print(f"Saved visualized image to {output_image_path}")
            return output_image_path
        else:
            return None
    
    def _build_grid_from_cells(self, cells: List[Dict], structure: TableStructure) -> List[List[Optional[Dict]]]:
        """
        Build a 2D grid from cells to detect spanning.
        Returns a grid where each cell is either a cell dict or None.
        """
        # Create grid based on row/column structure
        num_rows = len(structure.rows)
        num_cols = len(structure.columns)
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        
        # Sort cells by position (top-left first)
        sorted_cells = sorted(cells, key=lambda c: (c['bbox'][1], c['bbox'][0]))
        
        # Assign cells to grid positions - find the top-left position for each cell
        for cell in sorted_cells:
            bbox = BBox(*cell['bbox'])
            
            # Find the top-left grid position this cell belongs to
            best_row = None
            best_col = None
            best_overlap = 0
            
            for row_idx, row in enumerate(structure.rows):
                for col_idx, col in enumerate(structure.columns):
                    grid_cell_bbox = self.find_cell_bbox(row, col)
                    
                    # Check overlap
                    iou = grid_cell_bbox.iou(bbox)
                    center_x, center_y = bbox.center()
                    center_in_cell = grid_cell_bbox.contains_point(center_x, center_y)
                    
                    # Prefer top-left position with good overlap
                    if (iou > 0.2 or center_in_cell) and (best_row is None or (row_idx < best_row or (row_idx == best_row and col_idx < best_col))):
                        if iou > best_overlap or center_in_cell:
                            best_row = row_idx
                            best_col = col_idx
                            best_overlap = iou
            
            # Assign to best position if found
            if best_row is not None and best_col is not None:
                if grid[best_row][best_col] is None:
                    grid[best_row][best_col] = cell
        
        return grid
    
    def _calculate_spanning(self, grid: List[List[Optional[Dict]]], 
                           structure: TableStructure) -> List[List[Dict]]:
        """
        Calculate rowspan and colspan for each cell in the grid.
        Returns a list of rows, each containing cell dicts with rowspan/colspan info.
        """
        num_rows = len(grid)
        num_cols = len(grid[0]) if grid else 0
        
        # Track which grid positions are covered by spanning cells
        covered = [[False for _ in range(num_cols)] for _ in range(num_rows)]
        result_rows = []
        
        for row_idx in range(num_rows):
            result_row = []
            
            for col_idx in range(num_cols):
                # Skip if already covered by a spanning cell
                if covered[row_idx][col_idx]:
                    continue
                
                cell = grid[row_idx][col_idx]
                
                if cell is None:
                    # Empty cell
                    result_row.append({
                        'content': '',
                        'is_header': False,
                        'rowspan': 1,
                        'colspan': 1
                    })
                    covered[row_idx][col_idx] = True
                else:
                    # Calculate how many rows/columns this cell spans
                    cell_bbox = BBox(*cell['bbox'])
                    
                    # Get the base grid cell bbox
                    base_row_bbox = structure.rows[row_idx]
                    base_col_bbox = structure.columns[col_idx]
                    base_cell_bbox = self.find_cell_bbox(base_row_bbox, base_col_bbox)
                    
                    # Calculate rowspan - check which rows the cell extends into
                    rowspan = 1
                    cell_height = cell_bbox.ymax - cell_bbox.ymin
                    base_row_height = base_row_bbox.ymax - base_row_bbox.ymin
                    
                    # If cell is significantly taller than base row, it likely spans
                    if cell_height > base_row_height * 1.3:
                        for r in range(row_idx + 1, num_rows):
                            row_bbox = structure.rows[r]
                            # Check if cell extends into this row
                            if cell_bbox.ymax > row_bbox.ymin + (row_bbox.ymax - row_bbox.ymin) * 0.2:
                                rowspan += 1
                            else:
                                break
                    
                    # Calculate colspan - check which columns the cell extends into
                    colspan = 1
                    cell_width = cell_bbox.xmax - cell_bbox.xmin
                    base_col_width = base_col_bbox.xmax - base_col_bbox.xmin
                    
                    # If cell is significantly wider than base column, it likely spans
                    if cell_width > base_col_width * 1.3:
                        for c in range(col_idx + 1, num_cols):
                            col_bbox = structure.columns[c]
                            # Check if cell extends into this column
                            if cell_bbox.xmax > col_bbox.xmin + (col_bbox.xmax - col_bbox.xmin) * 0.2:
                                colspan += 1
                            else:
                                break
                    
                    # Mark covered positions
                    for r in range(row_idx, row_idx + rowspan):
                        for c in range(col_idx, col_idx + colspan):
                            if r < num_rows and c < num_cols:
                                covered[r][c] = True
                    
                    result_row.append({
                        'content': cell['content'],
                        'is_header': cell['is_header'],
                        'rowspan': rowspan,
                        'colspan': colspan
                    })
            
            if result_row:
                result_rows.append(result_row)
        
        return result_rows
    
    def export_html(self, output_html_path: str):
        """
        Export table to Markdown format with HTML table supporting spanning cells
        
        Args:
            output_html_path: Path to save Markdown/HTML file
        """
        data = self.parse_to_model_format()
        cells = data['table']['cells']
        
        if not cells:
            print("Warning: No cells to export")
            return
        
        # Parse structure to get rows and columns
        structure = self.parse_structure()
        
        # Build grid and calculate spanning
        grid = self._build_grid_from_cells(cells, structure)
        rows_with_spanning = self._calculate_spanning(grid, structure)
        
        # Generate Markdown with HTML table
        md_content = f"""# Table Recognition Result

**Source Image:** `{Path(data['image_path']).name}`  
**Image Size:** {data['table']['image_width']} × {data['table']['image_height']} pixels  
**Total Cells:** {len(cells)}

## Table

<table>
"""
        
        # Add table rows with spanning support
        for row_idx, row in enumerate(rows_with_spanning):
            md_content += "  <tr>\n"
            
            for cell in row:
                tag = "th" if cell['is_header'] else "td"
                content = self._escape_html(cell['content'])
                
                # Add rowspan and colspan attributes if > 1
                attrs = []
                if cell['rowspan'] > 1:
                    attrs.append(f'rowspan="{cell["rowspan"]}"')
                if cell['colspan'] > 1:
                    attrs.append(f'colspan="{cell["colspan"]}"')
                
                attr_str = ' ' + ' '.join(attrs) if attrs else ''
                md_content += f"    <{tag}{attr_str}>{content}</{tag}>\n"
            
            md_content += "  </tr>\n"
        
        md_content += """</table>

---

**Note:** This table was generated from parsed table structure. Spanning cells are detected based on bounding box overlaps.
"""
        
        # Save as Markdown file
        output_path = Path(output_html_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Saved Markdown/HTML export to {output_html_path}")
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#39;'))
    
    def save_json(self, output_path: str):
        """Parse and save to JSON format"""
        data = self.parse_to_model_format()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved parsed data to {output_path}")
        print(f"  Found {len(data['table']['cells'])} cells")
        print(f"  Image: {data['image_path']}")


def parse_pub1m_directory(
    xml_dir: str,
    words_dir: str,
    output_dir: str,
    image_dir: Optional[str] = None,
    pattern: str = "*.xml",
    visualize: bool = False,
    export_html: bool = False
):
    """
    Parse all XML files in a directory
    
    Args:
        xml_dir: Directory containing XML annotation files
        words_dir: Directory containing words JSON files
        output_dir: Output directory for parsed JSON files
        image_dir: Optional directory containing images
        pattern: File pattern to match (default: "*.xml")
        visualize: Whether to generate visualization images
        export_html: Whether to export HTML files
    """
    xml_dir = Path(xml_dir)
    words_dir = Path(words_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(xml_dir.glob(pattern))
    
    print(f"Found {len(xml_files)} XML files to process")
    
    for xml_file in xml_files:
        try:
            # Find corresponding words file
            words_file = words_dir / f"{xml_file.stem}_words.json"
            
            if not words_file.exists():
                print(f"Warning: Words file not found for {xml_file.name}")
                continue
            
            # Find corresponding image file
            image_file = None
            if image_dir:
                # Try different extensions
                for ext in ['.jpg', '.png', '.jpeg']:
                    candidate = Path(image_dir) / f"{xml_file.stem}{ext}"
                    if candidate.exists():
                        image_file = str(candidate)
                        break
            
            # Parse
            parser = Pub1MParser(
                xml_path=str(xml_file),
                words_path=str(words_file),
                image_path=image_file
            )
            
            # Save output
            output_file = output_dir / f"{xml_file.stem}.json"
            parser.save_json(str(output_file))
            
        except Exception as e:
            print(f"Error processing {xml_file.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse PubTables1M dataset")
    parser.add_argument("--xml", type=str, required=True, help="Path to XML file or directory")
    parser.add_argument("--words", type=str, required=True, help="Path to words JSON file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file or directory")
    parser.add_argument("--image", type=str, default=None, help="Path to image file or directory (optional)")
    parser.add_argument("--batch", action="store_true", help="Process directory in batch mode")
    
    args = parser.parse_args()
    
    if args.batch or Path(args.xml).is_dir():
        # Batch processing
        parse_pub1m_directory(
            xml_dir=args.xml,
            words_dir=args.words,
            output_dir=args.output,
            image_dir=args.image
        )
    else:
        # Single file processing
        parser = Pub1MParser(
            xml_path=args.xml,
            words_path=args.words,
            image_path=args.image
        )
        parser.save_json(args.output)

