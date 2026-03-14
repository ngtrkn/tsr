"""
Unified Sequence Serialization for Table Recognition
Implements coordinate discretization and token quartet representation
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# Special tokens
SEP_TOKEN = "<Sep>"
PAD_TOKEN = "<Pad>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

# Structural HTML tokens
TABLE_START = "<table>"
TABLE_END = "</table>"
ROW_START = "<tr>"
ROW_END = "</tr>"
CELL_START = "<td>"
CELL_END = "</td>"
HEADER_START = "<th>"
HEADER_END = "</th>"

# Spatial coordinate tokens
XMIN_TOKEN = "<Xmin>"
YMIN_TOKEN = "<Ymin>"
XMAX_TOKEN = "<Xmax>"
YMAX_TOKEN = "<Ymax>"

# Grid dimensions for coordinate discretization
GRID_WIDTH = 1024
GRID_HEIGHT = 1280


@dataclass
class CellData:
    """Represents a single table cell"""
    content: str
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    is_header: bool = False


@dataclass
class TableData:
    """Represents a complete table structure"""
    cells: List[CellData]
    image_width: int
    image_height: int


class CoordinateDiscretizer:
    """Discretizes continuous coordinates to fixed grid tokens"""
    
    def __init__(self, grid_width: int = GRID_WIDTH, grid_height: int = GRID_HEIGHT):
        self.grid_width = grid_width
        self.grid_height = grid_height
        
    def discretize_coordinate(self, coord: float, max_dim: int, grid_dim: int) -> int:
        """Discretize a single coordinate value"""
        normalized = coord / max_dim
        discrete = int(normalized * grid_dim)
        return max(0, min(discrete, grid_dim - 1))
    
    def discretize_bbox(self, bbox: Tuple[float, float, float, float], 
                       img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Discretize bounding box coordinates to grid space"""
        xmin, ymin, xmax, ymax = bbox
        return (
            self.discretize_coordinate(xmin, img_width, self.grid_width),
            self.discretize_coordinate(ymin, img_height, self.grid_height),
            self.discretize_coordinate(xmax, img_width, self.grid_width),
            self.discretize_coordinate(ymax, img_height, self.grid_height)
        )
    
    def continuous_to_tokens(self, bbox: Tuple[float, float, float, float],
                            img_width: int, img_height: int) -> List[str]:
        """Convert continuous bbox to discrete spatial tokens"""
        xmin, ymin, xmax, ymax = self.discretize_bbox(bbox, img_width, img_height)
        return [
            f"{XMIN_TOKEN}{xmin}",
            f"{YMIN_TOKEN}{ymin}",
            f"{XMAX_TOKEN}{xmax}",
            f"{YMAX_TOKEN}{ymax}"
        ]


class SequenceSerializer:
    """Serializes table data into unified autoregressive sequence"""
    
    def __init__(self, grid_width: int = GRID_WIDTH, grid_height: int = GRID_HEIGHT):
        self.discretizer = CoordinateDiscretizer(grid_width, grid_height)
        self.grid_width = grid_width
        self.grid_height = grid_height
    
    def serialize_table(self, table: TableData) -> List[str]:
        """
        Serialize table into unified sequence: y = {c, b, t, <Sep>}
        where c = content, b = bbox tokens, t = structural HTML tags
        """
        sequence = [BOS_TOKEN, TABLE_START]
        
        # Group cells by rows (assuming cells are ordered)
        # In practice, you'd need row detection logic
        current_row = []
        prev_y = None
        
        for cell in table.cells:
            # Simple row detection: if ymin changes significantly, new row
            _, ymin, _, _ = cell.bbox
            if prev_y is not None and abs(ymin - prev_y) > 10:
                # End previous row and start new one
                if current_row:
                    sequence.extend(self._serialize_row(current_row, table))
                    sequence.append(ROW_END)
                sequence.append(ROW_START)
                current_row = []
            
            current_row.append(cell)
            prev_y = ymin
        
        # Serialize last row
        if current_row:
            sequence.extend(self._serialize_row(current_row, table))
            sequence.append(ROW_END)
        
        sequence.append(TABLE_END)
        sequence.append(EOS_TOKEN)
        
        return sequence
    
    def _serialize_row(self, cells: List[CellData], table: TableData) -> List[str]:
        """Serialize a row of cells"""
        row_tokens = []
        
        for cell in cells:
            # Structural tag
            tag_start = HEADER_START if cell.is_header else CELL_START
            row_tokens.append(tag_start)
            
            # Bounding box tokens (spatial tokens)
            bbox_tokens = self.discretizer.continuous_to_tokens(
                cell.bbox, table.image_width, table.image_height
            )
            row_tokens.extend(bbox_tokens)
            
            # Content tokens (character-level)
            content_tokens = list(cell.content)
            row_tokens.extend(content_tokens)
            
            # Closing tag
            tag_end = HEADER_END if cell.is_header else CELL_END
            row_tokens.append(tag_end)
            
            # Separator
            row_tokens.append(SEP_TOKEN)
        
        return row_tokens
    
    def create_vocabulary(self, sequences: Optional[List[List[str]]] = None) -> Dict[str, int]:
        """Create vocabulary from sequences or use default"""
        vocab = {
            PAD_TOKEN: 0,
            BOS_TOKEN: 1,
            EOS_TOKEN: 2,
            SEP_TOKEN: 3,
            TABLE_START: 4,
            TABLE_END: 5,
            ROW_START: 6,
            ROW_END: 7,
            CELL_START: 8,
            CELL_END: 9,
            HEADER_START: 10,
            HEADER_END: 11,
            XMIN_TOKEN: 12,
            YMIN_TOKEN: 13,
            XMAX_TOKEN: 14,
            YMAX_TOKEN: 15,
        }
        
        # Add coordinate tokens
        for i in range(self.grid_width):
            vocab[f"{XMIN_TOKEN}{i}"] = len(vocab)
            vocab[f"{XMAX_TOKEN}{i}"] = len(vocab)
        
        for i in range(self.grid_height):
            vocab[f"{YMIN_TOKEN}{i}"] = len(vocab)
            vocab[f"{YMAX_TOKEN}{i}"] = len(vocab)
        
        # Add character tokens (ASCII printable + common unicode)
        for i in range(32, 127):  # ASCII printable
            vocab[chr(i)] = len(vocab)
        
        # Add common unicode characters
        for char in "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ":
            vocab[char] = len(vocab)
        
        # Add tokens from sequences if provided
        if sequences:
            for seq in sequences:
                for token in seq:
                    if token not in vocab:
                        vocab[token] = len(vocab)
        
        return vocab
    
    def tokens_to_ids(self, tokens: List[str], vocab: Dict[str, int]) -> List[int]:
        """Convert tokens to token IDs"""
        return [vocab.get(token, vocab[PAD_TOKEN]) for token in tokens]
    
    def ids_to_tokens(self, ids: List[int], id_to_token: Dict[int, str]) -> List[str]:
        """Convert token IDs back to tokens"""
        return [id_to_token.get(id, PAD_TOKEN) for id in ids]


