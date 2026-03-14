"""
Dataset classes for table recognition
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from .serialization import SequenceSerializer, TableData, CellData


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to pad sequences to same length
    """
    # Find max sequence length
    max_len = max(
        max(item["input_ids"].shape[0], item["token_ids"].shape[0])
        for item in batch
    )
    
    # Pad all sequences
    padded_input_ids = []
    padded_token_ids = []
    padded_struct_mask = []
    padded_cont_mask = []
    padded_bboxes = []
    padded_bbox_mask = []
    images = []
    
    # Get padding token ID (should be 0 based on serialization.py)
    pad_token_id = 0  # PAD_TOKEN is always mapped to 0
    
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        
        # Pad input_ids
        pad_len = max_len - seq_len
        if pad_len > 0:
            padded_input = torch.cat([
                item["input_ids"],
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ])
        else:
            padded_input = item["input_ids"]
        padded_input_ids.append(padded_input)
        
        # Pad token_ids
        if pad_len > 0:
            padded_token = torch.cat([
                item["token_ids"],
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ])
        else:
            padded_token = item["token_ids"]
        padded_token_ids.append(padded_token)
        
        # Pad masks
        if pad_len > 0:
            padded_struct = torch.cat([
                item["structure_mask"],
                torch.zeros(pad_len, dtype=torch.bool)
            ])
            padded_cont = torch.cat([
                item["content_mask"],
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        else:
            padded_struct = item["structure_mask"]
            padded_cont = item["content_mask"]
        padded_struct_mask.append(padded_struct)
        padded_cont_mask.append(padded_cont)
        
        # Pad bboxes if present
        if "bboxes" in item:
            bbox_len = item["bboxes"].shape[0]
            bbox_pad_len = max_len - bbox_len
            if bbox_pad_len > 0:
                padded_bbox = torch.cat([
                    item["bboxes"],
                    torch.zeros((bbox_pad_len, 4), dtype=torch.float32)
                ])
                padded_bbox_m = torch.cat([
                    item["bbox_mask"],
                    torch.zeros(bbox_pad_len, dtype=torch.bool)
                ])
            else:
                padded_bbox = item["bboxes"]
                padded_bbox_m = item["bbox_mask"]
            padded_bboxes.append(padded_bbox)
            padded_bbox_mask.append(padded_bbox_m)
        
        images.append(item["image"])
    
    result = {
        "image": torch.stack(images),
        "input_ids": torch.stack(padded_input_ids),
        "token_ids": torch.stack(padded_token_ids),
        "structure_mask": torch.stack(padded_struct_mask),
        "content_mask": torch.stack(padded_cont_mask),
    }
    
    if padded_bboxes:
        result["bboxes"] = torch.stack(padded_bboxes)
        result["bbox_mask"] = torch.stack(padded_bbox_mask)
    
    return result


class TableDataset(Dataset):
    """
    Dataset for table recognition
    
    Supports two input formats:
    1. JSON file/directory with full data (legacy format)
    2. List of (image_path, label_path) tuples (simplified format)
    
    Format 1 (legacy):
    {
        "image_path": "path/to/image.jpg",
        "table": {
            "cells": [...],
            "image_width": 1024,
            "image_height": 1280
        }
    }
    
    Format 2 (simplified):
    List of tuples: [(image_path, label_path), ...]
    where label_path points to JSON with table structure
    """
    def __init__(
        self,
        data_path: str,
        vocab: Optional[Dict[str, int]] = None,
        image_size: Tuple[int, int] = (512, 640),
        augment: bool = False,
        use_simplified_format: bool = False,
    ):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.augment = augment
        self.use_simplified_format = use_simplified_format
        
        # Load data
        if use_simplified_format:
            # Format 2: List of (image_path, label_path) tuples
            if self.data_path.is_file():
                with open(self.data_path, 'r') as f:
                    self.data_pairs = json.load(f)
            else:
                raise ValueError(f"Simplified format requires a JSON file, got: {data_path}")
            
            # Load all labels to build vocabulary
            self.data = []
            for image_path, label_path in self.data_pairs:
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
                    # Add image_path to label data for compatibility
                    label_data['image_path'] = image_path
                    self.data.append(label_data)
        else:
            # Format 1: Legacy format (JSON file or directory)
            if self.data_path.is_file():
                with open(self.data_path, 'r') as f:
                    self.data = json.load(f)
                    if not isinstance(self.data, list):
                        self.data = [self.data]
            elif self.data_path.is_dir():
                # Load all JSON files in directory
                self.data = []
                for json_file in self.data_path.glob("*.json"):
                    with open(json_file, 'r') as f:
                        self.data.append(json.load(f))
            else:
                raise ValueError(f"Invalid data path: {data_path}")
            
            self.data_pairs = None
        
        # Initialize serializer
        self.serializer = SequenceSerializer()
        
        # Build vocabulary
        if vocab is None:
            # Create vocabulary from all sequences
            sequences = []
            for item in self.data:
                table = self._load_table(item)
                seq = self.serializer.serialize_table(table)
                sequences.append(seq)
            self.vocab = self.serializer.create_vocabulary(sequences)
        else:
            self.vocab = vocab
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def _load_table(self, item: Dict) -> TableData:
        """Load table data from item"""
        table_info = item["table"]
        cells = []
        
        for cell_info in table_info["cells"]:
            cell = CellData(
                content=cell_info["content"],
                bbox=tuple(cell_info["bbox"]),
                is_header=cell_info.get("is_header", False)
            )
            cells.append(cell)
        
        return TableData(
            cells=cells,
            image_width=table_info["image_width"],
            image_height=table_info["image_height"]
        )
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        image = Image.open(image_path).convert("RGB")
        
        # Resize
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # Convert to tensor and normalize
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image
    
    def _create_masks(self, tokens: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create structure and content masks"""
        structure_tokens = {
            "<table>", "</table>", "<tr>", "</tr>", "<td>", "</td>",
            "<th>", "</th>", "<Sep>", "<Xmin>", "<Ymin>", "<Xmax>", "<Ymax>"
        }
        
        struct_mask = torch.zeros(len(tokens), dtype=torch.bool)
        cont_mask = torch.zeros(len(tokens), dtype=torch.bool)
        
        for i, token in enumerate(tokens):
            if any(token.startswith(st) for st in structure_tokens):
                struct_mask[i] = True
            else:
                cont_mask[i] = True
        
        return struct_mask, cont_mask
    
    def _extract_bboxes(self, tokens: List[str], table: TableData) -> Optional[torch.Tensor]:
        """Extract bounding boxes from tokens"""
        from .serialization import XMIN_TOKEN, YMIN_TOKEN, XMAX_TOKEN, YMAX_TOKEN
        
        bboxes = []
        bbox_mask = []
        
        i = 0
        while i < len(tokens):
            if tokens[i] in ["<td>", "<th>"]:
                # Extract bbox tokens
                if (i + 4 < len(tokens) and
                    tokens[i+1].startswith(XMIN_TOKEN) and
                    tokens[i+2].startswith(YMIN_TOKEN) and
                    tokens[i+3].startswith(XMAX_TOKEN) and
                    tokens[i+4].startswith(YMAX_TOKEN)):
                    
                    xmin = int(tokens[i+1].replace(XMIN_TOKEN, ""))
                    ymin = int(tokens[i+2].replace(YMIN_TOKEN, ""))
                    xmax = int(tokens[i+3].replace(XMAX_TOKEN, ""))
                    ymax = int(tokens[i+4].replace(YMAX_TOKEN, ""))
                    
                    # Normalize to [0, 1]
                    x = (xmin + xmax) / 2.0 / self.serializer.grid_width
                    y = (ymin + ymax) / 2.0 / self.serializer.grid_height
                    w = (xmax - xmin) / self.serializer.grid_width
                    h = (ymax - ymin) / self.serializer.grid_height
                    
                    bboxes.append([x, y, w, h])
                    bbox_mask.append(True)
                    i += 5
                    continue
            
            bboxes.append([0, 0, 0, 0])
            bbox_mask.append(False)
            i += 1
        
        if len(bboxes) > 0:
            return torch.tensor(bboxes, dtype=torch.float32), torch.tensor(bbox_mask, dtype=torch.bool)
        return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        if self.use_simplified_format:
            # Format 2: Get image and label paths from pairs
            image_path, label_path = self.data_pairs[idx]
            
            # Load label
            with open(label_path, 'r') as f:
                item = json.load(f)
        else:
            # Format 1: Legacy format
            item = self.data[idx]
            image_path = item["image_path"]
        
        # Load image
        image = self._load_image(image_path)
        
        # Load table and serialize
        table = self._load_table(item)
        tokens = self.serializer.serialize_table(table)
        
        # Convert to token IDs
        token_ids = self.serializer.tokens_to_ids(tokens, self.vocab)
        
        # Create right-shifted input (for training)
        input_ids = [self.vocab["<BOS>"]]
        input_ids.extend(token_ids[:-1])
        
        # Create masks
        struct_mask, cont_mask = self._create_masks(tokens)
        
        # Extract bboxes
        bbox_data = self._extract_bboxes(tokens, table)
        
        result = {
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "structure_mask": struct_mask,
            "content_mask": cont_mask,
        }
        
        if bbox_data is not None:
            bboxes, bbox_mask = bbox_data
            result["bboxes"] = bboxes
            result["bbox_mask"] = bbox_mask
        
        return result

