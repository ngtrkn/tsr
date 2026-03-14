"""
Loss Functions for Multi-Task Learning
Includes: CE loss, L1 loss, IoU loss, Column Consistency loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def calculate_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    format: str = "xywh"
) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) for bounding boxes
    
    Args:
        pred_boxes: (B, N, 4) predicted boxes
        target_boxes: (B, N, 4) target boxes
        format: "xywh" or "xyxy"
    Returns:
        (B, N) IoU scores
    """
    if format == "xywh":
        # Convert to xyxy
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
        
        target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
    else:  # xyxy
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(-1)
    
    # Calculate intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    return iou


class UnifiedCELoss(nn.Module):
    """
    Unified Cross-Entropy Loss for entire sequence
    Treats spatial tokens and text characters with equal priority
    """
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, T, vocab_size) prediction logits
            targets: (B, T) target token IDs
        Returns:
            Scalar loss
        """
        B, T, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        return self.ce_loss(logits_flat, targets_flat)


class HybridRegressionLoss(nn.Module):
    """
    Hybrid regression loss combining L1 and IoU
    """
    def __init__(self, l1_weight: float = 1.0, iou_weight: float = 1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.iou_weight = iou_weight
        self.l1_loss = nn.L1Loss(reduction='mean')
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_boxes: (B, T, 4) normalized (x, y, w, h)
            target_boxes: (B, T, 4) normalized (x, y, w, h)
            valid_mask: (B, T) mask for valid boxes
        Returns:
            (total_loss, l1_loss, iou_loss)
        """
        if valid_mask is not None:
            # Apply mask
            pred_boxes = pred_boxes[valid_mask]
            target_boxes = target_boxes[valid_mask]
        
        # L1 loss
        l1_loss = self.l1_loss(pred_boxes, target_boxes)
        
        # IoU loss (1 - IoU)
        iou = calculate_iou(pred_boxes, target_boxes, format="xywh")
        iou_loss = 1.0 - iou.mean()
        
        # Total loss
        total_loss = self.l1_weight * l1_loss + self.iou_weight * iou_loss
        
        return total_loss, l1_loss, iou_loss


class ColumnConsistencyLoss(nn.Module):
    """
    Column Consistency Loss
    Minimizes prediction variance across tokens in the same logical column
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        column_logits: torch.Tensor,
        column_assignments: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            column_logits: (B, T, max_columns) column assignment predictions
            column_assignments: (B, T) true column assignments
            valid_mask: (B, T) mask for valid tokens
        Returns:
            Scalar consistency loss
        """
        if valid_mask is not None:
            column_logits = column_logits[valid_mask]
            column_assignments = column_assignments[valid_mask]
        
        # Get column predictions
        column_probs = F.softmax(column_logits, dim=-1)  # (N, max_columns)
        
        # Group by column
        max_col = column_assignments.max().item() + 1
        consistency_loss = 0.0
        count = 0
        
        for col_id in range(max_col):
            col_mask = (column_assignments == col_id)
            if col_mask.sum() > 1:  # Need at least 2 tokens in same column
                col_probs = column_probs[col_mask]  # (K, max_columns)
                
                # Calculate variance of predictions within column
                col_mean = col_probs.mean(dim=0, keepdim=True)  # (1, max_columns)
                col_var = ((col_probs - col_mean) ** 2).mean()
                
                consistency_loss += col_var
                count += 1
        
        if count > 0:
            consistency_loss = consistency_loss / count
        else:
            consistency_loss = torch.tensor(0.0, device=column_logits.device)
        
        return consistency_loss


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss function
    L = λ₁ CE_struc + λ₂ CE_cont + λ₃ L1_bbox + λ₄ IoU + λ₅ Consistency
    """
    def __init__(
        self,
        lambda_struc: float = 1.0,
        lambda_cont: float = 1.0,
        lambda_l1: float = 1.0,
        lambda_iou: float = 1.0,
        lambda_consistency: float = 0.1,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.lambda_struc = lambda_struc
        self.lambda_cont = lambda_cont
        self.lambda_l1 = lambda_l1
        self.lambda_iou = lambda_iou
        self.lambda_consistency = lambda_consistency
        
        self.ce_loss = UnifiedCELoss(ignore_index=ignore_index)
        self.regression_loss = HybridRegressionLoss(l1_weight=1.0, iou_weight=1.0)
        self.consistency_loss = ColumnConsistencyLoss()
    
    def forward(
        self,
        outputs: dict,
        targets: dict,
    ) -> dict:
        """
        Args:
            outputs: Dictionary with:
                - logits: (B, T, vocab_size)
                - regression: (B, T, 4) optional
                - column_logits: (B, T, max_columns) optional
            targets: Dictionary with:
                - token_ids: (B, T)
                - bboxes: (B, T, 4) optional
                - column_assignments: (B, T) optional
                - structure_mask: (B, T) mask for structural tokens
                - content_mask: (B, T) mask for content tokens
        Returns:
            Dictionary with individual and total losses
        """
        logits = outputs["logits"]
        token_ids = targets["token_ids"]
        
        # Unified CE loss (treats all tokens equally)
        total_ce_loss = self.ce_loss(logits, token_ids)
        
        # Separate structure and content losses if masks provided
        if "structure_mask" in targets and "content_mask" in targets:
            struct_mask = targets["structure_mask"]
            cont_mask = targets["content_mask"]
            
            if struct_mask.sum() > 0:
                struct_logits = logits[struct_mask]
                struct_targets = token_ids[struct_mask]
                ce_struc = F.cross_entropy(
                    struct_logits.view(-1, struct_logits.size(-1)),
                    struct_targets.view(-1),
                    ignore_index=self.ce_loss.ignore_index
                )
            else:
                ce_struc = torch.tensor(0.0, device=logits.device)
            
            if cont_mask.sum() > 0:
                cont_logits = logits[cont_mask]
                cont_targets = token_ids[cont_mask]
                ce_cont = F.cross_entropy(
                    cont_logits.view(-1, cont_logits.size(-1)),
                    cont_targets.view(-1),
                    ignore_index=self.ce_loss.ignore_index
                )
            else:
                ce_cont = torch.tensor(0.0, device=logits.device)
        else:
            ce_struc = total_ce_loss
            ce_cont = total_ce_loss
        
        # Regression losses
        l1_loss = torch.tensor(0.0, device=logits.device)
        iou_loss = torch.tensor(0.0, device=logits.device)
        
        if "regression" in outputs and "bboxes" in targets:
            regression_pred = outputs["regression"]
            bboxes = targets["bboxes"]
            valid_mask = targets.get("bbox_mask", None)
            
            reg_loss, l1_loss, iou_loss = self.regression_loss(
                regression_pred, bboxes, valid_mask
            )
        else:
            reg_loss = torch.tensor(0.0, device=logits.device)
        
        # Consistency loss
        consistency_loss = torch.tensor(0.0, device=logits.device)
        
        if "column_logits" in outputs and "column_assignments" in targets:
            column_logits = outputs["column_logits"]
            column_assignments = targets["column_assignments"]
            valid_mask = targets.get("column_mask", None)
            
            consistency_loss = self.consistency_loss(
                column_logits, column_assignments, valid_mask
            )
        
        # Total loss
        total_loss = (
            self.lambda_struc * ce_struc +
            self.lambda_cont * ce_cont +
            self.lambda_l1 * l1_loss +
            self.lambda_iou * iou_loss +
            self.lambda_consistency * consistency_loss
        )
        
        return {
            "total_loss": total_loss,
            "ce_struc": ce_struc,
            "ce_cont": ce_cont,
            "l1_loss": l1_loss,
            "iou_loss": iou_loss,
            "consistency_loss": consistency_loss,
        }


