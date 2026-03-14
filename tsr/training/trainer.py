"""
Training utilities for table recognition model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import os
from pathlib import Path


class Trainer:
    """Trainer for table recognition model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        log_interval: int = 100,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        # Loss function
        self.criterion = criterion
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        loss_dict = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            
            # Prepare targets
            targets = {
                "token_ids": batch["token_ids"].to(self.device),
                "structure_mask": batch["structure_mask"].to(self.device),
                "content_mask": batch["content_mask"].to(self.device),
            }
            
            if "bboxes" in batch:
                targets["bboxes"] = batch["bboxes"].to(self.device)
                targets["bbox_mask"] = batch["bbox_mask"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                images,
                input_ids=input_ids,
                return_regression="bboxes" in targets
            )
            
            # Calculate loss
            if self.criterion is not None:
                loss_dict_batch = self.criterion(outputs, targets)
                loss = loss_dict_batch["total_loss"]
            else:
                # Simple CE loss
                loss = nn.functional.cross_entropy(
                    outputs["logits"].view(-1, outputs["logits"].size(-1)),
                    targets["token_ids"].view(-1),
                    ignore_index=0
                )
                loss_dict_batch = {"total_loss": loss}
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for k, v in loss_dict_batch.items():
                if isinstance(v, torch.Tensor):
                    loss_dict[k] = loss_dict.get(k, 0.0) + v.item()
                else:
                    loss_dict[k] = loss_dict.get(k, 0.0) + v
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "lr": self.optimizer.param_groups[0]["lr"]
            })
            
            self.global_step += 1
            
            # Logging
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"\nStep {self.global_step}, Loss: {avg_loss:.4f}")
        
        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / num_batches
        
        loss_dict["total_loss"] = avg_loss
        
        return loss_dict
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        loss_dict = {}
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            
            targets = {
                "token_ids": batch["token_ids"].to(self.device),
                "structure_mask": batch["structure_mask"].to(self.device),
                "content_mask": batch["content_mask"].to(self.device),
            }
            
            if "bboxes" in batch:
                targets["bboxes"] = batch["bboxes"].to(self.device)
                targets["bbox_mask"] = batch["bbox_mask"].to(self.device)
            
            outputs = self.model(
                images,
                input_ids=input_ids,
                return_regression="bboxes" in targets
            )
            
            if self.criterion is not None:
                loss_dict_batch = self.criterion(outputs, targets)
                loss = loss_dict_batch["total_loss"]
            else:
                loss = nn.functional.cross_entropy(
                    outputs["logits"].view(-1, outputs["logits"].size(-1)),
                    targets["token_ids"].view(-1),
                    ignore_index=0
                )
                loss_dict_batch = {"total_loss": loss}
            
            total_loss += loss.item()
            for k, v in loss_dict_batch.items():
                if isinstance(v, torch.Tensor):
                    loss_dict[k] = loss_dict.get(k, 0.0) + v.item()
                else:
                    loss_dict[k] = loss_dict.get(k, 0.0) + v
        
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / num_batches
        
        loss_dict["total_loss"] = avg_loss
        
        return loss_dict
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / "latest.pth")
        
        # Save epoch checkpoint
        torch.save(checkpoint, self.save_dir / f"epoch_{epoch}.pth")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / "best.pth")
            print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        return checkpoint.get("epoch", 0)
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Main training loop"""
        start_epoch = 0
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resumed training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_losses = self.train_epoch(epoch)
            print(f"\nTrain Losses: {train_losses}")
            
            # Validate
            if self.val_loader is not None:
                val_losses = self.validate()
                print(f"Val Losses: {val_losses}")
                
                # Check if best
                is_best = val_losses["total_loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses["total_loss"]
            else:
                is_best = False
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Update learning rate
            self.scheduler.step()


