"""
Experiment Framework for Table Recognition System
Validates improvements from Foundation Phase to Improvement Phase
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from tsr.models.model import TableRecognitionModel
from tsr.losses.losses import MultiTaskLoss
from tsr.data.dataset import TableDataset


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    phase: str  # "foundation" or "improvement"
    encoder_backbone: str = "resnet31"  # Use resnet31 for variable sizes
    embed_dim: int = 768
    decoder_layers: int = 6
    decoder_heads: int = 8
    ffn_dim: int = 3072
    dropout: float = 0.1
    
    # Phase 1: Foundation
    use_unified_ce_loss: bool = True
    
    # Phase 2: Improvements
    use_hybrid_regression: bool = False
    use_parallel_decoder: bool = False
    use_html_refiner: bool = False
    use_gc_attention: bool = False
    token_compression: Optional[float] = None
    
    # Training
    batch_size: int = 4
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Loss weights
    lambda_struc: float = 1.0
    lambda_cont: float = 1.0
    lambda_l1: float = 1.0
    lambda_iou: float = 1.0
    lambda_consistency: float = 0.1


@dataclass
class ExperimentResults:
    """Results from an experiment"""
    config: ExperimentConfig
    train_losses: List[float]
    val_losses: List[float]
    training_time: float
    inference_time: float
    model_size_mb: float
    num_parameters: int
    memory_usage_mb: float


class ExperimentRunner:
    """Runs experiments and tracks results"""
    
    def __init__(self, output_dir: str = "./experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def create_model(self, config: ExperimentConfig, vocab_size: int) -> TableRecognitionModel:
        """Create model from configuration"""
        return TableRecognitionModel(
            vocab_size=vocab_size,
            encoder_backbone=config.encoder_backbone,
            embed_dim=config.embed_dim,
            decoder_layers=config.decoder_layers,
            decoder_heads=config.decoder_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            use_html_refiner=config.use_html_refiner,
            use_gc_attention=config.use_gc_attention,
            token_compression=config.token_compression,
            use_hybrid_regression=config.use_hybrid_regression,
            use_parallel_decoder=config.use_parallel_decoder,
        )
    
    def create_loss_function(self, config: ExperimentConfig):
        """Create loss function from configuration"""
        if config.use_unified_ce_loss:
            return MultiTaskLoss(
                lambda_struc=config.lambda_struc,
                lambda_cont=config.lambda_cont,
                lambda_l1=config.lambda_l1 if config.use_hybrid_regression else 0.0,
                lambda_iou=config.lambda_iou if config.use_hybrid_regression else 0.0,
                lambda_consistency=config.lambda_consistency if config.use_hybrid_regression else 0.0,
            )
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self, model, train_loader, optimizer, criterion, device, config):
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            optimizer.zero_grad()
            
            if isinstance(criterion, MultiTaskLoss):
                outputs = model(images, input_ids=input_ids, return_regression=config.use_hybrid_regression)
                targets = {
                    "token_ids": batch["token_ids"].to(device),
                    "structure_mask": batch["structure_mask"].to(device),
                    "content_mask": batch["content_mask"].to(device),
                }
                if "bboxes" in batch and config.use_hybrid_regression:
                    targets["bboxes"] = batch["bboxes"].to(device)
                    targets["bbox_mask"] = batch["bbox_mask"].to(device)
                
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["total_loss"]
            else:
                outputs = model(images, input_ids=input_ids)
                loss = criterion(
                    outputs["logits"].view(-1, outputs["logits"].size(-1)),
                    batch["token_ids"].view(-1).to(device)
                )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, model, val_loader, criterion, device, config):
        """Validate model"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            if isinstance(criterion, MultiTaskLoss):
                outputs = model(images, input_ids=input_ids, return_regression=config.use_hybrid_regression)
                targets = {
                    "token_ids": batch["token_ids"].to(device),
                    "structure_mask": batch["structure_mask"].to(device),
                    "content_mask": batch["content_mask"].to(device),
                }
                if "bboxes" in batch and config.use_hybrid_regression:
                    targets["bboxes"] = batch["bboxes"].to(device)
                    targets["bbox_mask"] = batch["bbox_mask"].to(device)
                
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["total_loss"]
            else:
                outputs = model(images, input_ids=input_ids)
                loss = criterion(
                    outputs["logits"].view(-1, outputs["logits"].size(-1)),
                    batch["token_ids"].view(-1).to(device)
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def measure_inference_time(self, model, sample_batch, device, num_runs: int = 10):
        """Measure inference time"""
        model.eval()
        images = sample_batch["image"].to(device)
        input_ids = sample_batch["input_ids"].to(device)
        
        # Warmup
        for _ in range(3):
            _ = model(images, input_ids=input_ids)
        
        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(num_runs):
            _ = model(images, input_ids=input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        return (end_time - start_time) / num_runs
    
    def get_model_size(self, model):
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: str = "cuda",
    ) -> ExperimentResults:
        """Run a single experiment"""
        print(f"\n{'='*60}")
        print(f"Running Experiment: {config.name}")
        print(f"Phase: {config.phase}")
        print(f"{'='*60}")
        
        # Get vocab size from dataset
        vocab_size = len(train_loader.dataset.vocab)
        
        # Create model
        model = self.create_model(config, vocab_size)
        model = model.to(device)
        
        # Create loss function
        criterion = self.create_loss_function(config)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Training
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        for epoch in range(config.num_epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, device, config)
            train_losses.append(train_loss)
            
            if val_loader:
                val_loss = self.validate(model, val_loader, criterion, device, config)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{config.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{config.num_epochs}: Train Loss: {train_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Measure inference time
        sample_batch = next(iter(train_loader))
        inference_time = self.measure_inference_time(model, sample_batch, device)
        
        # Get model statistics
        model_size_mb = self.get_model_size(model)
        num_parameters = sum(p.numel() for p in model.parameters())
        
        # Memory usage (approximate)
        if torch.cuda.is_available():
            memory_usage_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            memory_usage_mb = 0.0
        
        results = ExperimentResults(
            config=config,
            train_losses=train_losses,
            val_losses=val_losses,
            training_time=training_time,
            inference_time=inference_time,
            model_size_mb=model_size_mb,
            num_parameters=num_parameters,
            memory_usage_mb=memory_usage_mb,
        )
        
        self.results.append(results)
        
        print(f"\nResults:")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Inference Time: {inference_time*1000:.2f}ms")
        print(f"  Model Size: {model_size_mb:.2f} MB")
        print(f"  Parameters: {num_parameters:,}")
        print(f"  Final Train Loss: {train_losses[-1]:.4f}")
        if val_losses:
            print(f"  Final Val Loss: {val_losses[-1]:.4f}")
        
        return results
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save all results to JSON"""
        results_dict = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert config to dict
            result_dict["config"] = asdict(result.config)
            results_dict.append(result_dict)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def generate_comparison_report(self, filename: str = "comparison_report.md"):
        """Generate a comparison report"""
        if len(self.results) < 2:
            print("Need at least 2 experiments for comparison")
            return
        
        report = "# Experiment Comparison Report\n\n"
        report += "## Overview\n\n"
        report += "This report compares experiments from Foundation Phase to Improvement Phase.\n\n"
        
        # Create comparison table
        report += "## Results Comparison\n\n"
        report += "| Experiment | Phase | Final Train Loss | Final Val Loss | Training Time (s) | Inference Time (ms) | Model Size (MB) | Parameters |\n"
        report += "|------------|-------|------------------|----------------|-------------------|---------------------|-----------------|-------------|\n"
        
        for result in self.results:
            val_loss = result.val_losses[-1] if result.val_losses else "N/A"
            report += f"| {result.config.name} | {result.config.phase} | {result.train_losses[-1]:.4f} | {val_loss} | {result.training_time:.2f} | {result.inference_time*1000:.2f} | {result.model_size_mb:.2f} | {result.num_parameters:,} |\n"
        
        # Improvement analysis
        if len(self.results) >= 2:
            foundation = self.results[0]
            improvements = self.results[1:]
            
            report += "\n## Improvement Analysis\n\n"
            
            for imp in improvements:
                report += f"### {imp.config.name} vs {foundation.config.name}\n\n"
                
                # Loss improvement
                loss_improvement = ((foundation.train_losses[-1] - imp.train_losses[-1]) / foundation.train_losses[-1]) * 100
                report += f"- **Loss Improvement**: {loss_improvement:+.2f}%\n"
                
                # Speed improvement
                speed_improvement = ((foundation.inference_time - imp.inference_time) / foundation.inference_time) * 100
                report += f"- **Inference Speed**: {speed_improvement:+.2f}%\n"
                
                # Model size change
                size_change = ((imp.model_size_mb - foundation.model_size_mb) / foundation.model_size_mb) * 100
                report += f"- **Model Size Change**: {size_change:+.2f}%\n"
                
                report += "\n"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Comparison report saved to {output_path}")


def create_foundation_experiments() -> List[ExperimentConfig]:
    """Create foundation phase experiments"""
    experiments = []
    
    # Phase 1.1: Basic foundation with unified CE loss
    experiments.append(ExperimentConfig(
        name="Foundation_Basic",
        phase="foundation",
        encoder_backbone="resnet31",
        use_unified_ce_loss=True,
        use_hybrid_regression=False,
        use_html_refiner=False,
        use_gc_attention=False,
        use_parallel_decoder=False,
        num_epochs=5,
    ))
    
    return experiments


def create_improvement_experiments() -> List[ExperimentConfig]:
    """Create improvement phase experiments"""
    experiments = []
    
    # Phase 2.A: Hybrid Regression
    experiments.append(ExperimentConfig(
        name="Improvement_HybridRegression",
        phase="improvement",
        encoder_backbone="resnet31",
        use_unified_ce_loss=True,
        use_hybrid_regression=True,
        use_html_refiner=False,
        use_gc_attention=False,
        use_parallel_decoder=False,
        num_epochs=5,
    ))
    
    # Phase 2.B: HTML Refiner
    experiments.append(ExperimentConfig(
        name="Improvement_HTMLRefiner",
        phase="improvement",
        encoder_backbone="resnet31",
        use_unified_ce_loss=True,
        use_hybrid_regression=True,
        use_html_refiner=True,
        use_gc_attention=False,
        use_parallel_decoder=False,
        num_epochs=5,
    ))
    
    # Phase 2.C: GCAttention
    experiments.append(ExperimentConfig(
        name="Improvement_GCAttention",
        phase="improvement",
        encoder_backbone="resnet31",
        use_unified_ce_loss=True,
        use_hybrid_regression=True,
        use_html_refiner=True,
        use_gc_attention=True,
        use_parallel_decoder=False,
        num_epochs=5,
    ))
    
    # Phase 2.D: Token Compression
    experiments.append(ExperimentConfig(
        name="Improvement_TokenCompression",
        phase="improvement",
        encoder_backbone="resnet31",
        use_unified_ce_loss=True,
        use_hybrid_regression=True,
        use_html_refiner=True,
        use_gc_attention=True,
        token_compression=0.8,  # 20% reduction
        use_parallel_decoder=False,
        num_epochs=5,
    ))
    
    # Phase 2.E: All Improvements Combined
    experiments.append(ExperimentConfig(
        name="Improvement_AllCombined",
        phase="improvement",
        encoder_backbone="resnet31",
        use_unified_ce_loss=True,
        use_hybrid_regression=True,
        use_html_refiner=True,
        use_gc_attention=True,
        token_compression=0.8,
        use_parallel_decoder=False,  # Parallel decoder needs special handling
        num_epochs=5,
    ))
    
    return experiments

