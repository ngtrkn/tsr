#!/usr/bin/env python3
"""
Quick test script to verify experiment framework works
Creates dummy data and runs a minimal experiment
"""
import torch
import json
import tempfile
from pathlib import Path
from experiments.experiment_framework import (
    ExperimentRunner,
    ExperimentConfig,
    create_foundation_experiments,
)
from tsr.data.dataset import TableDataset
from torch.utils.data import DataLoader


def create_dummy_data(num_samples: int = 10, output_dir: str = None):
    """Create dummy JSON data for testing"""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_dir = Path(output_dir)
    
    # Create dummy data
    dummy_data = []
    for i in range(num_samples):
        data = {
            "image_path": f"/dummy/image_{i}.jpg",
            "table": {
                "cells": [
                    {
                        "content": f"Cell {j}",
                        "bbox": [10 + j*50, 10, 50 + j*50, 30],
                        "is_header": j < 3
                    }
                    for j in range(5)
                ],
                "image_width": 300,
                "image_height": 100
            }
        }
        dummy_data.append(data)
    
    # Save to JSON file
    json_path = output_dir / "dummy_data.json"
    with open(json_path, 'w') as f:
        json.dump(dummy_data, f, indent=2)
    
    return str(json_path)


def main():
    print("Creating dummy data...")
    data_path = create_dummy_data(num_samples=10)
    
    print("Loading dataset...")
    dataset = TableDataset(
        data_path=data_path,
        image_size=(224, 224),  # Smaller for quick test
        augment=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues in test
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {len(dataset.vocab)}")
    
    # Create minimal experiment config
    config = ExperimentConfig(
        name="QuickTest",
        phase="foundation",
        encoder_backbone="resnet31",
        embed_dim=256,  # Smaller for quick test
        decoder_layers=2,
        decoder_heads=4,
        ffn_dim=1024,
        batch_size=2,
        num_epochs=1,  # Just 1 epoch for quick test
    )
    
    # Run experiment
    runner = ExperimentRunner(output_dir="./test_results")
    
    print("\nRunning quick test experiment...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        result = runner.run_experiment(
            config=config,
            train_loader=dataloader,
            val_loader=None,
            device=device,
        )
        
        print("\n✅ Quick test passed!")
        print(f"Final loss: {result.train_losses[-1]:.4f}")
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


