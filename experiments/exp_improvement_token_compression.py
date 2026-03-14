#!/usr/bin/env python3
"""
Experiment: Improvement - Token Compression
Phase 2, Initiative C: Architectural Optimization

Features:
- Adds token compression (20% reduction)
- Reduces vision token length
- Expected: ~20% faster inference
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from pathlib import Path
from experiments.base_experiment import ExperimentConfig, run_experiment
from tsr.data.dataset import TableDataset, collate_fn


def main():
    parser = argparse.ArgumentParser(description="Token Compression Improvement Experiment")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (JSON file or directory)"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Path to validation data (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiment_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=0.8,
        help="Token compression ratio (0.8 = 20% reduction)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., 'latest.pth', 'best.pth', 'epoch_5.pth', or full path)"
    )
    args = parser.parse_args()
    
    # Create data loaders
    print("Loading datasets...")
    
    # Check if using simplified format (dataset_list.json)
    use_simplified = Path(args.data_path).name == "dataset_list.json" or "dataset_list" in args.data_path
    
    train_dataset = TableDataset(
        data_path=args.data_path,
        image_size=(512, 640),
        augment=False,
        use_simplified_format=use_simplified,
    )
    
    val_dataset = None
    if args.val_path:
        val_dataset = TableDataset(
            data_path=args.val_path,
            vocab=train_dataset.vocab,
            image_size=(512, 640),
            augment=False,
            use_simplified_format=use_simplified,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == "cuda" else False,
        collate_fn=collate_fn,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if args.device == "cuda" else False,
            collate_fn=collate_fn,
        )
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")
    
    # Create experiment config
    config = ExperimentConfig(
        name="Improvement_TokenCompression",
        phase="improvement",
        encoder_backbone="resnet31",
        embed_dim=512,  # Reduced for 12GB VRAM
        decoder_layers=4,  # Reduced for 12GB VRAM
        decoder_heads=8,
        ffn_dim=2048,  # Reduced for 12GB VRAM
        dropout=0.1,
        use_unified_ce_loss=True,
        use_hybrid_regression=True,
        use_html_refiner=True,
        use_gc_attention=True,
        use_parallel_decoder=False,
        token_compression=args.compression_ratio,  # Enable token compression
        batch_size=args.batch_size if args.batch_size else 2,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=4,
        use_mixed_precision=True,
        gradient_checkpointing=False,
    )
    
    # Run experiment (pass vocabulary for checkpoint saving)
    run_experiment(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        output_dir=args.output_dir,
        vocab=train_dataset.vocab,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()

