#!/usr/bin/env python3
"""
Run experiments to validate improvements from Foundation to Improvement Phase
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from experiments.experiment_framework import (
    ExperimentRunner,
    create_foundation_experiments,
    create_improvement_experiments,
)
from tsr.data.dataset import TableDataset


def main():
    parser = argparse.ArgumentParser(description="Run table recognition experiments")
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
        "--foundation_only",
        action="store_true",
        help="Run only foundation phase experiments"
    )
    parser.add_argument(
        "--improvement_only",
        action="store_true",
        help="Run only improvement phase experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Run specific experiment by name"
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
        default=5,
        help="Number of training epochs"
    )
    
    args = parser.parse_args()
    
    # Create data loaders
    print("Loading datasets...")
    train_dataset = TableDataset(
        data_path=args.data_path,
        image_size=(512, 640),
        augment=False,
    )
    
    val_dataset = None
    if args.val_path:
        val_dataset = TableDataset(
            data_path=args.val_path,
            vocab=train_dataset.vocab,
            image_size=(512, 640),
            augment=False,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if args.device == "cuda" else False,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if args.device == "cuda" else False,
        )
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    # Get experiments to run
    all_experiments = []
    
    if not args.improvement_only:
        foundation_experiments = create_foundation_experiments()
        for exp in foundation_experiments:
            exp.batch_size = args.batch_size
            exp.num_epochs = args.num_epochs
        all_experiments.extend(foundation_experiments)
    
    if not args.foundation_only:
        improvement_experiments = create_improvement_experiments()
        for exp in improvement_experiments:
            exp.batch_size = args.batch_size
            exp.num_epochs = args.num_epochs
        all_experiments.extend(improvement_experiments)
    
    # Filter by specific experiment if requested
    if args.experiment:
        all_experiments = [exp for exp in all_experiments if exp.name == args.experiment]
        if not all_experiments:
            print(f"Error: Experiment '{args.experiment}' not found")
            return
    
    print(f"\nRunning {len(all_experiments)} experiment(s)...")
    
    # Run experiments
    for exp_config in all_experiments:
        try:
            runner.run_experiment(
                config=exp_config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=args.device,
            )
        except Exception as e:
            print(f"Error running experiment {exp_config.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results and generate report
    if runner.results:
        runner.save_results()
        runner.generate_comparison_report()
        print("\n" + "="*60)
        print("All experiments completed!")
        print("="*60)


if __name__ == "__main__":
    main()


