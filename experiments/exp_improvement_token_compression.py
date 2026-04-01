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
from experiments.base_experiment import ExperimentConfig, run_experiment, run_validation_only
from tsr.data.dataset import TableDataset, collate_fn
from tsr.utils.vocab import load_vocab_auto


EXPERIMENT_NAME = "Improvement_TokenCompression"


def main():
    parser = argparse.ArgumentParser(description="Token Compression Improvement Experiment")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training data (JSON file or directory). Required unless --validate_only."
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Path to validation data"
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
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run validation only (requires --val_path)"
    )
    parser.add_argument(
        "--num_inference_samples",
        type=int,
        default=3,
        help="Number of random inference samples to display during validation-only mode"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: limit dataset to 100 samples for quick testing"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Path to vocab file (.txt or .json) to use instead of building from data"
    )

    args = parser.parse_args()

    max_samples = 100 if args.debug else None
    if args.debug:
        print("[Debug mode] Limiting datasets to 100 samples")

    image_size = (512, 640)

    config = ExperimentConfig(
        name=EXPERIMENT_NAME,
        phase="improvement",
        encoder_backbone="resnet31",
        embed_dim=512,
        decoder_layers=4,
        decoder_heads=8,
        ffn_dim=2048,
        dropout=0.1,
        use_unified_ce_loss=True,
        use_hybrid_regression=True,
        use_html_refiner=True,
        use_gc_attention=True,
        use_parallel_decoder=False,
        token_compression=args.compression_ratio,
        batch_size=args.batch_size if args.batch_size else 2,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=4,
        use_mixed_precision=True,
        gradient_checkpointing=True,
        image_size=image_size,
    )

    ext_vocab = None
    if args.vocab:
        ext_vocab = load_vocab_auto(args.vocab)
        print(f"Loaded external vocabulary ({len(ext_vocab)} tokens) from {args.vocab}")

    if args.validate_only:
        if not args.val_path:
            parser.error("--val_path is required for --validate_only")

        checkpoint = args.resume
        if checkpoint is None:
            checkpoint = str(Path(args.output_dir) / "checkpoints" / EXPERIMENT_NAME / "latest.pth")
            print(f"No --resume specified, using default: {checkpoint}")

        use_simplified = Path(args.val_path).name == "dataset_list.json" or "dataset_list" in args.val_path

        print("Loading validation dataset...")
        val_dataset = TableDataset(
            data_path=args.val_path,
            vocab=ext_vocab,
            image_size=image_size,
            augment=False,
            use_simplified_format=use_simplified,
            max_samples=max_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Vocabulary size: {len(val_dataset.vocab)}")

        run_validation_only(
            config=config,
            val_loader=val_loader,
            device=args.device,
            output_dir=args.output_dir,
            vocab=val_dataset.vocab,
            checkpoint_path=checkpoint,
            num_inference_samples=args.num_inference_samples,
        )
    else:
        if not args.data_path:
            parser.error("--data_path is required for training")

        use_simplified = Path(args.data_path).name == "dataset_list.json" or "dataset_list" in args.data_path

        print("Loading datasets...")
        train_dataset = TableDataset(
            data_path=args.data_path,
            vocab=ext_vocab,
            image_size=image_size,
            augment=False,
            use_simplified_format=use_simplified,
            max_samples=max_samples,
        )

        val_dataset = None
        if args.val_path:
            val_dataset = TableDataset(
                data_path=args.val_path,
                vocab=train_dataset.vocab,
                image_size=image_size,
                augment=False,
                use_simplified_format=use_simplified,
                max_samples=max_samples,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=False,
                collate_fn=collate_fn,
            )

        print(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation samples: {len(val_dataset)}")
        print(f"Vocabulary size: {len(train_dataset.vocab)}")

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
