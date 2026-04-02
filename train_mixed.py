#!/usr/bin/env python3
"""
Mixed-dataset training: combine TSR and OCR datasets in one training run.

Usage examples:

  # TSR only (same as train.py via experiments)
  python train_mixed.py --tsr_data data/train_list.json --val_path data/val_list.json

  # OCR only
  python train_mixed.py --ocr_data data/ocr_train.txt

  # Mixed TSR + OCR
  python train_mixed.py --tsr_data data/train_list.json --ocr_data data/ocr_train.txt

  # Resume with vocab auto-extension
  python train_mixed.py --tsr_data data/train_list.json --ocr_data data/ocr_jp.txt \
      --resume experiment_results/checkpoints/Mixed/latest.pth
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader, ConcatDataset

from experiments.base_experiment import ExperimentConfig, run_experiment
from tsr.data.dataset import TableDataset, collate_fn
from tsr.data.ocr_dataset import OCRDataset
from tsr.utils.vocab import load_vocab_auto


def main():
    parser = argparse.ArgumentParser(description="Mixed TSR + OCR training")

    # Data sources (at least one required)
    parser.add_argument("--tsr_data", type=str, default=None,
                        help="Path to TSR training data (JSON list / dir)")
    parser.add_argument("--ocr_data", type=str, nargs="+", default=None,
                        help="Path(s) to OCR training data (pipe-delimited txt)")
    parser.add_argument("--ocr_base_dir", type=str, default=None,
                        help="Base directory for resolving OCR image paths")
    parser.add_argument("--val_path", type=str, default=None,
                        help="Path to validation data (TSR format)")

    # Vocab
    parser.add_argument("--vocab", type=str, default=None,
                        help="Path to initial vocab (.txt or .json)")

    # Model / training
    parser.add_argument("--output_dir", type=str, default="./experiment_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="Mixed_TSR_OCR")
    parser.add_argument("--debug", action="store_true",
                        help="Limit each dataset to 100 samples")

    args = parser.parse_args()

    if not args.tsr_data and not args.ocr_data:
        parser.error("At least one of --tsr_data or --ocr_data is required")

    max_samples = 100 if args.debug else None
    image_size = (512, 640)

    # --- 1. Load initial vocab ------------------------------------------
    vocab = None
    if args.vocab:
        vocab = load_vocab_auto(args.vocab)
        print(f"Loaded initial vocab ({len(vocab)} tokens) from {args.vocab}")

    # --- 2. Build datasets ----------------------------------------------
    datasets = []

    tsr_dataset = None
    if args.tsr_data:
        use_simplified = "dataset_list" in args.tsr_data
        tsr_dataset = TableDataset(
            data_path=args.tsr_data,
            vocab=vocab,
            image_size=image_size,
            augment=False,
            use_simplified_format=use_simplified,
            max_samples=max_samples,
        )
        vocab = tsr_dataset.vocab
        datasets.append(tsr_dataset)
        print(f"TSR dataset: {len(tsr_dataset)} samples")

    # If no TSR dataset built the vocab, create a minimal base vocab
    if vocab is None:
        from tsr.data.serialization import SequenceSerializer
        vocab = SequenceSerializer().create_vocabulary()

    ocr_datasets = []
    if args.ocr_data:
        for ocr_path in args.ocr_data:
            ocr_ds = OCRDataset(
                data_path=ocr_path,
                vocab=vocab,
                image_size=image_size,
                max_samples=max_samples,
                base_dir=args.ocr_base_dir,
            )
            ocr_datasets.append(ocr_ds)
            datasets.append(ocr_ds)
            print(f"OCR dataset ({ocr_path}): {len(ocr_ds)} samples")

    # After OCR datasets may have extended vocab, sync back to TSR dataset
    if tsr_dataset is not None and len(vocab) != len(tsr_dataset.vocab):
        tsr_dataset.update_vocab(vocab)

    print(f"Final vocabulary size: {len(vocab)} tokens")
    print(f"Total training samples: {sum(len(d) for d in datasets)}")

    # --- 3. Combined loader ---------------------------------------------
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = ConcatDataset(datasets)

    train_loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # --- 4. Validation loader (optional, TSR-only) ----------------------
    val_loader = None
    if args.val_path:
        use_simplified = "dataset_list" in args.val_path
        val_dataset = TableDataset(
            data_path=args.val_path,
            vocab=vocab,
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

    # --- 5. Config & run ------------------------------------------------
    config = ExperimentConfig(
        name=args.experiment_name,
        phase="improvement",
        encoder_backbone="resnet31",
        embed_dim=512,
        decoder_layers=4,
        decoder_heads=8,
        ffn_dim=2048,
        dropout=0.1,
        use_unified_ce_loss=True,
        use_hybrid_regression=bool(args.tsr_data),
        use_html_refiner=True,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=4,
        use_mixed_precision=True,
        gradient_checkpointing=True,
        image_size=image_size,
    )

    run_experiment(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        output_dir=args.output_dir,
        vocab=vocab,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
