"""
Training script for table recognition model
"""
import argparse
import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path

from tsr.models.model import TableRecognitionModel
from tsr.data.dataset import TableDataset
from tsr.losses.losses import MultiTaskLoss
from tsr.training.trainer import Trainer


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, vocab_size: int):
    """Create model from configuration"""
    model = TableRecognitionModel(
        vocab_size=vocab_size,
        encoder_backbone=config.get("encoder_backbone", "swin_b"),
        embed_dim=config.get("embed_dim", 768),
        decoder_layers=config.get("decoder_layers", 6),
        decoder_heads=config.get("decoder_heads", 8),
        ffn_dim=config.get("ffn_dim", 3072),
        dropout=config.get("dropout", 0.1),
        use_html_refiner=config.get("use_html_refiner", True),
        use_gc_attention=config.get("use_gc_attention", True),
        token_compression=config.get("token_compression", None),
        use_hybrid_regression=config.get("use_hybrid_regression", True),
        use_parallel_decoder=config.get("use_parallel_decoder", False),
    )
    return model


def create_dataloaders(config: dict):
    """Create data loaders"""
    train_dataset = TableDataset(
        data_path=config["data"]["train_path"],
        image_size=tuple(config["data"]["image_size"]),
        augment=config["data"].get("augment", False),
    )
    
    val_dataset = None
    if "val_path" in config["data"]:
        val_dataset = TableDataset(
            data_path=config["data"]["val_path"],
            vocab=train_dataset.vocab,  # Use same vocabulary
            image_size=tuple(config["data"]["image_size"]),
            augment=False,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"].get("num_workers", 4),
            pin_memory=True,
        )
    
    return train_loader, val_loader, train_dataset.vocab


def main():
    parser = argparse.ArgumentParser(description="Train table recognition model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader, vocab = create_dataloaders(config)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    print("Creating model...")
    model = create_model(config, vocab_size)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    loss_config = config.get("loss", {})
    criterion = MultiTaskLoss(
        lambda_struc=loss_config.get("lambda_struc", 1.0),
        lambda_cont=loss_config.get("lambda_cont", 1.0),
        lambda_l1=loss_config.get("lambda_l1", 1.0),
        lambda_iou=loss_config.get("lambda_iou", 1.0),
        lambda_consistency=loss_config.get("lambda_consistency", 0.1),
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        save_dir=config["training"]["save_dir"],
        log_interval=config["training"].get("log_interval", 100),
    )
    
    # Train
    trainer.train(
        num_epochs=config["training"]["num_epochs"],
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()


