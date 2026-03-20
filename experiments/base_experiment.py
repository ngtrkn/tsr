"""
Base experiment script that can be reused by individual experiment scripts
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import random
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict, field

from tsr.models.model import TableRecognitionModel
from tsr.losses.losses import MultiTaskLoss
from tsr.data.dataset import TableDataset
from tsr.data.serialization import SequenceSerializer, EOS_TOKEN
from tsr.metrics.tsr_metrics import calculate_table_metrics, tokens_to_html
from tqdm import tqdm

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    phase: str  # "foundation" or "improvement"
    encoder_backbone: str = "convstem"  # Changed to convstem (smallest) for extreme memory savings
    embed_dim: int = 384  # Further reduced from 512 for extreme memory savings
    decoder_layers: int = 3  # Further reduced from 4 for extreme memory savings
    decoder_heads: int = 6  # Reduced from 8 for memory savings
    ffn_dim: int = 1536  # Further reduced from 2048 for extreme memory savings
    dropout: float = 0.1
    
    # Phase 1: Foundation
    use_unified_ce_loss: bool = True
    
    # Phase 2: Improvements
    use_hybrid_regression: bool = False
    use_parallel_decoder: bool = False
    use_html_refiner: bool = False
    use_gc_attention: bool = False
    token_compression: Optional[float] = 0.8  # Enable token compression by default (20% reduction)
    
    # Training
    batch_size: int = 1  # Minimum batch size for extreme memory savings
    num_epochs: int = 5
    learning_rate: float = 2e-4  # Increased from 1e-4 for faster convergence
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    use_mixed_precision: bool = True  # FP16 training for memory efficiency
    gradient_checkpointing: bool = True  # Enabled by default for extreme memory savings
    image_size: Tuple[int, int] = (384, 512)  # Reduced image size for memory savings
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True  # Enable learning rate scheduling
    lr_warmup_steps: int = 500  # Number of warmup steps
    lr_scheduler_type: str = "cosine"  # "cosine", "step", or "none"
    lr_decay_factor: float = 0.1  # For step scheduler
    lr_decay_epochs: List[int] = field(default_factory=lambda: [30, 60, 90])  # For step scheduler
    
    # Loss weights
    lambda_struc: float = 1.0
    lambda_cont: float = 1.0
    lambda_l1: float = 1.0
    lambda_iou: float = 1.0
    lambda_consistency: float = 0.1


def create_model(config: ExperimentConfig, vocab_size: int) -> TableRecognitionModel:
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


def create_loss_function(config: ExperimentConfig):
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


def train_epoch(model, train_loader, optimizer, criterion, device, config, scaler=None,
                warmup_scheduler=None, global_step=0):
    """Train for one epoch with gradient accumulation and mixed precision support"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulation_steps = config.gradient_accumulation_steps
    current_step = global_step
    
    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing:
        if hasattr(model.decoder, 'gradient_checkpointing_enable'):
            model.decoder.gradient_checkpointing_enable()
        if hasattr(model.encoder, 'gradient_checkpointing_enable'):
            model.encoder.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for encoder and decoder")
    
    optimizer.zero_grad()
    
    pbar = tqdm(total=len(train_loader))
    for batch_idx, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        try:
            # Mixed precision forward pass
            if config.use_mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
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
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
            else:
                # Standard precision
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
                    loss = loss_dict["total_loss"] / accumulation_steps
                else:
                    outputs = model(images, input_ids=input_ids)
                    loss = criterion(
                        outputs["logits"].view(-1, outputs["logits"].size(-1)),
                        batch["token_ids"].view(-1).to(device)
                    ) / accumulation_steps
        
            # Backward pass
            if config.use_mixed_precision and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if config.use_mixed_precision and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Learning rate warmup (update after optimizer step)
                if warmup_scheduler is not None:
                    warmup_scheduler.step()
                
                optimizer.zero_grad()
                current_step += 1
        except:
            optimizer.zero_grad()
            current_step += 1
            continue
        
        total_loss += loss.item() * accumulation_steps  # Scale back for logging
        num_batches += 1

        # Update the progress bar manually by the desired increment
        pbar.update(1) 
        # Add a dynamic description to the bar
        pbar.set_description(f"{loss.item()}")
    
    # Handle remaining gradients if batch doesn't divide evenly
    if num_batches % accumulation_steps != 0:
        if config.use_mixed_precision and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, device, config, scaler=None, vocab: Optional[Dict[str, int]] = None):
    """Validate model with mixed precision support and calculate metrics"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Metrics accumulators
    total_tokens = 0
    correct_tokens = 0
    total_structure_tokens = 0
    correct_structure_tokens = 0
    total_content_tokens = 0
    correct_content_tokens = 0
    total_sequences = 0
    exact_match_sequences = 0
    
    # TEDS metrics
    total_teds = 0.0
    teds_count = 0
    
    pad_token_id = 0  # PAD token ID
    id_to_token = {v: k for k, v in vocab.items()} if vocab else None
    
    for batch in val_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["token_ids"].to(device)
        
        if config.use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                if isinstance(criterion, MultiTaskLoss):
                    outputs = model(images, input_ids=input_ids, return_regression=config.use_hybrid_regression)
                    targets = {
                        "token_ids": target_ids,
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
                        target_ids.view(-1)
                    )
        else:
            if isinstance(criterion, MultiTaskLoss):
                outputs = model(images, input_ids=input_ids, return_regression=config.use_hybrid_regression)
                targets = {
                    "token_ids": target_ids,
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
                    target_ids.view(-1)
                )
        
        total_loss += loss.item()
        num_batches += 1
        
        # Calculate accuracy metrics
        logits = outputs["logits"]  # (B, T, vocab_size)
        pred_ids = logits.argmax(dim=-1)  # (B, T)
        
        # Mask out padding tokens
        mask = (target_ids != pad_token_id)
        
        # Token-level accuracy (excluding padding)
        batch_correct = (pred_ids == target_ids) & mask
        batch_total = mask.sum().item()
        batch_correct_count = batch_correct.sum().item()
        
        total_tokens += batch_total
        correct_tokens += batch_correct_count
        
        # Structure and content token accuracy
        if "structure_mask" in batch:
            struct_mask = batch["structure_mask"].to(device)
            cont_mask = batch["content_mask"].to(device)
            
            # Structure tokens
            struct_batch_mask = mask & struct_mask
            struct_batch_total = struct_batch_mask.sum().item()
            struct_batch_correct = (batch_correct & struct_batch_mask).sum().item()
            
            total_structure_tokens += struct_batch_total
            correct_structure_tokens += struct_batch_correct
            
            # Content tokens
            cont_batch_mask = mask & cont_mask
            cont_batch_total = cont_batch_mask.sum().item()
            cont_batch_correct = (batch_correct & cont_batch_mask).sum().item()
            
            total_content_tokens += cont_batch_total
            correct_content_tokens += cont_batch_correct
        
        # Sequence-level exact match and TEDS calculation
        B = target_ids.shape[0]
        for b in range(B):
            seq_mask = mask[b]
            if seq_mask.sum() > 0:
                seq_pred = pred_ids[b][seq_mask]
                seq_target = target_ids[b][seq_mask]
                if torch.equal(seq_pred, seq_target):
                    exact_match_sequences += 1
                
                # Calculate TEDS for this sequence
                if vocab is not None and id_to_token is not None:
                    try:
                        # Convert to token strings
                        pred_tokens = [id_to_token.get(int(id.item()), "<Pad>") for id in seq_pred.cpu()]
                        gt_tokens = [id_to_token.get(int(id.item()), "<Pad>") for id in seq_target.cpu()]
                        
                        # Remove padding tokens
                        pred_tokens = [t for t in pred_tokens if t != "<Pad>"]
                        gt_tokens = [t for t in gt_tokens if t != "<Pad>"]
                        
                        # Calculate TEDS
                        if len(pred_tokens) > 0 and len(gt_tokens) > 0:
                            table_metrics = calculate_table_metrics(pred_tokens, gt_tokens, vocab)
                            if 'teds' in table_metrics:
                                total_teds += table_metrics['teds']
                                teds_count += 1
                    except Exception:
                        # Skip if TEDS calculation fails
                        pass
            total_sequences += 1
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    token_accuracy = (correct_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
    structure_accuracy = (correct_structure_tokens / total_structure_tokens * 100) if total_structure_tokens > 0 else 0.0
    content_accuracy = (correct_content_tokens / total_content_tokens * 100) if total_content_tokens > 0 else 0.0
    exact_match_rate = (exact_match_sequences / total_sequences * 100) if total_sequences > 0 else 0.0
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float('inf')
    
    # Calculate average TEDS
    avg_teds = (total_teds / teds_count) if teds_count > 0 else 0.0
    
    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "token_accuracy": token_accuracy,
        "structure_accuracy": structure_accuracy,
        "content_accuracy": content_accuracy,
        "exact_match_rate": exact_match_rate,
        "teds": avg_teds,  # Tree-Edit-Distance-based Similarity
    }
    
    return metrics


@torch.no_grad()
def measure_inference_time(model, sample_batch, device, num_runs: int = 10):
    """Measure inference time"""
    model.eval()
    images = sample_batch["image"].to(device)
    input_ids = sample_batch["input_ids"].to(device)
    
    # Warmup
    for _ in range(3):
        _ = model(images, input_ids=input_ids)
    
    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = model(images, input_ids=input_ids)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / num_runs


def get_model_size(model):
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def run_experiment(
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str = "cuda",
    output_dir: str = "./experiment_results",
    vocab: Optional[Dict[str, int]] = None,
    resume_from: Optional[str] = None,
) -> Dict:
    """Run a single experiment and return results"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {config.name}")
    print(f"Phase: {config.phase}")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get vocab size from dataset
    vocab_size = len(train_loader.dataset.vocab)
    
    # Create model
    model = create_model(config, vocab_size)
    model = model.to(device)
    
    # Create loss function
    criterion = create_loss_function(config)
    
    # Create optimizer with higher learning rate for faster convergence
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),  # Standard AdamW betas
        eps=1e-8,
    )
    
    # Learning rate scheduler
    scheduler = None
    if config.use_lr_scheduler:
        if config.lr_scheduler_type == "cosine":
            # Cosine annealing with warm restarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs,
                eta_min=config.learning_rate * 0.01,  # Minimum LR is 1% of initial
            )
            print(f"Using CosineAnnealingLR scheduler (T_max={config.num_epochs})")
        elif config.lr_scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.lr_decay_epochs,
                gamma=config.lr_decay_factor,
            )
            print(f"Using MultiStepLR scheduler (milestones={config.lr_decay_epochs})")
    
    # Learning rate warmup
    warmup_scheduler = None
    if config.lr_warmup_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        def warmup_lambda(step):
            if step < config.lr_warmup_steps:
                return float(step) / float(max(1, config.lr_warmup_steps))
            return 1.0
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        print(f"Using learning rate warmup ({config.lr_warmup_steps} steps)")
    
    # Mixed precision scaler
    scaler = None
    if config.use_mixed_precision and device == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision training (FP16)")
    
    # Create checkpoint directory
    checkpoint_dir = Path(output_dir) / "checkpoints" / config.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training state
    start_epoch = 0
    train_losses = []
    val_losses = []
    val_metrics_history = []  # Store all validation metrics
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Resume from checkpoint if provided
    if resume_from:
        resume_path = Path(resume_from)
        # Handle absolute paths directly
        if resume_path.is_absolute():
            # Use absolute path as-is
            pass
        else:
            # Try relative to checkpoint directory first, then as-is
            resume_path = checkpoint_dir / resume_path
            if not resume_path.exists():
                resume_path = Path(resume_from)
        
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        
        resume_state = load_checkpoint(resume_path, model, optimizer, device, scaler)
        checkpoint_epoch = resume_state["epoch"]
        
        # Check if training was already completed (epoch >= num_epochs)
        # If so, reset to epoch 0 to start fresh training
        if checkpoint_epoch >= config.num_epochs:
            print(f"\n⚠️  Checkpoint shows training completed (epoch {checkpoint_epoch} >= {config.num_epochs})")
            print(f"   Resetting to epoch 0 to start fresh training")
            print(f"   Model weights will be reused, but training history will be reset")
            start_epoch = 0
            train_losses = []
            val_losses = []
            val_metrics_history = []
            best_val_loss = float('inf')
            best_epoch = 0
            # Still use vocab from checkpoint
            if resume_state["vocab"] is not None:
                vocab = resume_state["vocab"]
        else:
            # Normal resume: continue from checkpoint epoch
            start_epoch = checkpoint_epoch
            train_losses = resume_state["train_losses"]
            val_losses = resume_state["val_losses"]
            val_metrics_history = resume_state["val_metrics_history"]
            best_val_loss = resume_state["best_val_loss"]
            best_epoch = resume_state["best_epoch"]
            
            # Use vocab from checkpoint if available
            if resume_state["vocab"] is not None:
                vocab = resume_state["vocab"]
            
            # Update config from checkpoint (in case some settings changed)
            # But keep user-provided config for things like num_epochs
            checkpoint_config = resume_state["config"]
            print(f"\nResuming with config from checkpoint:")
            print(f"  Epochs completed: {start_epoch}")
            print(f"  Remaining epochs: {config.num_epochs - start_epoch}")
    
    start_time = time.time()
    
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size} (batch_size={config.batch_size} × accumulation={config.gradient_accumulation_steps})")
    print(f"Initial learning rate: {config.learning_rate}")
    if scheduler is not None:
        print(f"Using {config.lr_scheduler_type} learning rate scheduler")
    if warmup_scheduler is not None:
        print(f"Learning rate warmup: {config.lr_warmup_steps} steps")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Random inference with TEDS will be displayed during validation")
    
    # Calculate global step for warmup
    global_step = start_epoch * len(train_loader) if start_epoch > 0 else 0
    
    for epoch in range(start_epoch, config.num_epochs):
        # Update warmup scheduler during training
        epoch_global_step = global_step
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, config, scaler,
            warmup_scheduler=warmup_scheduler, global_step=epoch_global_step
        )
        global_step += len(train_loader)
        
        # Update learning rate scheduler (after epoch, skip if still in warmup)
        if scheduler is not None:
            # Only update main scheduler if warmup is complete
            if warmup_scheduler is None or global_step >= config.lr_warmup_steps:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                if epoch % 5 == 0 or epoch == start_epoch:  # Print LR every 5 epochs
                    print(f"  Current learning rate: {current_lr:.6f}")
        train_losses.append(train_loss)
        
        if val_loader:
            val_metrics = validate(model, val_loader, criterion, device, config, scaler, vocab=vocab)
            val_loss = val_metrics["loss"]
            val_losses.append(val_loss)
            val_metrics_history.append(val_metrics)  # Store full metrics
            
            # Print metrics
            print(f"Epoch {epoch+1}/{config.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Perplexity: {val_metrics['perplexity']:.2f}")
            print(f"  Val Token Acc: {val_metrics['token_accuracy']:.2f}% | "
                  f"Structure Acc: {val_metrics['structure_accuracy']:.2f}% | "
                  f"Content Acc: {val_metrics['content_accuracy']:.2f}%")
            print(f"  Exact Match Rate: {val_metrics['exact_match_rate']:.2f}% | "
                  f"TEDS: {val_metrics['teds']:.4f}")
            
            # Perform random inference and print in Markdown/HTML format
            try:
                perform_random_inference(model, val_loader, vocab, device, config, epoch + 1)
            except Exception as e:
                print(f"Warning: Could not perform random inference: {e}")
            
            # Save best model (based on validation loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                save_checkpoint(
                    model, optimizer, config, vocab, epoch + 1, 
                    checkpoint_dir / "best.pth",
                    train_loss, val_loss,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    val_metrics_history=val_metrics_history,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    scaler=scaler,
                    data_loader=val_loader if val_loader else train_loader,
                    device=device,
                )
                print(f"  → Saved best model (val_loss: {val_loss:.4f}, "
                      f"token_acc: {val_metrics['token_accuracy']:.2f}%, "
                      f"TEDS: {val_metrics['teds']:.4f})")
        else:
            print(f"Epoch {epoch+1}/{config.num_epochs}: Train Loss: {train_loss:.4f}")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, config, vocab, epoch + 1,
            checkpoint_dir / "latest.pth",
            train_loss, val_losses[-1] if val_losses else None,
            train_losses=train_losses,
            val_losses=val_losses,
            val_metrics_history=val_metrics_history,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            scaler=scaler,
            data_loader=val_loader if val_loader else train_loader,
            device=device,
        )
        
        # Save epoch checkpoint (every 5 epochs or last epoch)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.num_epochs:
            save_checkpoint(
                model, optimizer, config, vocab, epoch + 1,
                checkpoint_dir / f"epoch_{epoch+1}.pth",
                train_loss, val_losses[-1] if val_losses else None,
                train_losses=train_losses,
                val_losses=val_losses,
                val_metrics_history=val_metrics_history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                scaler=scaler,
                data_loader=val_loader if val_loader else train_loader,
                device=device,
            )
    
    training_time = time.time() - start_time
    
    # Measure inference time
    sample_batch = next(iter(train_loader))
    inference_time = measure_inference_time(model, sample_batch, device)
    
    # Get model statistics
    model_size_mb = get_model_size(model)
    num_parameters = sum(p.numel() for p in model.parameters())
    
    # Memory usage (approximate)
    if torch.cuda.is_available():
        memory_usage_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        memory_usage_mb = 0.0
    
    # Get final validation metrics if available
    final_val_metrics = {}
    if val_loader and val_losses:
        final_val_metrics = validate(model, val_loader, criterion, device, config, scaler, vocab=vocab)
    
    results = {
        "config": asdict(config),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_metrics_history": val_metrics_history,  # Store all validation metrics for each epoch
        "final_val_metrics": final_val_metrics,
        "training_time": training_time,
        "inference_time": inference_time,
        "model_size_mb": model_size_mb,
        "num_parameters": num_parameters,
        "memory_usage_mb": memory_usage_mb,
    }
    
    # Save results
    results_path = output_dir / f"{config.name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults:")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Inference Time: {inference_time*1000:.2f}ms")
    print(f"  Model Size: {model_size_mb:.2f} MB")
    print(f"  Parameters: {num_parameters:,}")
    print(f"  Final Train Loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"  Final Val Loss: {val_losses[-1]:.4f}")
        if final_val_metrics:
            print(f"  Final Val Metrics:")
            print(f"    Token Accuracy: {final_val_metrics.get('token_accuracy', 0):.2f}%")
            print(f"    Structure Accuracy: {final_val_metrics.get('structure_accuracy', 0):.2f}%")
            print(f"    Content Accuracy: {final_val_metrics.get('content_accuracy', 0):.2f}%")
            print(f"    Exact Match Rate: {final_val_metrics.get('exact_match_rate', 0):.2f}%")
            print(f"    TEDS: {final_val_metrics.get('teds', 0):.4f}")
            print(f"    Perplexity: {final_val_metrics.get('perplexity', 0):.2f}")
    print(f"\nResults saved to {results_path}")
    print(f"Best model checkpoint: {checkpoint_dir / 'best.pth'}")
    print(f"Latest checkpoint: {checkpoint_dir / 'latest.pth'}")
    
    return results


def ids_to_tokens(token_ids: torch.Tensor, vocab: Dict[str, int]) -> list:
    """Convert token IDs to tokens using vocabulary"""
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = []
    for id in token_ids.cpu().numpy():
        token = id_to_token.get(int(id), "<Pad>")
        if token == EOS_TOKEN:
            break
        tokens.append(token)
    return tokens


def tokens_to_markdown_html(tokens: List[str], vocab: Optional[Dict[str, int]] = None) -> str:
    """Convert tokens to HTML table in markdown format"""
    from tsr.metrics.tsr_metrics import tokens_to_html
    html = tokens_to_html(tokens, vocab)
    return html


def perform_random_inference(
    model: TableRecognitionModel,
    data_loader: DataLoader,
    vocab: Dict[str, int],
    device: str,
    config: ExperimentConfig,
    epoch: int,
):
    """Perform inference on a random sample and print results in Markdown/HTML format with TEDS"""
    model.eval()
    
    # Get a random batch
    dataset = data_loader.dataset
    random_idx = random.randint(0, len(dataset) - 1)
    sample = dataset[random_idx]
    
    # Get image and prepare for generation
    image = sample["image"].unsqueeze(0).to(device)  # (1, C, H, W)
    
    # Get ground truth length to set appropriate max_length
    ground_truth_ids = sample["token_ids"]
    gt_length = (ground_truth_ids != 0).sum().item()  # Count non-padding tokens
    # Use max of ground truth length + buffer, or 1024, whichever is larger
    max_length = max(gt_length + 100, 1024)
    
    # Generate
    with torch.no_grad():
        if config.use_mixed_precision and device == "cuda":
            with torch.cuda.amp.autocast():
                generated_ids = model.generate(
                    image,
                    max_length=max_length,
                    temperature=1.0,
                )
        else:
            generated_ids = model.generate(
                image,
                max_length=max_length,
                temperature=1.0,
            )
    
    # Convert to tokens
    tokens = ids_to_tokens(generated_ids[0], vocab)
    
    # Get ground truth tokens for comparison (already loaded above)
    ground_truth_tokens = ids_to_tokens(ground_truth_ids, vocab)
    
    # Debug: Print token length info
    print(f"\n[Debug] Generation Info:")
    print(f"  Max length used: {max_length}")
    print(f"  Ground truth length: {gt_length}")
    print(f"  Generated length: {len(tokens)}")
    print(f"  Generated IDs shape: {generated_ids.shape}")
    
    # Calculate TEDS
    from tsr.metrics.tsr_metrics import calculate_table_metrics
    table_metrics = calculate_table_metrics(tokens, ground_truth_tokens, vocab)
    teds_score = table_metrics.get('teds', 0.0)
    
    # Convert to HTML
    pred_html = tokens_to_markdown_html(tokens, vocab)
    gt_html = tokens_to_markdown_html(ground_truth_tokens, vocab)
    
    # Print in Markdown format
    print(f"\n{'='*80}")
    print(f"# Random Validation Sample (Epoch {epoch})")
    print(f"{'='*80}\n")
    print(f"**TEDS Score:** {teds_score:.4f}")
    print(f"**Generated Tokens:** {len(tokens)} | **Ground Truth Tokens:** {len(ground_truth_tokens)}\n")
    
    print("## Predicted Table\n")
    print("```html")
    print(pred_html)
    print("```\n")
    
    print("## Ground Truth Table\n")
    print("```html")
    print(gt_html)
    print("```\n")
    
    # Also show as rendered HTML in markdown
    print("## Rendered Tables\n")
    print("### Predicted\n")
    print(pred_html)
    print("\n### Ground Truth\n")
    print(gt_html)
    print(f"\n{'='*80}\n")
    
    model.train()  # Set back to training mode


def load_checkpoint(
    checkpoint_path: Path,
    model: TableRecognitionModel,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict:
    """
    Load checkpoint and restore model, optimizer, and training state
    
    Returns:
        Dictionary with:
            - epoch: Starting epoch (resume from epoch + 1)
            - train_losses: List of training losses
            - val_losses: List of validation losses
            - val_metrics_history: List of validation metrics
            - best_val_loss: Best validation loss so far
            - best_epoch: Best epoch so far
            - config: ExperimentConfig from checkpoint
            - vocab: Vocabulary from checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  ✓ Loaded model state")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  ✓ Loaded optimizer state")
    
    # Load scaler state if available
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"  ✓ Loaded scaler state")
    
    # Extract training state
    start_epoch = checkpoint.get("epoch", 0)
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    val_metrics_history = checkpoint.get("val_metrics_history", [])
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    best_epoch = checkpoint.get("best_epoch", 0)
    
    # Load config (convert dict back to ExperimentConfig)
    config_dict = checkpoint.get("config", {})
    config = ExperimentConfig(**config_dict)
    
    # Load vocab
    vocab = checkpoint.get("vocab", None)
    
    print(f"  ✓ Resuming from epoch {start_epoch + 1}")
    print(f"  ✓ Training losses history: {len(train_losses)} epochs")
    print(f"  ✓ Validation losses history: {len(val_losses)} epochs")
    print(f"  ✓ Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    return {
        "epoch": start_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_metrics_history": val_metrics_history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "config": config,
        "vocab": vocab,
    }


def save_checkpoint(
    model: TableRecognitionModel,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    vocab: Optional[Dict[str, int]],
    epoch: int,
    checkpoint_path: Path,
    train_loss: float,
    val_loss: Optional[float] = None,
    train_losses: Optional[List[float]] = None,
    val_losses: Optional[List[float]] = None,
    val_metrics_history: Optional[List[Dict]] = None,
    best_val_loss: Optional[float] = None,
    best_epoch: Optional[int] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    data_loader: Optional[DataLoader] = None,
    device: str = "cuda",
):
    """Save model checkpoint with all necessary information"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "vocab": vocab,  # Save vocabulary for inference
        "train_losses": train_losses,  # Save training history
        "val_losses": val_losses,  # Save validation history
        "val_metrics_history": val_metrics_history,  # Save validation metrics history
        "best_val_loss": best_val_loss,  # Save best validation loss
        "best_epoch": best_epoch,  # Save best epoch
    }
    
    # Save scaler state if using mixed precision
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    
    # Note: Random inference is now performed during validation, not at checkpoint save

