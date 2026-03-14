"""
Example usage of the Table Recognition System
"""
import torch
from tsr.models.model import TableRecognitionModel
from tsr.data.serialization import SequenceSerializer, TableData, CellData
from tsr.losses.losses import MultiTaskLoss

# Example: Create a simple table
def create_example_table():
    """Create an example table for demonstration"""
    cells = [
        CellData(content="Name", bbox=(10, 10, 100, 30), is_header=True),
        CellData(content="Age", bbox=(110, 10, 180, 30), is_header=True),
        CellData(content="City", bbox=(190, 10, 280, 30), is_header=True),
        CellData(content="John", bbox=(10, 40, 100, 60), is_header=False),
        CellData(content="25", bbox=(110, 40, 180, 60), is_header=False),
        CellData(content="NYC", bbox=(190, 40, 280, 60), is_header=False),
        CellData(content="Jane", bbox=(10, 70, 100, 90), is_header=False),
        CellData(content="30", bbox=(110, 70, 180, 90), is_header=False),
        CellData(content="LA", bbox=(190, 70, 280, 90), is_header=False),
    ]
    
    table = TableData(
        cells=cells,
        image_width=300,
        image_height=100
    )
    
    return table


def example_serialization():
    """Example: Serialize table to sequence"""
    print("=" * 50)
    print("Example: Table Serialization")
    print("=" * 50)
    
    table = create_example_table()
    serializer = SequenceSerializer()
    
    # Serialize table
    sequence = serializer.serialize_table(table)
    print(f"\nSerialized sequence (first 20 tokens):")
    print(sequence[:20])
    
    # Create vocabulary
    vocab = serializer.create_vocabulary([sequence])
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Convert to token IDs
    token_ids = serializer.tokens_to_ids(sequence, vocab)
    print(f"\nToken IDs (first 20): {token_ids[:20]}")


def example_model_creation():
    """Example: Create and use the model"""
    print("\n" + "=" * 50)
    print("Example: Model Creation")
    print("=" * 50)
    
    vocab_size = 10000
    
    # Note: Swin-B requires 224x224 input. For other sizes, use "resnet31" or "convstem"
    # Option 1: Use Swin-B with 224x224 input
    print("\nUsing Swin-B backbone (requires 224x224 input)...")
    model = TableRecognitionModel(
        vocab_size=vocab_size,
        encoder_backbone="swin_b",
        embed_dim=768,
        decoder_layers=6,
        decoder_heads=8,
        use_hybrid_regression=True,
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy input - Swin-B expects 224x224
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, vocab_size, (batch_size, 50))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images, input_ids=input_ids, return_regression=True)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    if 'regression' in outputs:
        print(f"  Regression: {outputs['regression'].shape}")
    if 'column_logits' in outputs:
        print(f"  Column logits: {outputs['column_logits'].shape}")
    
    # Option 2: Use ResNet31 or ConvStem for variable input sizes
    print("\n" + "-" * 50)
    print("Using ResNet31 backbone (supports variable input sizes)...")
    model_resnet = TableRecognitionModel(
        vocab_size=vocab_size,
        encoder_backbone="resnet31",
        embed_dim=768,
        decoder_layers=6,
        decoder_heads=8,
        use_hybrid_regression=True,
    )
    
    # Can use any input size with ResNet31
    images_var = torch.randn(batch_size, 3, 512, 640)
    input_ids_var = torch.randint(0, vocab_size, (batch_size, 50))
    
    with torch.no_grad():
        outputs_var = model_resnet(images_var, input_ids=input_ids_var, return_regression=True)
    
    print(f"  Logits: {outputs_var['logits'].shape}")


def example_loss_computation():
    """Example: Compute multi-task loss"""
    print("\n" + "=" * 50)
    print("Example: Loss Computation")
    print("=" * 50)
    
    vocab_size = 10000
    batch_size = 2
    seq_len = 50
    
    # Create model - use resnet31 for variable input sizes
    model = TableRecognitionModel(
        vocab_size=vocab_size,
        encoder_backbone="resnet31",  # Use resnet31 instead of swin_b for variable sizes
        use_hybrid_regression=True,
    )
    
    # Create dummy data - can use any size with resnet31
    images = torch.randn(batch_size, 3, 512, 640)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create masks
    struct_mask = torch.rand(batch_size, seq_len) > 0.5
    cont_mask = ~struct_mask
    
    # Forward pass
    outputs = model(images, input_ids=input_ids, return_regression=True)
    
    # Add dummy regression targets
    outputs["regression"] = torch.rand(batch_size, seq_len, 4)
    outputs["column_logits"] = torch.rand(batch_size, seq_len, 20)
    
    # Prepare targets
    targets = {
        "token_ids": target_ids,
        "structure_mask": struct_mask,
        "content_mask": cont_mask,
        "bboxes": torch.rand(batch_size, seq_len, 4),
        "bbox_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "column_assignments": torch.randint(0, 10, (batch_size, seq_len)),
        "column_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }
    
    # Compute loss
    criterion = MultiTaskLoss(
        lambda_struc=1.0,
        lambda_cont=1.0,
        lambda_l1=1.0,
        lambda_iou=1.0,
        lambda_consistency=0.1,
    )
    
    losses = criterion(outputs, targets)
    
    print(f"\nLoss components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    print("Table Recognition System - Example Usage\n")
    
    # Run examples
    example_serialization()
    example_model_creation()
    example_loss_computation()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)

