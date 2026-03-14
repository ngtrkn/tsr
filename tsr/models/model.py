"""
Main End-to-End Multi-Task Learning Model
Implements unified sequence generation with hybrid regression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .encoder import VisualEncoder
from .decoder import TransformerDecoder, ParallelDecoder


class HybridRegressionHead(nn.Module):
    """
    Hybrid regression head for continuous coordinate prediction
    Outputs normalized (x, y, w, h) coordinates
    """
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4),  # (x, y, w, h)
            nn.Sigmoid()  # Normalize to [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim) decoder features
        Returns:
            (B, T, 4) normalized coordinates (x, y, w, h)
        """
        return self.head(x)


class ColumnConsistencyHead(nn.Module):
    """
    Predicts column assignments for consistency loss
    """
    def __init__(self, embed_dim: int = 768, max_columns: int = 20):
        super().__init__()
        self.max_columns = max_columns
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, max_columns),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim) decoder features
        Returns:
            (B, T, max_columns) column assignment logits
        """
        return self.head(x)


class TableRecognitionModel(nn.Module):
    """
    End-to-End Multi-Task Learning Model for Table Recognition
    """
    def __init__(
        self,
        vocab_size: int,
        encoder_backbone: str = "swin_b",
        embed_dim: int = 768,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        use_html_refiner: bool = True,
        use_gc_attention: bool = True,
        token_compression: Optional[float] = None,
        use_hybrid_regression: bool = True,
        use_parallel_decoder: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.use_hybrid_regression = use_hybrid_regression
        self.use_parallel_decoder = use_parallel_decoder
        
        # Visual Encoder
        self.encoder = VisualEncoder(
            backbone=encoder_backbone,
            embed_dim=embed_dim,
            use_gc_attention=use_gc_attention,
            token_compression=token_compression,
        )
        
        # Decoder
        if use_parallel_decoder:
            self.decoder = ParallelDecoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_layers=decoder_layers,
                num_heads=decoder_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
        else:
            self.decoder = TransformerDecoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_layers=decoder_layers,
                num_heads=decoder_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_html_refiner=use_html_refiner,
            )
        
        # Hybrid Regression Head (for continuous coordinates)
        if use_hybrid_regression:
            self.regression_head = HybridRegressionHead(embed_dim)
            self.column_consistency_head = ColumnConsistencyHead(embed_dim)
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        return_regression: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, C, H, W) input images
            input_ids: (B, T) right-shifted target tokens (for training)
            return_regression: Whether to return regression predictions
        Returns:
            Dictionary with:
                - logits: (B, T, vocab_size) or (B, num_queries, max_tokens, vocab_size)
                - regression: (B, T, 4) if return_regression=True
                - column_logits: (B, T, max_columns) if return_regression=True
        """
        # Encode images
        encoder_output = self.encoder(images)  # (B, N, embed_dim)
        
        if self.use_parallel_decoder:
            # Parallel decoding
            logits = self.decoder(encoder_output)
            outputs = {"logits": logits}
        else:
            # Sequential decoding
            if input_ids is None:
                raise ValueError("input_ids required for sequential decoder")
            
            # Get logits and features
            if return_regression and self.use_hybrid_regression:
                logits, decoder_features = self.decoder(
                    input_ids, encoder_output, return_features=True
                )
            else:
                logits = self.decoder(input_ids, encoder_output)
                decoder_features = None
            
            outputs = {"logits": logits}
            
            # Hybrid regression
            if return_regression and self.use_hybrid_regression and decoder_features is not None:
                regression = self.regression_head(decoder_features)
                column_logits = self.column_consistency_head(decoder_features)
                
                outputs["regression"] = regression
                outputs["column_logits"] = column_logits
        
        return outputs
    
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate sequence autoregressively
        
        Args:
            images: (B, C, H, W) input images
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        Returns:
            (B, T) generated token IDs
        """
        self.eval()
        device = images.device
        B = images.shape[0]
        
        # Encode images
        encoder_output = self.encoder(images)
        
        # Initialize with BOS token
        generated = torch.full((B, 1), 1, device=device, dtype=torch.long)  # BOS = 1
        
        for _ in range(max_length - 1):
            # Get logits
            logits = self.decoder(generated, encoder_output)  # (B, T, vocab_size)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS - but be more conservative about stopping
            # Only stop if:
            # 1. We've generated a reasonable minimum length (at least 50 tokens)
            # 2. We've seen EOS token
            # This prevents premature stopping during early training when model might predict EOS incorrectly
            # During early training, models often predict EOS too early, so we require a minimum length
            if (next_token == 2).all():  # EOS = 2
                # Only stop on EOS if we've generated enough tokens
                # This prevents premature stopping during early training
                if generated.shape[1] >= 50:
                    break
                # If we get EOS too early (< 50 tokens), continue generating
                # This helps during early training when EOS might be predicted incorrectly
                # We'll stop at max_length anyway
        
        return generated

