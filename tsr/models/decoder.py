"""
Transformer Decoder with NoPE (No Positional Encoding)
Implements right-shifted token mechanism and parallel decoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class NoPEDecoderLayer(nn.Module):
    """
    Decoder layer without explicit positional embeddings
    Relies on causal attention mask for relative positioning
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        
        # Cross-attention to encoder
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim) decoder input (right-shifted tokens)
            encoder_output: (B, N, embed_dim) encoder features
            causal_mask: (T, T) causal attention mask
        Returns:
            (B, T, embed_dim)
        """
        # Self-attention with causal mask
        residual = x
        x = self.self_attn_norm(x)
        x_attn, _ = self.self_attn(
            x, x, x,
            attn_mask=causal_mask,
            is_causal=causal_mask is None,  # Use built-in causal if mask is None
        )
        x = residual + self.dropout(x_attn)
        
        # Cross-attention to encoder
        residual = x
        x = self.cross_attn_norm(x)
        x_attn, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = residual + self.dropout(x_attn)
        
        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        
        return x


class HTMLRefiner(nn.Module):
    """
    Non-causal attention module between structure and content decoders
    Allows cells to share dense structural features
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim) structural features
        Returns:
            (B, T, embed_dim) refined features
        """
        residual = x
        x = self.norm(x)
        x_attn, _ = self.attn(x, x, x)  # Non-causal self-attention
        x = residual + self.dropout(x_attn)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder with NoPE and right-shifted token mechanism
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        use_html_refiner: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Token embedding (no positional encoding)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            NoPEDecoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # HTML Refiner (non-causal attention)
        self.use_html_refiner = use_html_refiner
        if use_html_refiner:
            self.html_refiner = HTMLRefiner(embed_dim, num_heads, dropout)
        
        # Output projection
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        # This is a placeholder - actual checkpointing happens in forward pass
        self._gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self._gradient_checkpointing = False
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) right-shifted target tokens
            encoder_output: (B, N, embed_dim) encoder features
            causal_mask: Optional (T, T) causal mask
            return_features: If True, return both logits and features
        Returns:
            (B, T, vocab_size) logits, or (logits, features) if return_features=True
        """
        B, T = input_ids.shape
        
        # Embed tokens (no positional encoding)
        x = self.token_embedding(input_ids)  # (B, T, embed_dim)
        x = self.embed_dropout(x)
        
        # Create causal mask if not provided
        if causal_mask is None:
            causal_mask = self.create_causal_mask(T, x.device)
        
        # Pass through decoder layers (with optional gradient checkpointing)
        use_checkpoint = getattr(self, '_gradient_checkpointing', False)
        for i, layer in enumerate(self.layers):
            if use_checkpoint and self.training:
                # Gradient checkpointing trades compute for memory
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, encoder_output, causal_mask, use_reentrant=False
                )
            else:
                x = layer(x, encoder_output, causal_mask)
        
        # HTML Refiner (non-causal)
        if self.use_html_refiner:
            x = self.html_refiner(x)
        
        # Output projection
        features = self.output_norm(x)  # (B, T, embed_dim)
        logits = self.output_proj(features)  # (B, T, vocab_size)
        
        if return_features:
            return logits, features
        return logits


class ParallelDecoder(nn.Module):
    """
    DREAM-style parallel decoder for multiple elements
    Uses N element queries to generate sequences simultaneously
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_queries: int = 100,  # Number of parallel element queries
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_queries = num_queries
        
        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        # Feature aggregator (cross-attention to encoder)
        self.aggregator_layers = nn.ModuleList([
            NoPEDecoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Multi-token prediction heads
        self.max_tokens_per_element = 5  # Predict up to 5 tokens per element
        self.token_heads = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size)
            for _ in range(self.max_tokens_per_element)
        ])
    
    def forward(
        self,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (B, N, embed_dim) encoder features
        Returns:
            (B, num_queries, max_tokens_per_element, vocab_size) logits
        """
        B = encoder_output.shape[0]
        
        # Expand query embeddings
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, embed_dim)
        
        # Aggregate features through cross-attention
        for layer in self.aggregator_layers:
            # Use queries as both query and key/value for self-attention
            queries = layer(queries, encoder_output, causal_mask=None)
        
        # Predict multiple tokens per element
        logits_list = []
        for head in self.token_heads:
            logits = head(queries)  # (B, num_queries, vocab_size)
            logits_list.append(logits)
        
        logits = torch.stack(logits_list, dim=2)  # (B, num_queries, max_tokens, vocab_size)
        
        return logits

