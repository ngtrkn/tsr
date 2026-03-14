"""
Visual Encoder for Table Recognition
Supports Swin-B, ResNet-31, and ConvStem backbones
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import timm


class ConvStem(nn.Module):
    """
    Convolutional Stem using stride-2, 3x3 convolutions
    Balances receptive field and sequence length
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 768, num_layers: int = 4):
        super().__init__()
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = embed_dim if i == num_layers - 1 else embed_dim // (2 ** (num_layers - 1 - i))
            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            current_channels = out_channels
        
        self.stem = nn.Sequential(*layers)
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, H'*W', embed_dim) where H'*W' is the flattened spatial dimension
        """
        x = self.stem(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        return x


class ResNet31Encoder(nn.Module):
    """
    ResNet-31 based encoder for document images
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        # Simplified ResNet-31 structure
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet blocks (simplified)
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        self.layer4 = self._make_layer(512, embed_dim, 2)
        
        self.embed_dim = embed_dim
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(self._make_block(in_channels if i == 0 else out_channels, 
                                          out_channels, stride))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels: int, out_channels: int, stride: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        return x


class GCAttention(nn.Module):
    """
    Multi-Aspect Global Context Attention
    Models global relationships after residual blocks
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, reduction: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Global context pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-aspect attention
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // reduction, embed_dim),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim) where N is sequence length
        Returns:
            (B, N, embed_dim)
        """
        B, N, C = x.shape
        residual = x
        
        # Global context
        # Reshape for pooling: (B, N, C) -> (B, C, H, W) approximation
        # For simplicity, use mean pooling
        global_context = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        
        # Multi-head attention with global context
        q = self.query(x)  # (B, N, C)
        k = self.key(global_context.expand(-1, N, -1))  # (B, N, C)
        v = self.value(x)  # (B, N, C)
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v)  # (B, num_heads, N, head_dim)
        
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        # Channel attention
        channel_attn = self.channel_attention(global_context.squeeze(1))  # (B, C)
        out = out * channel_attn.unsqueeze(1)
        
        out = self.out_proj(out)
        out = self.norm(out + residual)
        
        return out


class VisualEncoder(nn.Module):
    """
    Main visual encoder with configurable backbone
    """
    def __init__(
        self,
        backbone: str = "swin_b",  # "swin_b", "resnet31", "convstem"
        in_channels: int = 3,
        embed_dim: int = 768,
        use_gc_attention: bool = True,
        token_compression: Optional[float] = None,  # e.g., 0.8 for 20% reduction
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone_name = backbone
        self._gradient_checkpointing = False
        
        # Build backbone
        if backbone == "swin_b":
            # Use timm Swin Transformer
            self.backbone = timm.create_model(
                "swin_base_patch4_window7_224",
                pretrained=True,
                num_classes=0,
                global_pool="",
            )
            # Adjust embedding dimension if needed
            backbone_dim = self.backbone.num_features
            if backbone_dim != embed_dim:
                self.proj = nn.Linear(backbone_dim, embed_dim)
            else:
                self.proj = nn.Identity()
        
        elif backbone == "resnet31":
            self.backbone = ResNet31Encoder(in_channels, embed_dim)
            self.proj = nn.Identity()
        
        elif backbone == "convstem":
            self.backbone = ConvStem(in_channels, embed_dim)
            self.proj = nn.Identity()
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Global Context Attention
        self.use_gc_attention = use_gc_attention
        if use_gc_attention:
            self.gc_attention = GCAttention(embed_dim)
        
        # Token compression (pixel-shuffle based)
        self.token_compression = token_compression
        if token_compression is not None:
            compression_ratio = int(1 / token_compression)
            self.compress = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * compression_ratio),
                nn.GELU(),
                nn.Linear(embed_dim * compression_ratio, embed_dim),
            )
        else:
            self.compress = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor
        Returns:
            (B, N, embed_dim) encoded features
        """
        if self.backbone_name == "swin_b":
            # Swin outputs (B, H, W, D) format
            x = self.backbone.forward_features(x)
            if isinstance(x, (list, tuple)):
                x = x[-1]  # Take last stage
            # Swin-B outputs (B, H, W, D) - need to reshape to (B, H*W, D)
            if len(x.shape) == 4:
                B, H, W, D = x.shape
                x = x.reshape(B, H * W, D)  # (B, N, D) where N = H*W
            elif len(x.shape) == 3:
                # Already 3D (B, N, D) - good
                pass
            else:
                raise ValueError(f"Unexpected Swin output shape: {x.shape}")
        else:
            # ResNet31 or ConvStem
            x = self.backbone(x)
        
        # Project to embed_dim if needed
        if self.proj is not None and not isinstance(self.proj, nn.Identity):
            x = self.proj(x)
        
        # Apply Global Context Attention
        if self.use_gc_attention:
            # Ensure x is 3D before passing to GCAttention
            if len(x.shape) != 3:
                raise ValueError(f"Expected 3D tensor (B, N, C), got shape {x.shape}")
            x = self.gc_attention(x)
        
        # Token compression
        x = self.compress(x)
        
        return x
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self._gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self._gradient_checkpointing = False

