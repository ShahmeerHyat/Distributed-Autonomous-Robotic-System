"""
Vision Transformer with Head-Width Splitting
=============================================
Based on SPViT paper — splits attention heads across devices.
Each device handles a subset of heads, runs them in parallel.
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Cut image into patches and embed each one."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)       # (B, embed_dim, H', W')
        x = x.flatten(2)             # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)        # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Full MSA — used on the coordinator device to merge partial results.
    Also used standalone when all heads are on one device.
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class PartialMSA(nn.Module):
    """
    Partial MSA — runs only a SUBSET of heads.
    This is what each worker device runs in SPViT's head-width splitting.

    How it works:
        - Full model has e.g. 8 heads
        - Device 0 gets heads [0,1,2,3] — runs PartialMSA with num_heads=4
        - Device 1 gets heads [4,5,6,7] — runs PartialMSA with num_heads=4
        - Both run simultaneously (parallel)
        - Results are summed and projected on coordinator
    """
    def __init__(self, embed_dim=256, total_heads=8, head_indices=None):
        super().__init__()
        self.head_indices = head_indices  # which heads this device handles
        self.num_heads = len(head_indices)
        self.head_dim = embed_dim // total_heads
        self.scale = math.sqrt(self.head_dim)
        self.total_heads = total_heads
        self.embed_dim = embed_dim

        # Only the Q,K,V weights for OUR heads
        partial_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(embed_dim, partial_dim * 3, bias=False)

        # Partial output projection
        self.out_proj = nn.Linear(partial_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        partial_dim = self.num_heads * self.head_dim

        # Compute Q, K, V only for our heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention for our heads
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, partial_dim)

        # Partial output projection
        return self.out_proj(out)


class MLP(nn.Module):
    """Feed-forward network — runs after attention in each block."""
    def __init__(self, embed_dim=256, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One encoder block: LayerNorm → MSA → residual → LayerNorm → MLP → residual"""
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Full ViT for person detection on simulated robot.
    Configured for 32x32 CIFAR-style input (Webots camera output).
    """
    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 num_classes=2, embed_dim=256, depth=6, num_heads=8,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # num_classes=2: person / no-person

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                              # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                      # (B, N+1, D)
        x = self.dropout(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])                            # classify from CLS token

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
