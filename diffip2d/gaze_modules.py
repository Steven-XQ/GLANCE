"""
Gaze integration modules for Diff-IP2D.

Provides:
- generate_gaze_heatmap: Convert gaze (x,y) to 32x32 Gaussian heatmap
- GazeEncoder: CNN to encode heatmaps to 512-D feature vectors
- GazeTemporalCrossAttention: Cross-attention with learnable temporal bias
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_gaze_heatmap(gaze_xy, heatmap_size=32, sigma=2.0):
    """Convert normalized gaze coordinates to a Gaussian heatmap.

    Args:
        gaze_xy: (2,) array with (x, y) in [0, 1], or None for missing/blink.
        heatmap_size: spatial resolution of heatmap.
        sigma: Gaussian sigma in pixel space of the heatmap.

    Returns:
        (1, H, W) float32 numpy array.
    """
    if gaze_xy is None:
        return np.zeros((1, heatmap_size, heatmap_size), dtype=np.float32)

    gx, gy = float(gaze_xy[0]), float(gaze_xy[1])
    gx = np.clip(gx, 0.0, 1.0)
    gy = np.clip(gy, 0.0, 1.0)

    # Create coordinate grids in [0, 1]
    coords = np.linspace(0, 1, heatmap_size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing='ij')

    # Gaussian in normalized space
    sigma_norm = sigma / heatmap_size
    heatmap = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma_norm ** 2))

    # Normalize peak to 1
    peak = heatmap.max()
    if peak > 0:
        heatmap = heatmap / peak

    return heatmap[np.newaxis].astype(np.float32)  # (1, H, W)


class GazeEncoder(nn.Module):
    """Encode 32x32 gaze heatmaps to feature vectors via CNN.

    Architecture: 4 conv layers (stride 2) + adaptive pool + linear.
    Input: (B*T, 1, 32, 32) -> Output: (B*T, output_dim)
    """

    def __init__(self, output_dim=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),    # 32 -> 16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 16 -> 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 4 -> 2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                      # 2 -> 1
        )
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B*T, 1, 32, 32) gaze heatmaps
        Returns:
            (B*T, output_dim) feature vectors
        """
        x = self.cnn(x)           # (B*T, 256, 1, 1)
        x = x.view(x.size(0), -1) # (B*T, 256)
        x = self.fc(x)            # (B*T, output_dim)
        return x


class GazeTemporalCrossAttention(nn.Module):
    """Cross-attention from hand features to gaze features with learnable temporal bias.

    The temporal bias is a learnable (1, num_heads, T_max, T_max) parameter added
    to attention scores before softmax. This allows the model to discover the
    optimal temporal offset between gaze and hand (biological prior: gaze leads
    hand by ~200-500ms).
    """

    def __init__(self, dim, num_heads, T_max=20, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable temporal bias: (1, num_heads, T_max, T_max)
        # Initialized near zero so attention starts neutral
        self.temporal_bias = nn.Parameter(torch.zeros(1, num_heads, T_max, T_max))
        nn.init.trunc_normal_(self.temporal_bias, std=0.02)

    def forward(self, q_hand, kv_gaze):
        """
        Args:
            q_hand: (B, T_h, dim) hand trajectory features (queries)
            kv_gaze: (B, T_g, dim) gaze features (keys/values)
        Returns:
            (B, T_h, dim)
        """
        B, T_h, C = q_hand.shape
        T_g = kv_gaze.shape[1]

        q = self.proj_q(q_hand).reshape(B, T_h, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(kv_gaze).reshape(B, T_g, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(kv_gaze).reshape(B, T_g, self.num_heads, self.head_dim).transpose(1, 2)

        # (B, num_heads, T_h, T_g)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add learnable temporal bias (slice to actual sequence lengths)
        attn = attn + self.temporal_bias[:, :, :T_h, :T_g]

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, num_heads, T_h, head_dim)
        out = out.transpose(1, 2).reshape(B, T_h, C)
        out = self.proj_drop(self.proj(out))
        return out
