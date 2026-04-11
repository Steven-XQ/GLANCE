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
    """Encode 32x32 gaze heatmaps to feature vectors via CNN, optionally
    fused with a raw-coordinate MLP stream.

    The heatmap CNN captures spatial context (where the gaze cluster is and
    its shape under blink-zeros), while the coordinate MLP preserves precise
    sub-pixel location that the CNN's striding/pooling smooths away.

    Input: (B*T, 1, 32, 32) heatmap, optional (B*T, 3) coord = (x, y, valid)
    Output: (B*T, output_dim) feature vectors
    """

    def __init__(self, output_dim=512, coord_only=False):
        super().__init__()
        self.coord_only = coord_only
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

        # Coordinate stream: (x, y, valid) -> output_dim
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x, coord=None):
        """
        Args:
            x: (B*T, 1, 32, 32) gaze heatmaps
            coord: (B*T, 3) optional raw (x, y, valid_flag) — if None, only
                CNN features are used (backwards compatible).
        Returns:
            (B*T, output_dim) feature vectors
        """
        if self.coord_only:
            assert coord is not None, "GazeEncoder(coord_only=True) requires coord input"
            return self.coord_mlp(coord)
        h = self.cnn(x)            # (B*T, 256, 1, 1)
        h = h.view(h.size(0), -1)  # (B*T, 256)
        h = self.fc(h)             # (B*T, output_dim)
        if coord is not None:
            h = h + self.coord_mlp(coord)
        return h


class GazeTemporalCrossAttention(nn.Module):
    """Cross-attention from hand features to gaze features with learnable temporal bias.

    The temporal bias is a learnable (1, num_heads, T_max, T_max) parameter added
    to attention scores before softmax. This allows the model to discover the
    optimal temporal offset between gaze and hand (biological prior: gaze leads
    hand by ~200-500ms).
    """

    def __init__(self, dim, num_heads, T_max=20, attn_drop=0., proj_drop=0., fixed_delta=0, bias_init_delta=0, bias_init_amp=2.0, bias_init_sigma=1.0):
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

        # When fixed_delta > 0, force hand[t] to attend only to gaze[max(0, t-delta)]
        # via a hard -inf mask (no learnable bias). This bakes in the biological
        # prior that gaze leads hand by `delta` frames (~167ms per frame at 6fps).
        self.fixed_delta = fixed_delta
        if fixed_delta == 0:
            if bias_init_delta > 0:
                # Soft prior: learnable bias initialized with a Gaussian bump
                # centered at offset (t_h - delta) along the key axis. Lets
                # the model start with the biological prior and adapt.
                t_h = torch.arange(T_max).float().unsqueeze(1)  # (T_max, 1)
                t_g = torch.arange(T_max).float().unsqueeze(0)  # (1, T_max)
                target = (t_h - bias_init_delta).clamp(min=0)
                bump = bias_init_amp * torch.exp(-((t_g - target) ** 2) / (2 * bias_init_sigma ** 2))
                bias_init = bump.unsqueeze(0).unsqueeze(0).expand(1, num_heads, T_max, T_max).contiguous()
                self.temporal_bias = nn.Parameter(bias_init)
            else:
                # Default: zero-init learnable bias
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

        if self.fixed_delta > 0:
            # Hard mask: hand[t] only attends to gaze[clamp(t-delta, 0, T_g-1)].
            # Clamp on both ends because future-frame queries (t >= T_g) would
            # otherwise index past the available gaze (only obs frames).
            t_idx = torch.arange(T_h, device=attn.device)
            target = torch.clamp(t_idx - self.fixed_delta, min=0, max=T_g - 1)  # (T_h,)
            mask = torch.full((T_h, T_g), float('-inf'), device=attn.device)
            mask[t_idx, target] = 0.0
            attn = attn + mask
        else:
            # Add learnable temporal bias (slice to actual sequence lengths)
            attn = attn + self.temporal_bias[:, :, :T_h, :T_g]

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, num_heads, T_h, head_dim)
        out = out.transpose(1, 2).reshape(B, T_h, C)
        out = self.proj_drop(self.proj(out))
        return out
