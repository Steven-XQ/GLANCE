# Developed by Junyi Ma
# Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
# https://github.com/IRMVLab/Diff-IP2D
# We thank OCT (Liu et al.), Diffuseq (Gong et al.), and USST (Bao et al.) for providing the codebases.

from transformers import AutoConfig
import torch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

class SideFusionEncoder(nn.Module):
    """
    transform 5 channel to 1 channel
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim = encoder_hidden_dims
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential(
            # linear(input_dims, encoder_hidden_dims),
            # SiLU(),
            # linear(encoder_hidden_dims, output_dims), 
            linear(input_dims, output_dims), 
        )

    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]
        F = x.shape[2]
        x = x.view(B*T, F)
        x = self.feat_embed(x) 
        x = x.view(B, T, x.shape[-1])

        return x

class GazeSideFusionEncoder(nn.Module):
    """SideFusionEncoder with gaze-based gating on object features.

    When gaze_feat is provided, a learned gate modulates the object features
    before fusion, directing the model's attention to gaze-relevant objects.
    When gaze_feat is None, behaves identically to SideFusionEncoder.
    """

    def __init__(self, input_dims, output_dims, encoder_hidden_dims, gaze_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.feat_per_stream = input_dims // 3  # 512

        self.gate_net = nn.Sequential(
            nn.Linear(gaze_dim, self.feat_per_stream),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_per_stream, self.feat_per_stream),
            nn.Sigmoid(),
        )

        self.feat_embed = nn.Sequential(
            linear(input_dims, output_dims),
        )

    def forward(self, x, gaze_feat=None):
        """
        Args:
            x: (B, T, 1536) concatenated [global(512), hand(512), object(512)]
            gaze_feat: (B, T, 512) encoded gaze features, or None
        Returns:
            (B, T, 512)
        """
        B, T, F = x.shape
        if gaze_feat is not None:
            d = self.feat_per_stream
            global_f = x[:, :, :d]
            hand_f = x[:, :, d:2*d]
            obj_f = x[:, :, 2*d:3*d]

            gate = self.gate_net(gaze_feat.reshape(B * T, -1))
            gate = gate.view(B, T, d)
            obj_f = obj_f * gate

            x = th.cat([global_f, hand_f, obj_f], dim=2)

        x = x.view(B * T, self.input_dims)
        x = self.feat_embed(x)
        x = x.view(B, T, self.output_dims)
        return x


class MotionEncoder(nn.Module):
    """
    transform 9 channel to 512 channel
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim = encoder_hidden_dims
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential(
            linear(input_dims, output_dims), 
        )

    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]
        F = x.shape[2]
        x = x.view(B*T, F)
        x = self.feat_embed(x) 
        x = x.view(B, T, x.shape[-1])

        return x