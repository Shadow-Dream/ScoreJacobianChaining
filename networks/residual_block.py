import torch
import torch.nn as nn
from networks.self_attention import SelfAttention
from networks.miscs import silu

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.norm_11 = nn.GroupNorm(32, in_channels)
        self.conv_11 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.norm_12 = nn.GroupNorm(32, out_channels)
        self.drop_12 = nn.Dropout(0.1)
        self.conv_12 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.conv_21 = nn.Conv2d(in_channels, out_channels, 1)

        self.time_dense_1  = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t):
        
        way1 = self.norm_11(x)
        way1 = silu(way1)
        way1 = self.conv_11(way1)

        way2 = self.conv_21(x)

        feature_bias = self.time_dense_1(silu(t))[:, :, None, None]
        way1 = way1 + feature_bias

        way1 = self.norm_12(way1)
        way1 = silu(way1)
        way1 = self.drop_12(way1)
        way1 = self.conv_12(way1)

        return way1 + way2

class AttentionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.norm_11 = nn.GroupNorm(32, in_channels)
        self.conv_11 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.norm_12 = nn.GroupNorm(32, out_channels)
        self.drop_12 = nn.Dropout(0.1)
        self.conv_12 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.conv_21 = nn.Conv2d(in_channels, out_channels, 1)

        self.time_dense_1  = nn.Linear(time_emb_dim, out_channels)
        self.atten = SelfAttention(out_channels)

    def forward(self, x, t):
        way1 = self.norm_11(x)
        way1 = silu(way1)
        way1 = self.conv_11(way1)

        way2 = self.conv_21(x)

        feature_bias = self.time_dense_1(silu(t))[:, :, None, None]
        way1 = way1 + feature_bias

        way1 = self.norm_12(way1)
        way1 = silu(way1)
        way1 = self.drop_12(way1)
        way1 = self.conv_12(way1)

        attention = self.atten(way1 + way2)
        return attention
    
class SimpleAttentionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.norm_11 = nn.GroupNorm(32, in_channels)
        self.conv_11 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.norm_12 = nn.GroupNorm(32, out_channels)
        self.drop_12 = nn.Dropout(0.1)
        self.conv_12 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.conv_21 = nn.Conv2d(in_channels, out_channels, 1)

        self.atten = SelfAttention(out_channels)

    def forward(self, x):
        way1 = self.norm_11(x)
        way1 = silu(way1)
        way1 = self.conv_11(way1)

        way2 = self.conv_21(x)

        way1 = self.norm_12(way1)
        way1 = silu(way1)
        way1 = self.drop_12(way1)
        way1 = self.conv_12(way1)

        attention = self.atten(way1 + way2)
        return attention