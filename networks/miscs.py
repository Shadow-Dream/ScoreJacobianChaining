import torch
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
    
    def forward(self, x):
        return self.conv_0(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_0 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_0(x)
        return x

def silu(x):
    return torch.sigmoid(x) * x