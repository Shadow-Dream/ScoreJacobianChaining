import torch
import torch.nn as nn
from networks.residual_block import AttentionResidualBlock,ResidualBlock
from networks.embedding import EmbeddingDecreasing
from networks.miscs import Upsample,Downsample,silu

class UNet(nn.Module):
    def __init__(
        self, channels=64
    ):
        super().__init__()
        
        self.time_embed = EmbeddingDecreasing(channels)
        self.time_dense_1 = nn.Linear(channels, channels*4)
        self.time_dense_2 = nn.Linear(channels*4, channels*4)
    
        self.conv_0 = nn.Conv2d(3, channels, 3, 1, 1)
        
        self.down_conv_00 = ResidualBlock(channels, channels,channels*4)
        self.down_down_0 = Downsample(channels)

        self.down_conv_10 = AttentionResidualBlock(channels, channels*2,channels*4)
        self.down_down_1 = Downsample(channels*2)

        self.down_conv_20 = ResidualBlock(channels*2, channels*4,channels*4)
        self.down_down_2 = Downsample(channels*4)

        self.down_conv_30 = ResidualBlock(channels*4, channels*8,channels*4)
        self.down_down_3 = Downsample(channels*8)

        self.mid_conv_0 = ResidualBlock(channels*8, channels*8,channels*4)
        self.mid_conv_1 = AttentionResidualBlock(channels*8, channels*8,channels*4)
        self.mid_conv_2 = ResidualBlock(channels*8, channels*8,channels*4)

        self.up_up_0 = Upsample(channels*8)
        self.up_conv_00 = ResidualBlock(channels*16, channels*4,channels*4)

        self.up_up_1 = Upsample(channels*4)
        self.up_conv_10 = AttentionResidualBlock(channels*8, channels*2,channels*4)

        self.up_up_2 = Upsample(channels*2)
        self.up_conv_20 = ResidualBlock(channels*4, channels,channels*4)

        self.up_up_3 = Upsample(channels)
        self.up_conv_30 = ResidualBlock(channels*2, channels,channels*4)
        
        self.norm_0 = nn.GroupNorm(32, channels)
        self.conv_1 = nn.Conv2d(channels, 3, 3, 1, 1)
    
    def forward(self, x, t):
        t = self.time_embed(t)
        t = self.time_dense_1(t)
        t = silu(t)
        t = self.time_dense_2(t)
        shortcut = []

        down0 = self.conv_0(x)

        down0 = self.down_conv_00(down0, t)
        shortcut.append(down0)
        down0 = self.down_down_0(down0)

        down1 = self.down_conv_10(down0, t)
        shortcut.append(down1)
        down1 = self.down_down_1(down1)

        down2 = self.down_conv_20(down1, t)
        shortcut.append(down2)
        down2 = self.down_down_2(down2)

        down3 = self.down_conv_30(down2, t)
        shortcut.append(down3)
        down3 = self.down_down_3(down3)

        mid0 = self.mid_conv_0(down3,t)
        mid1 = self.mid_conv_1(mid0,t)
        mid2 = self.mid_conv_2(mid1,t)

        up0 = self.up_up_0(mid2)
        up0 = torch.cat([up0, shortcut.pop()], 1)
        up0 = self.up_conv_00(up0,t)

        up1 = self.up_up_1(up0)
        up1 = torch.cat([up1, shortcut.pop()], 1)
        up1 = self.up_conv_10(up1,t)

        up2 = self.up_up_2(up1)
        up2 = torch.cat([up2, shortcut.pop()], 1)
        up2 = self.up_conv_20(up2,t)

        up3 = self.up_up_3(up2)
        up3 = torch.cat([up3, shortcut.pop()], 1)
        up3 = self.up_conv_30(up3,t)

        norm = self.norm_0(up3)
        noise_out = self.conv_1(norm)

        return noise_out