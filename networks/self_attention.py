import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels * 3, 1)
        self.conv2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        qkv = self.norm(x)
        qkv = self.conv1(qkv)
        q, k, v = torch.split(qkv, self.channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(batch_size, height * width, channels)
        k = k.view(batch_size, channels, height * width)
        v = v.permute(0, 2, 3, 1).view(batch_size, height * width, channels)

        relavant = torch.softmax(torch.bmm(q, k) * (channels ** (-0.5)), -1)
        attention = torch.bmm(relavant, v)
        attention = attention.view(batch_size, height, width, channels)
        attention = attention.permute(0, 3, 1, 2)
        attention = self.conv2(attention) + x
        return attention