import torch
import torch.nn as nn
import torch.nn.functional as func
from networks.embedding import EmbeddingIncreasing

class ColorNetwork(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.embed = EmbeddingIncreasing(channels=channels)
        self.color_input = nn.Linear(channels*3,channels*3)
        self.color_out = nn.Linear(channels*3,3)
    
    def forward(self,direction,feature):
        direction = self.embed(direction)
        feature = feature + direction
        feature = func.relu(self.color_input(feature))
        color = func.sigmoid(self.color_out(feature))
        return color