import torch
import torch.nn as nn
import torch.nn.functional as func
from networks.embedding import EmbeddingIncreasing

class DensityNetwork(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.embed = EmbeddingIncreasing(channels=channels)
        self.dense_input = nn.Linear(channels*3,channels*3)
        self.dense_middle = nn.Linear(channels*3,channels*3)
        self.dense_out = nn.Linear(channels*3,1)
    
    def forward(self,position):
        position = self.embed(position)
        feature = func.relu(self.dense_input(position))
        feature = self.dense_middle(feature)
        density = func.relu(self.dense_out(func.relu(feature)))
        return density,feature

