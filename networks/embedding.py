import torch
import torch.nn as nn

class EmbeddingIncreasing(nn.Module):
    def __init__(self, channels, scale = 10):
        super().__init__()
        self.channels = channels
        self.scale = scale
        self.embedding_coef = 2**torch.arange(self.channels // 2).type(torch.float32).cuda()

    def forward(self, x):
        embedding_vector = torch.matmul(x.view(*(x.shape),1),self.embedding_coef.view(1,self.channels // 2))
        embedding_vector = torch.cat((embedding_vector.sin(), embedding_vector.cos()), -1)
        embedding_vector = embedding_vector.view(*(embedding_vector.shape[:-2]),-1)
        return embedding_vector
    
class EmbeddingDecreasing(nn.Module):
    def __init__(self, channels, scale = 10):
        super().__init__()
        self.channels = channels
        self.scale = scale

    def forward(self, x):
        x = x.type(torch.float32)
        embedding_vector = torch.exp(-2 * self.scale * torch.arange(self.channels // 2) / self.channels).cuda()
        embedding_vector = torch.matmul(x.view(*(x.shape),1),embedding_vector.view(1,self.channels // 2))
        embedding_vector = torch.cat((embedding_vector.sin(), embedding_vector.cos()), -1)
        embedding_vector = embedding_vector.view(embedding_vector.shape[0],-1)
        return embedding_vector