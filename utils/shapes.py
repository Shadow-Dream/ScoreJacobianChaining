import torch

class SphereDensity:
    def __init__(self,position,radius):
        self.position = position
        self.radius = radius
    
    def __call__(self,positions):
        distances = torch.norm(positions - self.position,dim=-1,keepdim=True)
        one = torch.ones_like(distances,dtype=torch.float32).cuda()
        zero = torch.zeros_like(distances,dtype=torch.float32).cuda()
        return torch.where(distances<self.radius,one*10000,zero),distances
    
class SphereColor:
    def __init__(self,position,radius):
        self.position = position
        self.radius = radius
    
    def __call__(self,direction,distances):
        one = torch.ones_like(distances,dtype=torch.float32).cuda()
        zero = torch.zeros_like(distances,dtype=torch.float32).cuda()
        color = torch.where(distances<self.radius,one,zero)
        color = torch.concat([color,color,color],dim = -1)
        return color
    
class BoxDensity:
    def __init__(self,position,size):
        self.position = position
        self.size = size
    
    def __call__(self,positions):
        distances = torch.max(torch.abs(positions - self.position),dim = -1)[0].unsqueeze(-1)
        one = torch.ones_like(distances,dtype=torch.float32).cuda()
        zero = torch.zeros_like(distances,dtype=torch.float32).cuda()
        return torch.where(distances<self.size,one*10000,zero),distances
    
class BoxColor:
    def __init__(self,position,size):
        self.position = position
        self.size = size
    
    def __call__(self,direction,distances):
        one = torch.ones_like(distances,dtype=torch.float32).cuda()
        zero = torch.zeros_like(distances,dtype=torch.float32).cuda()
        color = torch.where(distances<self.size,one,zero)
        color = torch.concat([color,color,color],dim = -1)
        return color