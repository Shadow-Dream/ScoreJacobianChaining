import torch
import random
class Chain:
    def __init__(self,diffusion,nerf):
        diffusion.load()
        nerf.load()
        self.diffusion = diffusion
        self.nerf = nerf

    def train(self,dataset):
        _,positions,forwards,ups = dataset.batch()
        batch_size = dataset.batch_size
        
        self.nerf.density_optimizer.zero_grad()
        self.nerf.color_optimizer.zero_grad()
        color_map,_ = self.nerf.render(positions, forwards, ups)
        color_map = color_map.permute(0,3,1,2)

        with torch.no_grad():
            t = torch.randint(0,self.diffusion.schedule.num_timesteps,(batch_size,)).cuda()
            noise_map,noise = self.diffusion.get_images_at_timestamp(color_map,t)
            predict_noise = self.diffusion.view_network(noise_map, t)
        
        color_map.backward(predict_noise)
        self.nerf.density_optimizer.step()
        self.nerf.color_optimizer.step()
        return torch.mean(predict_noise)
    
    def load(self):
        self.diffusion.load()
        self.nerf.load()

    def save(self):
        self.diffusion.save()
        self.nerf.save()
        