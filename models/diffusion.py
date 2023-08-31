import torch
import numpy as np
import tqdm
import os

class Diffusion:
    def __init__(self,
                 schedule,
                 view_network,
                 view_optimizer,
                 image_size
                 ):
        
        self.schedule = schedule
        self.view_network = view_network
        self.view_optimizer = view_optimizer
        self.image_size = image_size

    def get_images_at_timestamp(self,images_0,t):
        sqrt_cumprod = self.schedule.sqrt_cumprod[t]
        sqrt_one_minus_cumprod = self.schedule.sqrt_one_minus_cumprod[t]
        batch_size = images_0.shape[0]
        batch_noise = torch.randn_like(images_0).cuda()
        batch_images_t = torch.mul(sqrt_cumprod.view(batch_size,1,1,1), images_0) + torch.mul(sqrt_one_minus_cumprod.view(batch_size,1,1,1),batch_noise)
        return batch_images_t,batch_noise

    def get_real_t_minus_one_images(self, images_0, images_t, t):
        real_image_0_coef = self.schedule.real_image_0_coef[t]
        real_image_t_coef = self.schedule.real_image_t_coef[t]
        batch_size = images_0.shape[0]
        real_mean = torch.mul(real_image_0_coef.view(batch_size,1,1,1), images_0) + torch.mul(real_image_t_coef.view(batch_size,1,1,1),images_t)
        return real_mean

    def get_predict_t_minus_one_images(self, image_t, t, clip = False):
        batch_size = image_t.shape[0]
        sqrt_recip_cumprod = self.schedule.sqrt_recip_cumprod[t].view(batch_size,1,1,1)
        sqrt_recip_cumprod_minus_one = self.schedule.sqrt_recip_cumprod_minus_one[t].view(batch_size,1,1,1)
        predict_noise = self.view_network(image_t, t)
        predict_images_0 = sqrt_recip_cumprod*image_t - sqrt_recip_cumprod_minus_one*predict_noise
        if clip==True:
            predict_images_0 = torch.clamp(predict_images_0,-1,1)
        predict_images = self.get_real_t_minus_one_images(predict_images_0,image_t,t)
        return predict_images,predict_noise

    def get_images_remove_noise(self, image_t, t, clip = False):
        real_log_var = self.schedule.real_log_var[t]
        predict_images,_ = self.get_predict_t_minus_one_images(image_t,t,clip)
        noise = torch.randn_like(image_t).cuda()
        predict_images = predict_images + torch.exp(0.5 * real_log_var).view(real_log_var.shape[0],1,1,1) * noise
        return predict_images
    
    @torch.no_grad()
    def generate_images(self,batch_size,clip = False):
        images = torch.randn((batch_size,3,self.image_size,self.image_size)).cuda()
        for i in tqdm.tqdm(reversed(range(self.schedule.num_timesteps))):
            t = (i * torch.ones((batch_size,)).cuda()).type(torch.long)
            images = self.get_images_remove_noise(images,t,clip)
        images = images.permute(0,2,3,1)
        images = images.cpu().detach()
        images = np.array(images)
        images = images * 255
        images = np.clip(images,0,255)
        images = images.astype(np.uint8)
        return images
    
    def get_images_remove_noise_ddim(self,images,t_s,t_d,param):
        batch_size = images.shape[0]

        cumprod_s = self.schedule.cumprod[t_s].view(batch_size,1,1,1)
        sqrt_recip_cumprod_s = self.schedule.sqrt_recip_cumprod[t_s].view(batch_size,1,1,1)
        sqrt_recip_cumprod_minus_one_s = self.schedule.sqrt_recip_cumprod_minus_one[t_s].view(batch_size,1,1,1)

        predict_noise = self.view_network(images, t_s)
        cumprod_d = self.schedule.cumprod[t_d].view(batch_size,1,1,1)

        predict_images_0 = sqrt_recip_cumprod_s * images - sqrt_recip_cumprod_minus_one_s * predict_noise

        sigma = param * (((1 - cumprod_d) / (1 - cumprod_s))**0.5) * ((1 - cumprod_s / cumprod_d)**0.5)

        noise = torch.randn_like(images)
        mean_pred = predict_images_0 * ((cumprod_d)**0.5) + ((1 - cumprod_d - sigma ** 2)**0.5) * predict_noise
        sample = mean_pred + sigma * noise
        return sample
    
    @torch.no_grad()
    def generate_images_ddim(self,batch_size,param):
        images = torch.randn((batch_size,3,self.image_size,self.image_size),dtype=torch.float32).cuda()

        for i in reversed(range(self.schedule.num_timesteps//4)):
            if i == 0:
                break
            t_s = (i * 4 * torch.ones((batch_size,)).cuda()).type(torch.long)
            t_d = t_s - 4
            images = self.get_images_remove_noise_ddim(images,t_s,t_d,param)

        images = images.permute(0,2,3,1)
        images = images.cpu().detach()
        images = np.array(images)
        images = (images+1)*127.5
        images = np.clip(images,0,255)
        images = images.astype(np.uint8)
        images_output = []
        for i in range(images.shape[0]):
            images_output.append(images[i])
        return images_output

    def load(self):
        if os.path.exists("weights/view_network.pth"):
            self.view_network.load_state_dict(torch.load("weights/view_network.pth"))

    def save(self):
        torch.save(self.view_network.state_dict(), "weights/view_network.pth")

    def train(self,dataset):
        with torch.no_grad():
            batch_images_0,_,_,_ = dataset.batch()
            batch_images_0 = batch_images_0.permute(0,3,1,2)
            t = torch.randint(0,self.schedule.num_timesteps,(batch_images_0.shape[0],)).cuda()
            batch_images_t,batch_noise = self.get_images_at_timestamp(batch_images_0,t)
        self.view_optimizer.zero_grad()
        predict_noise = self.view_network(batch_images_t,t)
        loss = torch.mean((predict_noise - batch_noise)**2)
        loss.backward()
        self.view_optimizer.step()
        return loss.detach()