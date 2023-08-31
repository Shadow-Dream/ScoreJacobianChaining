import torch
import math

class Schedule:
    def __init__(self,num_timesteps,cumprod,cumprod_plus_one,cumprod_minus_one,sqrt_cumprod,sqrt_one_minus_cumprod,
        betas,alphas,sqrt_alphas,real_image_0_coef,real_image_t_coef,real_var,real_log_var,log_betas,
        sqrt_recip_cumprod,sqrt_recip_cumprod_minus_one,coef_0,coef_1):
        self.num_timesteps = num_timesteps
        self.cumprod=cumprod
        self.cumprod_plus_one=cumprod_plus_one
        self.cumprod_minus_one=cumprod_minus_one
        self.sqrt_cumprod=sqrt_cumprod
        self.sqrt_one_minus_cumprod=sqrt_one_minus_cumprod
        self.betas=betas
        self.alphas=alphas
        self.sqrt_alphas=sqrt_alphas
        self.real_image_0_coef=real_image_0_coef
        self.real_image_t_coef=real_image_t_coef
        self.real_var=real_var
        self.real_log_var=real_log_var
        self.log_betas=log_betas
        self.sqrt_recip_cumprod=sqrt_recip_cumprod
        self.sqrt_recip_cumprod_minus_one=sqrt_recip_cumprod_minus_one
        self.coef_0 = coef_0
        self.coef_1 = coef_1

class LinearSchedule(Schedule):
    def __init__(self,num_timesteps,min_beta,max_beta):
        betas = torch.linspace(min_beta,max_beta,num_timesteps,dtype = torch.float32).cuda()
        alphas = 1 - betas
        cumprod = torch.cumprod(alphas,0)
        cumprod_plus_one = torch.concat([cumprod,torch.tensor([0],dtype=torch.float32).cuda()])[1:num_timesteps+1]
        cumprod_minus_one = torch.concat([torch.tensor([1],dtype=torch.float32).cuda(),cumprod])[0:num_timesteps]
        sqrt_cumprod = cumprod ** 0.5
        sqrt_one_minus_cumprod = (1 - cumprod) ** 0.5
    
        sqrt_alphas = alphas**0.5
        real_image_0_coef = betas*((cumprod_minus_one)**0.5)/(1-cumprod)
        real_image_t_coef = (1 - cumprod_minus_one)* sqrt_alphas / (1 - cumprod)

        real_var = betas*(1 - cumprod_minus_one)/(1 - cumprod)
        real_log_var = real_var
        real_log_var[0] = real_var[1]
        real_log_var = torch.log(real_log_var)

        log_betas = torch.log(betas)
        sqrt_recip_cumprod = (1/cumprod)**0.5
        sqrt_recip_cumprod_minus_one = (1 / cumprod - 1)**0.5

        coef_0 = betas/sqrt_one_minus_cumprod
        coef_1 = 1/sqrt_alphas
        super().__init__(
            num_timesteps,
            cumprod.to(torch.float32),
            cumprod_plus_one.to(torch.float32),
            cumprod_minus_one.to(torch.float32),
            sqrt_cumprod.to(torch.float32),
            sqrt_one_minus_cumprod.to(torch.float32),
            betas.to(torch.float32),
            alphas.to(torch.float32),
            sqrt_alphas.to(torch.float32),
            real_image_0_coef.to(torch.float32),
            real_image_t_coef.to(torch.float32),
            real_var.to(torch.float32),
            real_log_var.to(torch.float32),
            log_betas.to(torch.float32),
            sqrt_recip_cumprod.to(torch.float32),
            sqrt_recip_cumprod_minus_one.to(torch.float32),
            coef_0.to(torch.float32),
            coef_1.to(torch.float32))

class CosineSchedule(Schedule):
    def __init__(self,num_timesteps,param = 0.008,max_beta = 0.999):
        betas = self.generate_accurate_betas(num_timesteps,param,max_beta)
        alphas = 1 - betas
        cumprod = torch.cumprod(alphas,0)
        cumprod_plus_one = torch.concat([cumprod,torch.tensor([0],dtype=torch.float32).cuda()])[1:num_timesteps+1]
        cumprod_minus_one = torch.concat([torch.tensor([1],dtype=torch.float32).cuda(),cumprod])[0:num_timesteps]
        sqrt_cumprod = cumprod ** 0.5
        sqrt_one_minus_cumprod = (1 - cumprod) ** 0.5
    
        sqrt_alphas = alphas**0.5
        real_image_0_coef = betas*((cumprod_minus_one)**0.5)/(1-cumprod)
        real_image_t_coef = (1 - cumprod_minus_one)* sqrt_alphas / (1 - cumprod)

        real_var = betas*(1 - cumprod_minus_one)/(1 - cumprod)
        real_log_var = real_var
        real_log_var[0] = real_var[1]
        real_log_var = torch.log(real_log_var)

        log_betas = torch.log(betas)
        sqrt_recip_cumprod = (1/cumprod)**0.5
        sqrt_recip_cumprod_minus_one = (1 / cumprod - 1)**0.5

        coef_0 = betas/sqrt_one_minus_cumprod
        coef_1 = 1/sqrt_alphas
    
        super().__init__(
            num_timesteps,
            cumprod.to(torch.float32),
            cumprod_plus_one.to(torch.float32),
            cumprod_minus_one.to(torch.float32),
            sqrt_cumprod.to(torch.float32),
            sqrt_one_minus_cumprod.to(torch.float32),
            betas.to(torch.float32),
            alphas.to(torch.float32),
            sqrt_alphas.to(torch.float32),
            real_image_0_coef.to(torch.float32),
            real_image_t_coef.to(torch.float32),
            real_var.to(torch.float32),
            real_log_var.to(torch.float32),
            log_betas.to(torch.float32),
            sqrt_recip_cumprod.to(torch.float32),
            sqrt_recip_cumprod_minus_one.to(torch.float32),
            coef_0.to(torch.float32),
            coef_1.to(torch.float32))

    def generate_accurate_betas(self,num,param,max_beta):
        betas = []
        t2 = 0
        alpha_bar2 = math.cos(param / (1+param) * math.pi / 2) ** 2
        for i in range(num):
            t2 = (i + 1) / num
            alpha_bar1 = alpha_bar2
            alpha_bar2 = math.cos((t2 + param) / (1+param) * math.pi / 2) ** 2
            betas.append(min(1 - alpha_bar2 / alpha_bar1, max_beta))
        return torch.tensor(betas,dtype=torch.float64).cuda()