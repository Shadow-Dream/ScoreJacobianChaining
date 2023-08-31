from networks.density import DensityNetwork
from networks.color import ColorNetwork
from models.nerf import NeRF
from models.chain import Chain
from networks.unet import UNet
from utils.dataset import Dataset
from models.schedule import LinearSchedule
from models.diffusion import Diffusion
from torch.optim import Adam
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch

IMAGE_SIZE = 64
BATCH_SIZE = 4
ITERS = 100000
NUM_TIMESTEPS = 1000
LEARNING_RATE = 0.0001
MIN_BETA = 0.001
MAX_BETA = 0.02
CHANNELS = 64
DATASET_BEGIN = 0
DATASET_END = 16384
NEAR = 1
FAR = 8
NUM_SAMPLES = 32
FOCAL = 31.5*(3**0.5)
SAVE_DELTA = 10
SAMPLE_DELTA = 1000
CAMERA_POSITION = torch.tensor([[0,0,3]],dtype=torch.float32).cuda()
CAMERA_FORWARD = torch.tensor([[0,0,-1]],dtype=torch.float32).cuda()
CAMERA_UP = torch.tensor([[0,1,0]],dtype=torch.float32).cuda()

schedule = LinearSchedule(NUM_TIMESTEPS,MIN_BETA,MAX_BETA)
view_network = UNet(CHANNELS).cuda().eval()
density_network = DensityNetwork(CHANNELS).cuda().train()
color_network = ColorNetwork(CHANNELS).cuda().train()

view_optimizer = Adam(view_network.parameters(), lr=LEARNING_RATE)
density_optimizer = Adam(density_network.parameters(),LEARNING_RATE)
color_optimizer = Adam(color_network.parameters(),LEARNING_RATE)

dataset = Dataset("datasets",BATCH_SIZE,IMAGE_SIZE,DATASET_BEGIN,DATASET_END)

diffusion = Diffusion(
    image_size = IMAGE_SIZE,
    schedule = schedule,
    view_network = view_network,
    view_optimizer = view_optimizer)

nerf = NeRF(density_network,
    color_network,
    density_optimizer,
    color_optimizer,
    IMAGE_SIZE,
    FOCAL,
    NEAR,
    FAR,
    NUM_SAMPLES)

chain = Chain(diffusion,nerf)

chain.load()

losses = [1,1,1,1,1,1,1,1,1,1]
progress = tqdm(range(ITERS))
for iter in progress:
    loss = chain.train(dataset)
    losses.append(float(loss))
    if iter % SAVE_DELTA == 0:
        chain.save()
    if iter % SAMPLE_DELTA == 0:
        color_map = chain.nerf.sample(CAMERA_POSITION,CAMERA_FORWARD,CAMERA_UP)
        cv.imwrite("results/" + str(iter)+".jpg",color_map[0])
    
    loss_sample = np.mean(np.array(losses[-10:]))
    progress.set_postfix(**{
        "average loss":float(loss_sample),
        "loss": float(loss), 
    })











