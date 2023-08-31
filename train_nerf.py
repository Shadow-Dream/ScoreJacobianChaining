from networks.density import DensityNetwork
from networks.color import ColorNetwork
from models.nerf import NeRF
from utils.dataset import Dataset
from torch.optim import Adam
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch

CHANNELS = 64
LEARNING_RATE = 0.0003
BATCH_SIZE = 4
NEAR = 1
FAR = 8
NUM_SAMPLES = 32
ITERS = 100000
SAVE_DELTA = 10
SAMPLE_DELTA = 1000
IMAGE_SIZE = 64
DATASET_BEGIN = 0
DATASET_END = 16384
FOCAL = 31.5*(3**0.5)
CAMERA_POSITION = torch.tensor([[0,0,3]],dtype=torch.float32).cuda()
CAMERA_FORWARD = torch.tensor([[0,0,-1]],dtype=torch.float32).cuda()
CAMERA_UP = torch.tensor([[0,1,0]],dtype=torch.float32).cuda()

density_network = DensityNetwork(CHANNELS).cuda()
color_network = ColorNetwork(CHANNELS).cuda()
density_optimizer = Adam(density_network.parameters(),LEARNING_RATE)
color_optimizer = Adam(color_network.parameters(),LEARNING_RATE)
dataset = Dataset("datasets",BATCH_SIZE,IMAGE_SIZE,DATASET_BEGIN,DATASET_END)
model = NeRF(density_network,
             color_network,
             density_optimizer,
             color_optimizer,
             IMAGE_SIZE,
             FOCAL,
             NEAR,
             FAR,
             NUM_SAMPLES)
model.load()
losses = [1,1,1,1,1,1,1,1,1,1]
progress = tqdm(range(ITERS))
for iter in progress:
    loss = model.train(dataset)
    losses.append(float(loss))
    if iter % SAVE_DELTA == 0:
        model.save()
    if iter % SAMPLE_DELTA == 0:
        color_map = model.sample(CAMERA_POSITION,CAMERA_FORWARD,CAMERA_UP)
        cv.imwrite("results/" + str(iter)+".jpg",color_map[0])
    
    loss_sample = np.mean(np.array(losses[-10:]))
    progress.set_postfix(**{
        "average loss":float(loss_sample),
        "loss": float(loss), 
    })

