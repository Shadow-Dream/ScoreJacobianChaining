from networks.unet import UNet
from utils.dataset import Dataset
from models.schedule import LinearSchedule
from models.diffusion import Diffusion
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import torch

IMAGE_SIZE = 64
BATCH_SIZE = 8
ITERS = 1000000
NUM_TIMESTEPS = 1000
MIN_BETA = 0.001
MAX_BETA = 0.02
CHANNELS = 64
DATASET_BEGIN = 0
DATASET_END = 16384

LEARNING_RATE = 0.000003

SAVE_DELTA = 10
SAMPLE_DELTA = 1000

schedule = LinearSchedule(NUM_TIMESTEPS,MIN_BETA,MAX_BETA)
view_network = UNet(CHANNELS).cuda().train()
view_optimizer = Adam(view_network.parameters(), lr=LEARNING_RATE)
dataset = Dataset("datasets",BATCH_SIZE,IMAGE_SIZE,DATASET_BEGIN,DATASET_END)
model = Diffusion(
    image_size = IMAGE_SIZE,
    schedule = schedule,
    view_network = view_network,
    view_optimizer = view_optimizer)

model.load()
losses = [1]*50
progress = tqdm(range(ITERS))
for iter in progress:
    loss = model.train(dataset)
    losses.append(float(loss))
    if iter % SAVE_DELTA == 0:
        model.save()
    
    loss_sample = np.mean(np.array(losses[-50:]))
    progress.set_postfix(**{
        "average loss":float(loss_sample),
        "loss": float(loss), 
    })











