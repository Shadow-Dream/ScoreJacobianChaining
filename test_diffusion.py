from networks.unet import UNet
from utils.dataset import Dataset
from models.schedule import LinearSchedule
from models.diffusion import Diffusion
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import torch
import cv2 as cv

IMAGE_SIZE = 64
BATCH_SIZE = 16
ITERS = 1000000
NUM_TIMESTEPS = 1000
LEARNING_RATE = 0.00001
MIN_BETA = 0.001
MAX_BETA = 0.02
CHANNELS = 64
SAVE_DELTA = 10
DATASET_BEGIN = 0
DATASET_END = 1

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

images = model.generate_images(BATCH_SIZE)
for i in range(BATCH_SIZE):
    cv.imwrite("results/" + str(i) + ".jpg",images[i])










