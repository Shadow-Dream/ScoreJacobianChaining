from models.nerf import NeRF
from utils.dataset import Dataset
from utils.shapes import SphereColor,SphereDensity,BoxColor,BoxDensity
from networks.density import DensityNetwork
from networks.color import ColorNetwork
import cv2 as cv
import numpy as np
import torch

BATCH_SIZE = 4
NEAR = 1
FAR = 8
NUM_SAMPLES = 32
POSITION = torch.tensor([0,0,0],dtype=torch.float32).cuda()
RADIUS = 0.5
CAMERA_POSITION = torch.tensor([[0,0,3]],dtype=torch.float32).cuda()
CAMERA_FORWARD = torch.tensor([[0,0,-1]],dtype=torch.float32).cuda()
CAMERA_UP = torch.tensor([[0,1,0]],dtype=torch.float32).cuda()
IMAGE_SIZE = 64
DATASET_BEGIN = 16385
DATASET_END = 16485
FOCAL = 31.5*(3**0.5)
CHANNELS = 64

density_network = DensityNetwork(CHANNELS).cuda().eval()
color_network = ColorNetwork(CHANNELS).cuda().eval()
dataset = Dataset("datasets",BATCH_SIZE,IMAGE_SIZE,DATASET_BEGIN,DATASET_END)
model = NeRF(density_network,
             color_network,
             None,
             None,
             IMAGE_SIZE,
             FOCAL,
             NEAR,
             FAR,
             NUM_SAMPLES)
model.load()
images,positions,forwards,ups = dataset.batch()
color_map = model.sample(positions,forwards,ups)

images = np.array(images.detach().cpu()*255,dtype=np.uint8)
for i in range(color_map.shape[0]):
    cv.imwrite("results/gen"+str(i)+".jpg",color_map[i])
    cv.imwrite("results/ori"+str(i)+".jpg",images[i])

