import numpy as np
import torch
import cv2 as cv

class Dataset:
    def __init__(self,path,batch_size,image_size,dataset_begin,dataset_end):
        data_file = open(path+"/data.txt")
        data = data_file.readlines()[dataset_begin:dataset_end]
        images = []
        positions = []
        forwards = []
        ups = []

        def get_vector(vector_string):
            blocks = vector_string.split(", ")
            for i in range(len(blocks)):
                blocks[i] = float(blocks[i])
            return torch.tensor(blocks,dtype=torch.float32)

        for data_line in data:
            blocks = data_line.split(" (")
            index = str(int(blocks[0])+1)
            position = get_vector(blocks[1][:-1])
            forward = get_vector(blocks[2][:-1])
            up = get_vector(blocks[3][:-2])
            image = cv.imread(path+"/images/"+index+".jpg")
            image = cv.resize(image,(image_size,image_size))
            image = torch.tensor(image,dtype=torch.float32)/255
            images.append(image)
            positions.append(position)
            forwards.append(forward)
            ups.append(up)

        dataset_size = len(images)
        self.images = torch.cat(images).reshape(dataset_size,image_size,image_size,3)
        self.ups = torch.cat(ups).reshape(dataset_size,3)
        self.forwards = torch.cat(forwards).reshape(dataset_size,3)
        self.positions = torch.cat(positions).reshape(dataset_size,3)
        self.batch_size = batch_size
        
    def batch(self):
        indices = np.random.choice(self.images.shape[0],self.batch_size,replace=False)
        images = (self.images[indices]).cuda()
        forwards = (self.forwards[indices]).cuda()
        ups = (self.ups[indices]).cuda()
        positions = (self.positions[indices]).cuda()
        return images,positions,forwards,ups
    
    