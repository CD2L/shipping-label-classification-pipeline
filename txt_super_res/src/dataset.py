import os
import random
import torch
from torch import nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch_enhance
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, imgs_dir, interpolation=cv2.INTER_CUBIC):
        self.imgs_dir = imgs_dir
        self.imgs = os.listdir(imgs_dir)
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((100,250)),
            transforms.ColorJitter(brightness=.4, hue=.2),
            transforms.RandomPosterize(bits=2,p=0.1),
            transforms.RandomRotation(degrees=(-10, 10), interpolation=transforms.InterpolationMode.BILINEAR, fill=255),
            transforms.ToTensor()
        ])
        self.interpolation = interpolation

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_dir,self.imgs[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        img = self.transforms(img).type(torch.float32)  
        
        img_n = img.numpy()
        img_n = np.transpose(img_n, (1,2,0))
             
        rand = random.randint(0,1)
              
        image_blurred = cv2.resize(img_n, (int(img_n.shape[1]/3),int(img_n.shape[0]/3)))
        image_blurred = cv2.resize(image_blurred, (int(img_n.shape[1]),int(img_n.shape[0])), interpolation=cv2.INTER_LINEAR)

        image_blurred = image_blurred.reshape(1, *image_blurred.shape)
        image_blurred = torch.from_numpy(image_blurred).type(torch.float32)  
        
        return image_blurred, img
    
