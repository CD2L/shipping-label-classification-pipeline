from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

class ImageFolderDataset(Dataset):
    def __init__(self, imgs_dir, device = 'cuda') -> None:
        self.imgs_dir = imgs_dir
        self.imgs = []
        self.encoder_dict = {}
        self.decoder_dict = {}
        self.device = device
        
        img_scandir = os.scandir(imgs_dir)
        self.one_hot = torch.eye(len(list(img_scandir)))

        for idx, cls in enumerate(os.scandir(imgs_dir)):
            self.encoder_dict[cls.name] = idx
            self.decoder_dict[idx] = cls.name
            cls_dir = os.scandir(cls.path)
            for img in cls_dir:
                self.imgs.append((img.path, int(self.encoder_dict[cls.name])))
        
        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32,90)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, lbl = self.imgs[index]

        oh_lbl = self.one_hot[lbl]

        img = Image.open(img_path)
        img = self.transforms(img)

        return img, oh_lbl
