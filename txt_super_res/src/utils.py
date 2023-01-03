from re import L
from turtle import forward
import numpy as np
import math
import yaml
import matplotlib.pyplot as plt
import torch
import cv2
from math import log10, sqrt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

with open("args.yml", "r") as f:
    args = yaml.safe_load(f)

def plot_images_couples(x, y_pred, y, epoch):
    if len(x) != len(y_pred):
        raise Exception('Not the same length between images_be and images_af')
    fig, axes = plt.subplots(nrows=len(x), ncols=3)
    fig.tight_layout
    
    axes[0][0].set_title('x')
    axes[0][1].set_title('y_pred')
    axes[0][2].set_title('y')
    
    for row,ax in enumerate(axes):
        ax[0].imshow(x[row], cmap="gray")
        ax[0].axis("off")
        
        ax[1].imshow(y_pred[row], cmap="gray")
        ax[1].axis("off")
        
        ax[2].imshow(y[row], cmap="gray")
        ax[2].axis("off")
    plt.savefig(f"results/sample_epoch_{epoch}.jpg", dpi=1200)
    plt.close()
    
def PSNR(original, compressed):
    original = original.clone().cpu()
    compressed = compressed.clone().cpu()
    
    mse = torch.mean(torch.pow(original - compressed,2))
    
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr    

def train_test_split(dataset, train_size, batch_size, shuffle=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(math.floor(train_size * len(dataset)))

    if shuffle:
        np.random.shuffle(indices)

    train_sampler, test_sampler = (
        SubsetRandomSampler(indices[:split]),
        SubsetRandomSampler(indices[split:]),
    )

    return (
        DataLoader(
            dataset, batch_size=batch_size, num_workers=0, sampler=train_sampler
        ),
        DataLoader(
            dataset, batch_size=batch_size, num_workers=0, sampler=test_sampler
        ),
    )

def train(epoch, model, loss_fn, data_loader, optimizer, device, current_lr):
    model.train()
    train_loss = 0.0
    correct = 0.0

    for batch_idx, (x, y) in enumerate(tqdm(data_loader)):
        x, y = x.to(device), y.to(device)

        out = model(x)
        optimizer.zero_grad()
        loss = loss_fn(out,y).unsqueeze(0)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
            psnr = PSNR(y,out)

    train_loss /= len(data_loader)

    print(
        f"train epoch {epoch}/{args['num_epochs']} ",
        f"loss {train_loss:.5f} PSNR {psnr} lr {current_lr}",
    )

    return train_loss, psnr

def test(model, loss_fn, data_loader, device):
    model.eval()
    test_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(data_loader, 0):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            out = model(x)
                
            loss = loss_fn(out, y).unsqueeze(0)
            test_loss += loss.item()

            psnr = PSNR(y,out)
            
    test_loss /= len(data_loader)

    print(
        f"eval ",
        f"loss {test_loss:.5f} PSNR {psnr}",
    )

    return test_loss, psnr