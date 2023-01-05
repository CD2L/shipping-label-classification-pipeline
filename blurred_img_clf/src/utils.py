import numpy as np
import torch
import math
import os
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

with open('args.yml') as fp:
    args = yaml.safe_load(fp)

def plot_images(x, y_pred, y, epoch, labels):
    fig, axes = plt.subplots(nrows=len(x), ncols=3)
    fig.tight_layout
    
    axes[0][0].set_title('x')
    axes[0][1].set_title('y_pred')
    axes[0][2].set_title('y')
    for row,ax in enumerate(axes):
        ax[0].imshow(x[row], cmap="gray")
        ax[0].axis("off")
        
        ax[1].text(0,0,labels[np.argmax(y_pred[row])]+" ("+str(y_pred[row][np.argmax(y_pred[row])])+")")
        ax[1].axis("off")
        
        ax[2].text(0,0,labels[y[row]])
        ax[2].axis("off")

    if not os.path.isdir('./results'):
        os.mkdir('./results')
    plt.savefig(f"results/sample_epoch_{epoch}.jpg", dpi=1200)
    plt.close()

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

def train(epoch, model, loss_fn, data_loader, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0.0

    for batch_idx, (x, y) in enumerate(tqdm(data_loader)): 
        x, y = x.to(device), y.to(device)
        out = model(x)
        # print(y[0])
        # print(out[0])

        optimizer.zero_grad()
        loss = loss_fn(out,y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()

    train_loss /= len(data_loader)

    print(
        f"train epoch {epoch}/{args['num_epochs']} ",
        f"loss {train_loss:.5f}",
    )

    return train_loss

def test(model, loss_fn, data_loader, device):
    model.eval()
    test_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(data_loader, 0):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            out = model(x)
                
            loss = loss_fn(out, y).unsqueeze(0)
            test_loss += loss.item()
            
    test_loss /= len(data_loader)

    print(
        f"eval ",
        f"loss {test_loss:.5f}",
    )

    return test_loss