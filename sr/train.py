import math
import os
from turtle import forward
from matplotlib.pyplot import axes
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import ImageDataset
from src.model import SRResNetModel
from src.utils import PSNR, plot_images_couples, train, test, train_test_split
#import tensorflow as tf

class MGE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_p: torch.Tensor, y_t: torch.Tensor, eps:float = 1e-18):
        y_pred_ = y_p.cpu()
        y_true_ = y_t.cpu()

        sobel_h = torch.Tensor([[-1, -2, -1],
                                [0,   0,  0],
                                [1,   2,  1]])
        sobel_v = sobel_h.T
        
        weights_h = sobel_h.view(1,1,3,3)
        weights_v = sobel_v.view(1,1,3,3)
        ## Predicted img
        ypg_1 = torch.conv2d(y_pred_, weights_v, padding=1)  #correct
        ypg_2 = torch.conv2d(y_pred_, weights_h, padding=1)  #correct
    
        ypg = torch.sqrt((torch.pow(ypg_1,2) + torch.pow(ypg_2, 2))+ eps)
        ## Original img
        ytg_1 = torch.conv2d(y_true_, weights_v, padding=1)
        ytg_2 = torch.conv2d(y_true_, weights_h, padding=1)

        ytg = torch.sqrt((torch.pow(ytg_1,2) + torch.pow(ytg_2, 2))+ eps) #correct
        
        out = torch.pow(ytg - ypg, 2)
        mask = out.eq(0.0).float()
        out = out + mask * eps
        out = torch.sqrt(out + eps) * (1.0 - mask) 
        mge = torch.mean(out)

        return mge

class MSExMGE(nn.Module):
    def __init__(self, mse_weight=1, mge_weight=0.1) -> None:
        super().__init__()
        self.mge = MGE()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.mge_weight = mge_weight
        
    def forward(self, y_p, y_t):
        return self.mse(y_p, y_t)*self.mse_weight + self.mge(y_p,y_t)*self.mge_weight

def MeanGradientError(outputs, targets, weight=0.1):
    outputs = outputs.cpu().detach()
    targets = targets.cpu().detach()
    
    
    filter_x = tf.tile(tf.expand_dims(tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = float), axis = -1), [1, 1, outputs.shape[-1]])
    filter_x = tf.tile(tf.expand_dims(filter_x, axis = -1), [1, 1, 1, outputs.shape[-1]])
    filter_y = tf.tile(tf.expand_dims(tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = float), axis = -1), [1, 1, targets.shape[-1]])
    filter_y = tf.tile(tf.expand_dims(filter_y, axis = -1), [1, 1, 1, targets.shape[-1]])

    # output gradient
    output_gradient_x = tf.math.square(tf.nn.conv2d(outputs, filter_x, strides = 1, padding = 'SAME'))
    output_gradient_y = tf.math.square(tf.nn.conv2d(outputs, filter_y, strides = 1, padding = 'SAME'))

    #target gradient
    target_gradient_x = tf.math.square(tf.nn.conv2d(targets, filter_x, strides = 1, padding = 'SAME'))
    target_gradient_y = tf.math.square(tf.nn.conv2d(targets, filter_y, strides = 1, padding = 'SAME'))

    # square
    output_gradients = tf.math.sqrt(tf.math.add(output_gradient_x, output_gradient_y))
    target_gradients = tf.math.sqrt(tf.math.add(target_gradient_x, target_gradient_y))

    # compute mean gradient error
    shape = output_gradients.shape[1:3]
    mge = tf.math.reduce_sum(tf.math.squared_difference(output_gradients, target_gradients) / (shape[0] * shape[1]))

    return torch.tensor(mge * weight, requires_grad=True)
    
def main(): 
    torch.cuda.empty_cache()

    with open("args.yml", "r") as f:
        args = yaml.safe_load(f)

    show_sample = True
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoints = './checkpoints'
    if not os.path.exists(checkpoints):
        os.mkdir(checkpoints)
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
        
    dataset = ImageDataset(args['dataset_dirname'])
    fit, val = train_test_split(dataset, train_size=args["split_sizes"], batch_size=args['batch_size'], shuffle=True)
        
    model = SRResNetModel(2,1)
    model = nn.DataParallel(model)
    model = model.to(device)
    
    loss_fn = MSExMGE(mge_weight=0.25)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=0.1)
    
    loss_history = {"fit": [], "val": []}
    acc_history = {"fit": [], "val": []}
    psnr_history = {"fit": [], "val": []}
    
    for epoch in range(1, args['num_epochs'] + 1):
        train_loss, train_psnr = train(epoch, model, loss_fn, fit, optimizer, device)
        test_loss, test_psnr = test(model, loss_fn, val, device)

        loss_history["fit"].append(train_loss)
        loss_history["val"].append(test_loss)
        
        psnr_history["fit"].append(train_psnr)
        psnr_history["val"].append(test_psnr)

            
        if show_sample and not epoch % 5:
            sample_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0
            )

            data_iter = iter(sample_loader)
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            out = model(x)

            x = x.cpu().numpy().transpose([0, 2, 3, 1])
            y = y.cpu().numpy().transpose([0, 2, 3, 1])
            out = out.cpu().detach().numpy().transpose([0, 2, 3, 1])
            
            plot_images_couples(x, out, y, epoch)

            torch.save(
                {
                    "psnr": psnr_history,
                    "loss_history": loss_history,
                    "acc_history": acc_history,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                f"checkpoints/checkpoint_{epoch}.pkl",
            )
        
if __name__ == '__main__':
    main()
