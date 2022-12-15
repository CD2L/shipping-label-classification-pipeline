from re import T
import torch
from torch import nn
import torch_enhance

class SRResNetModel(nn.Module):
    def __init__(self, scale_factor=2, channels=3):
        super(SRResNetModel, self).__init__()
        super().__setattr__('training', True)
        self.model = torch_enhance.models.SRResNet(scale_factor=scale_factor, channels=channels)
                    
    def forward(self, x: torch.Tensor):
        x = self.model(x)
        maxpool = nn.MaxPool2d(2)
        out = maxpool(x)

        return out
   
class EDSRModel(nn.Module):
    def __init__(self, scale_factor=2, channels=3):
        super(EDSRModel, self).__init__()
        super().__setattr__('training', True)
        self.model = torch_enhance.models.EDSR(scale_factor=scale_factor, channels=channels)
                    
    def forward(self, x: torch.Tensor):
        x = self.model(x)
        maxpool = nn.MaxPool2d(2)
        out = maxpool(x)

        return out
   
