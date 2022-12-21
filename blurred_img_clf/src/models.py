import torch
from torch import nn

class ClfModel(nn.Module):
    def __init__(self) -> None:
        super(ClfModel, self).__init__()
        super(ClfModel, self).requires_grad_(True)
        self.features = nn.Sequential(
            # (?,90,32,1)

            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            # (?,45,16,32)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            # (?,23,8,64)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            # (?,12,4,128)

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            # (?,5,2,256)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5*2*256,3),
            nn.Softmax(dim=1)        
        )

    def forward(self, x):
        out = self.features(x)
        out = self.fc(out)

        return out