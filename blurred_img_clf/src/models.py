import torch
from torch import nn

class BICModel(nn.Module):
    def __init__(self) -> None:
        super(BICModel, self).__init__()
        super(BICModel, self).requires_grad_(True)

        self.layer0_k3 = nn.Sequential(
            # (?,90,32,1)

            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            # (?,45,16,32)
        )

        self.layer0_k5 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            # (?,45,16,32)
        )

        self.layer0_k7 = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            # (?,45,16,32)
        )

        self.layer0_k10 = nn.Sequential(
            nn.Conv2d(1, 32, 10, padding=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            # (?,45,16,32)
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            # (?,22,8,256)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            # (?,12,4,512)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            # (?,5,2,1024)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5*2*1024+22*8*256,3)
        )

    def forward(self, x):
        y_0 = self.layer0_k3(x)
        y_1 = self.layer0_k5(x)
        y_2 = self.layer0_k7(x)
        y_3 = self.layer0_k10(x)

        y = torch.cat((y_0,y_1,y_2,y_3),1)

        y = self.layer1(y)
        z = self.layer2(y)
        z = self.layer3(z)

        y = y.view(y.size(0), -1)
        z = z.view(z.size(0), -1)

        out = torch.cat((y,z),1)

        out = self.fc(out)

        return out