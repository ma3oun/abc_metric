import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, nChannels: int = 3, nClasses: int = 10) -> None:
        super().__init__()

        div = 1
        a1 = 64 // div

        a2 = 64 // div
        a3 = 128 // div
        a4 = 256 // div
        a5 = 512 // div

        self.a5 = a5

        self.conv1 = nn.Sequential(
            nn.Conv2d(nChannels, a1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(a1, a2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a2),
            nn.ReLU(),
            nn.Conv2d(a2, a2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(a2, a2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a2),
            nn.ReLU(),
            nn.Conv2d(a2, a2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(a2, a3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a3),
            nn.ReLU(),
            nn.Conv2d(a3, a3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a3),
        )

        self.m4x = nn.Sequential(
            nn.Conv2d(a2, a3, kernel_size=3, padding=1),
            nn.BatchNorm2d(a3),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(a3, a3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a3),
            nn.ReLU(),
            nn.Conv2d(a3, a3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a3),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Sequential(
            nn.Conv2d(a3, a4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a4),
            nn.ReLU(),
            nn.Conv2d(a4, a4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a4),
        )
        self.m6x = nn.Sequential(
            nn.Conv2d(a3, a4, kernel_size=3, padding=1),
            nn.BatchNorm2d(a4),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(a4, a4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a4),
            nn.ReLU(),
            nn.Conv2d(a4, a4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a4),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(a4, a5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a5),
            nn.ReLU(),
            nn.Conv2d(a5, a5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a5),
        )
        self.m8x = nn.Sequential(
            nn.Conv2d(a4, a5, kernel_size=3, padding=1),
            nn.BatchNorm2d(a5),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(a5, a5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a5),
            nn.ReLU(),
            nn.Conv2d(a5, a5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(a5),
        )

        self.last_layer = nn.Linear(a5, nClasses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.relu(y)

        z = self.conv2(y)
        z = y + z
        z = F.relu(z)

        a = self.conv3(z)
        a = a + z
        a = F.relu(a)

        b = self.m4x(a)
        b = F.relu(b)

        c = self.conv5(b)
        c = c + b
        c = F.relu(c)
        c = self.pool1(c)

        d = self.m6x(c)
        d = F.relu(d)

        e = self.conv7(d)
        e = F.relu(e)
        e = self.pool2(e)

        f = self.m8x(e)
        f = F.relu(f)

        g = self.conv9(f)
        g = g + f
        g = F.relu(g)

        h = F.avg_pool2d(g, (8, 8), stride=(1, 1))
        i = torch.reshape(h, (-1, self.a5))
        k = self.last_layer(i)

        return k


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize all parameters
        self.mlp1 = nn.Linear(32 * 32, 400)
        self.mlp2 = nn.Linear(400, 128)
        self.last_layer = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp1(x.flatten(start_dim=1))
        x = F.relu(y)

        z = self.mlp2(y)
        z = F.relu(z)

        a = self.last_layer(z)

        return a
