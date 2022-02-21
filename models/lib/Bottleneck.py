import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, c):
        super(Bottleneck, self).__init__()
        self.conv0 = nn.Conv2d(c, c, kernel_size=1, stride=1)
        self.bn0 = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c,c,  kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(c)
        self.leakyRELU = nn.LeakyReLU()
    
    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.leakyRELU(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyRELU(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.leakyRELU(x)

        return x