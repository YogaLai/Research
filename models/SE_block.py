from matplotlib import pyplot as plt
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels ,reduction_ratio=7):
        super(SEBlock,self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        identity = x
        x = self.global_avg_pooling(x).view(b,c)
        x = self.fc(x).view(b, c, 1,1)
        x = identity * x.expand_as(identity) + identity
        return x
