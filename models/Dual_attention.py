import torch
import torch.nn as nn

class DualAttention(nn.Module):
    def __init__(self, in_channels ,reduction_ratio=16, dilation=4):
        super(DualAttention,self).__init__()
        # reduction_ratio and dilation by bottlenect attention paper experiments
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.ca_conv1 = nn.Conv2d(in_channels, in_channels//reduction_ratio, 1)
        self.ca_conv2 = nn.Conv2d(in_channels//reduction_ratio, in_channels, 1)
        # self.ca_conv3 = nn.Conv2d(in_channels, in_channels, 1)

        self.sa_conv1 = nn.Conv2d(in_channels, in_channels//reduction_ratio, 1)
        self.sa_conv2 = nn.Conv2d(in_channels//reduction_ratio, in_channels//reduction_ratio, 3, dilation=dilation, padding=dilation)
        self.sa_conv3 = nn.Conv2d(in_channels//reduction_ratio, in_channels//reduction_ratio, 3, dilation=dilation, padding=dilation)
        self.sa_conv4 = nn.Conv2d(in_channels//reduction_ratio, 1, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self,x):
        b, c, h ,w = x.shape
        # CA
        ca = self.global_avg_pooling(x)
        ca = self.ca_conv1(ca)
        ca = self.ca_conv2(ca)
        ca = ca.expand(b, c, h, w)

        # SA
        sa = self.sa_conv1(x)
        sa = self.sa_conv2(sa)
        sa = self.sa_conv3(sa)
        sa = self.sa_conv4(sa)
        sa = sa.expand(b, c, h, w)

        # fusion
        attn = ca + sa
        attn = nn.functional.sigmoid(attn)

        x = attn * x + x
        return x

if __name__ == '__main__':
    feat = torch.ones([8, 32, 64, 208]).cuda()
    dual_attn = DualAttention(32).cuda()
    out = dual_attn(feat)