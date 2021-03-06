import torch
import torch.nn as nn
from models.Dual_attention import DualAttention
from models.lib.dcn import DeformableConv2d
import torch.nn.functional as F
import numpy as np
# from networks.correlation_package.correlation import Correlation

class MyCorrelation(nn.Module):
    def __init__(self, d=4):
        super(MyCorrelation,self).__init__()
        self.d = d
    
    def forward(self, feat1, feat2):
        h, w = feat1.size(2), feat1.size(3)
        cv = []
        feat2 = torch.nn.functional.pad(feat2, [self.d,self.d,self.d,self.d], "constant", 0)
        for i in range(2*self.d+1):
            for j in range(2*self.d+1):
                cv.append(torch.mean(feat1*feat2[:,:,i:(i+h),j:(j+w)], dim=1, keepdim=True))
        
        return torch.cat(cv, axis=1)

def torch_pwc_corr(refimg_fea, targetimg_fea):
    maxdisp=4
    b,c,h,w = refimg_fea.shape
    targetimg_fea = F.unfold(targetimg_fea, (2*maxdisp+1,2*maxdisp+1), padding=maxdisp).view(b,c,2*maxdisp+1, 2*maxdisp+1**2,h,w)
    cost = refimg_fea.view(b,c,h,w)[:,:,np.newaxis, np.newaxis] * targetimg_fea.view(b,c,2*maxdisp+1, 2*maxdisp+1**2,h,w)
    # # reweight
    # cost = cost.contiguous() 
    # b, c, ph, pw, h, w = cost.size()
    # cost = cost.view(b*ph*pw, c, h, w)
    # cost = reweight_net(cost)
    # cost = cost.view(b, ph, pw, c, h, w)
    # cost = cost.permute([0,3,1,2,4,5]).contiguous() 
    # # (B, 2C, U, V, H, W) -> (B, U, V, 2C, H, W)
    #     cost = cost.permute([0,2,3,1,4,5]).contiguous() 
    #     # (B, U, V, 2C, H, W) -> (BxUxV, 2C, H, W)
    #     cost = cost.view(x.size()[0]*sizeU*sizeV,c*2, x.size()[2], x.size()[3])
    #     cost = matchnet(cost)
    #     # (BxUxV, 2C, H, W) -> (BxUxV, 1, H, W)
    #     cost = cost.view(x.size()[0],sizeU,sizeV,1, x.size()[2],x.size()[3])
    #     cost = cost.permute([0,3,1,2,4,5]).contiguous() 

    cost = cost.sum(1)
    b, ph, pw, h, w = cost.size()
    cost = cost.view(b, ph * pw, h, w)/refimg_fea.size(1)
    return cost

class GwcCorrelation(nn.Module):
    def __init__(self, d=4, n_groups=16):
        super(GwcCorrelation,self).__init__()
        self.n_groups = n_groups
        self.corr = Correlation(pad_size=d, kernel_size=1, max_displacement=d, stride1=1, stride2=1, corr_multiply=1)
        self.cost_net = nn.Sequential(
            BasicConv(16,32, kernel_size=3, padding=1),
            BasicConv(32,64, kernel_size=3, padding=1, stride=2),
            BasicConv(64,64, kernel_size=3, padding=1),
            BasicConv(64,32, kernel_size=3, padding=1),
            BasicConv(32,16, kernel_size=4, padding=1, stride=2, deconv=True),
            nn.Conv2d(16,1, kernel_size=3, padding=1)
        )
            # BasicConv(64, 96, kernel_size=3, padding=1,   dilation=1),
            # BasicConv(96, 128, kernel_size=3, stride=2,    padding=1),   # down by 1/2
            # BasicConv(128, 128, kernel_size=3, padding=1,   dilation=1),
            # BasicConv(128, 64, kernel_size=3, padding=1,   dilation=1),
            # BasicConv(64, 32, kernel_size=4, padding=1, stride=2, deconv=True), # up by 1/2 
            # nn.Conv2d(32, 1  , kernel_size=3, stride=1, padding=1, bias=True),)

    def forward(self, feat1, feat2):
        h, w = feat1.size(2), feat1.size(3)
        feat1 = feat1.chunk(self.n_groups, dim=1)
        feat2 = feat2.chunk(self.n_groups, dim=1)
        cv = []
        for i in range(len(feat1)):
            cv.append(self.corr(feat1[i], feat2[i]).unsqueeze(1))
        cv = torch.cat(cv, axis=1).contiguous()
        b,c,d,h,w = cv.shape
        cv = cv.view(b*d, c, h, w)
        cv = self.cost_net(cv)
        cv = cv.view(b, d, h, w)
        return cv

class AttnCorrelation(nn.Module):
    def __init__(self, in_channel, d=4, reduction_ratio=16):
        super(AttnCorrelation,self).__init__()
        self.d = d
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.ca_conv1 = nn.Conv2d(in_channel, in_channel//reduction_ratio, 1)
        self.ca_conv2 = nn.Conv2d(in_channel//reduction_ratio, in_channel, 1)
    
    def forward(self, feat1, feat2):
        _, c, h, w = feat1.shape
        cv = []
        feat2 = torch.nn.functional.pad(feat2, [self.d,self.d,self.d,self.d], "constant", 0)
        for i in range(2*self.d+1):
            for j in range(2*self.d+1):
                similarity = feat1*feat2[:,:,i:(i+h),j:(j+w)]
                attn = self.global_avg_pooling(similarity)
                attn = self.ca_conv2(self.ca_conv1(attn))
                attn = torch.sigmoid(attn)
                cv.append(torch.mean(attn*similarity, dim=1, keepdim=True))
        
        return torch.cat(cv, axis=1)

def split_correlation(feat1, feat2, direction, d=4 ):
    assert direction == 'horizontal' or direction == 'vertical'
    h, w = feat1.size(2), feat1.size(3)
    cv = []
    if direction == 'horizontal':
        feat2 = torch.nn.functional.pad(feat2, [d,d,0,0], "constant", 0)
        for i in range(1):
            for j in range(2*d+1):
                cv.append(torch.mean(feat1*feat2[:,:,i:(i+h),j:(j+w)], dim=1, keepdim=True))
    else:
        feat2 = torch.nn.functional.pad(feat2, [0,0,d,d], "constant", 0)
        for i in range(2*d+1):
            for j in range(1):
                cv.append(torch.mean(feat1*feat2[:,:,i:(i+h),j:(j+w)], dim=1, keepdim=True))

    cv = torch.cat(cv, axis=1)
    cv = torch.nn.functional.leaky_relu(input=cv, negative_slope=0.1, inplace=False)

    return cv

# since = time.time()
# cv = correlation(feat1.cuda(), feat2.cuda())
# print(time.time() - since)

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bn=True, relu=True, dcn=False, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if self.use_bn: self.bn = nn.BatchNorm2d(out_channels)
        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        else:
            if dcn:
                self.conv = DeformableConv2d(in_channels, out_channels, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = torch.nn.functional.leaky_relu(x, inplace=True)
            # x = torch.nn.functional.relu(x, inplace=True)
        return x

class MatchingNet(nn.Module):
    # Matching net with 2D conv as mentioned in the paper
    def __init__(self, attention_list=None):
        super(MatchingNet, self).__init__()
        if attention_list != None:
            self.match = nn.Sequential(
                            BasicConv(64, 96, kernel_size=3, padding=1),
                            attention_list[0],
                            BasicConv(96, 128, kernel_size=3, padding=1, stride=2),   # down by 1/2
                            attention_list[1],
                            BasicConv(128, 128, kernel_size=3, padding=1),
                            attention_list[2],
                            BasicConv(128, 64, kernel_size=3, padding=1),
                            attention_list[3],
                            BasicConv(64, 32, kernel_size=4, padding=1, stride=2, deconv=True), # up by 1/2 
                            attention_list[4],
                            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=True),
                        )
        else:
            self.match = nn.Sequential(
                            BasicConv(64, 96, kernel_size=3, padding=1),
                            BasicConv(96, 128, kernel_size=3, padding=1, stride=2),   # down by 1/2
                            BasicConv(128, 128, kernel_size=3, padding=1),
                            BasicConv(128, 64, kernel_size=3, padding=1),
                            BasicConv(64, 32, kernel_size=4, padding=1, stride=2, deconv=True), # up by 1/2 
                            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=True),
                        )

    def forward(self, x):
            x = self.match(x)
            return x

class MatchingNetSmall(nn.Module):
    # Matching net with 2D conv as mentioned in the paper
    def __init__(self, dcn=False):
        super(MatchingNetSmall, self).__init__()
        self.match = nn.Sequential(
                        BasicConv(32, 48, kernel_size=3, padding=1, dcn=dcn),
                        BasicConv(48, 96, kernel_size=3, padding=1, stride=2, dcn=dcn),   # down by 1/2
                        BasicConv(96, 96, kernel_size=3, padding=1, dcn=dcn),
                        BasicConv(96, 48, kernel_size=3, padding=1, dcn=dcn),
                        BasicConv(48, 32, kernel_size=4, padding=1, stride=2, deconv=True), # up by 1/2 
                        nn.Conv2d(32, 1  , kernel_size=3, padding=1, bias=True),
                    )

    def forward(self, x):
            x = self.match(x)
            return x

class MatchingNetSmallRes(nn.Module):
    # Matching net with 2D conv as mentioned in the paper
    def __init__(self, attn_list=None, dcn=False):
        super(MatchingNetSmallRes, self).__init__()
        self.match_list = nn.ModuleList([
                        BasicConv(32, 48, kernel_size=3, padding=1, dcn=dcn),
                        BasicConv(48, 96, kernel_size=3, padding=1, stride=2, dcn=dcn),   # down by 1/2
                        BasicConv(96, 96, kernel_size=3, padding=1, dcn=dcn),
                        BasicConv(96, 48, kernel_size=3, padding=1, dcn=dcn),
                        BasicConv(48, 32, kernel_size=4, padding=1, stride=2, deconv=True), # up by 1/2 
                        nn.Conv2d(32, 1  , kernel_size=3, padding=1, bias=True),
        ])
        if attn_list != None:
            self.bam = attn_list

        # self.conv1x1_96 = nn.Conv2d(32, 96, kernel_size=1, bias=False, stride=2)
        # self.conv1x1_48 = nn.Conv2d(96, 48, kernel_size=1, bias=False)
        # self.conv1x1 = nn.Conv2d(32, 1, kernel_size=1, bias=False, stride=1)

    def forward(self, x):
        identity = x
        for i in range(len(self.match_list)):
            x = self.match_list[i](x)
            if hasattr(self, 'bam') and i < len(self.match_list)-1:
                x = self.bam[i](x)
            if i == len(self.match_list)-2:
                x = x + identity
        return x

class MatchingResNet(nn.Module):
    # Matching net with 2D conv as mentioned in the paper
    def __init__(self, dcn=False):
        super(MatchingResNet, self).__init__()
        self.conv1 = BasicConv(32, 32, kernel_size=3, padding=1, dcn=dcn)
        self.conv2 = BasicConv(32, 32, kernel_size=3, padding=1, dcn=dcn)
        self.conv3 = nn.Conv2d(32, 1 ,kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.conv3(x)
        return x

class MatchingNetDeep(nn.Module):
    # Matching net with 2D conv as mentioned in the paper
    def __init__(self, dcn=False):
        super(MatchingNetDeep, self).__init__()
        self.match = nn.Sequential(
                        BasicConv(32, 48, kernel_size=3, padding=1, dcn=dcn),
                        BasicConv(48, 96, kernel_size=3, padding=1, stride=2, dcn=dcn),   # down by 1/2
                        BasicConv(96, 96, kernel_size=3, padding=1, dcn=dcn),
                        BasicConv(96, 96, kernel_size=3, padding=1, dcn=dcn),
                        BasicConv(96, 48, kernel_size=3, padding=1, dcn=dcn),
                        BasicConv(48, 32, kernel_size=4, padding=1, stride=2, deconv=True), # up by 1/2 
                        nn.Conv2d(32, 1  , kernel_size=3, padding=1, bias=True),
                    )

    def forward(self, x):
            x = self.match(x)
            return x

class MatchingNetSmallAttn(nn.Module):
    # Matching net with 2D conv as mentioned in the paper
    def __init__(self, attention_list):
        super(MatchingNetSmallAttn, self).__init__()
        self.match = nn.Sequential(
                        BasicConv(32, 48, kernel_size=3, padding=1,   dilation=1),
                        attention_list[0],
                        BasicConv(48, 96, kernel_size=3, stride=2,    padding=1),   # down by 1/2
                        attention_list[1],
                        BasicConv(96, 96, kernel_size=3, padding=1,   dilation=1),
                        attention_list[2],
                        BasicConv(96, 48, kernel_size=3, padding=1,   dilation=1),
                        attention_list[3],
                        BasicConv(48, 32, kernel_size=4, padding=1, stride=2, deconv=True), # up by 1/2 
                        attention_list[4],
                        nn.Conv2d(32, 1  , kernel_size=3, stride=1, padding=1, bias=True),
                    )

    def forward(self, x):
            x = self.match(x)
            return x


def compute_cost(x,y, matchnet, md=3):
    sizeU = 2*md+1
    sizeV = 2*md+1
    b,c,height,width = x.shape

    with torch.cuda.device_of(x):
        # init cost as tensor matrix
        cost = x.new().resize_(x.size()[0], 2*c, 2*md+1,2*md+1, height,  width).zero_()

        for i in range(2*md+1):
            ind = i-md
            for j in range(2*md+1):
                indd = j-md
                # for each displacement hypothesis, we construct a feature map as the input of matching net
                # here we hold them together for parallel processing later
                cost[:,:c,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = x[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
                cost[:,c:,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = y[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]

        # (B, 2C, U, V, H, W) -> (B, U, V, 2C, H, W)
        cost = cost.permute([0,2,3,1,4,5]).contiguous() 
        # (B, U, V, 2C, H, W) -> (BxUxV, 2C, H, W)
        cost = cost.view(x.size()[0]*sizeU*sizeV,c*2, x.size()[2], x.size()[3])
        cost = matchnet(cost)
        # (BxUxV, 2C, H, W) -> (BxUxV, 1, H, W)
        cost = cost.view(x.size()[0],sizeU,sizeV,1, x.size()[2],x.size()[3])
        cost = cost.permute([0,3,1,2,4,5]).contiguous() 

        # (B, U, V, H, W)
        return cost

def compute_dc_cost(x,y, matchnet, md=3, relation='dot-product'):
    sizeU = 2*md+1
    sizeV = 2*md+1
    b,c,height,width = x.shape
    with torch.cuda.device_of(x):
        # init cost as tensor matrix
        # cost = x.new().resize_(x.size()[0], c, 2*md+1, 2*md+1, height, width).zero_()

        # for i in range(2*md+1):
        #     ind = i-md
        #     for j in range(2*md+1):
        #         indd = j-md
        #         # for each displacement hypothesis, we construct a feature map as the input of matching net
        #         # here we hold them together for parallel processing later
        #         # cost[:,:,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = torch.dot(
        #         #     x[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind], y[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]
        #         # )
        #         t = torch.dot(
        #             x[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind], y[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]
        #         )
        y = torch.nn.functional.unfold(y, (sizeU, sizeV), padding=md).view(b,c, sizeU, sizeV, height, width)
        cost = x.unsqueeze(2).unsqueeze(3)
        if relation == 'dot-product':
            cost = cost * y.view(b,c, sizeU, sizeV, height, width)
        elif relation == 'subtraction':
            cost = cost - y.view(b,c, sizeU, sizeV, height, width)

        # (B, 2C, U, V, H, W) -> (B, U, V, 2C, H, W)
        cost = cost.permute([0,2,3,1,4,5]).contiguous() 
        # (B, U, V, 2C, H, W) -> (BxUxV, 2C, H, W)
        cost = cost.view(x.size()[0]*sizeU*sizeV, c, x.size()[2], x.size()[3])
        cost = matchnet(cost)
        # (BxUxV, 2C, H, W) -> (BxUxV, 1, H, W)
        cost = cost.view(x.size()[0], sizeU, sizeV, 1, x.size()[2],x.size()[3])
        cost = cost.permute([0,3,1,2,4,5]).contiguous() 

        # (B, U, V, H, W)
        return cost

def compute_stereo_cost(x,y, matchnet, md=8):
    sizeU = 2*md+1
    sizeV = 2*1+1
    b,c,height,width = x.shape

    with torch.cuda.device_of(x):
        # init cost as tensor matrix
        cost = x.new().resize_(x.size()[0], 2*c, sizeU, sizeV, height, width).zero_()

        for i in range(sizeU):
            ind = i-md
            for j in range(sizeV):
                indd = j-1
                # for each displacement hypothesis, we construct a feature map as the input of matching net
                # here we hold them together for parallel processing later
                cost[:,:c,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = x[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
                cost[:,c:,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = y[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]

        # (B, 2C, U, V, H, W) -> (B, U, V, 2C, H, W)
        cost = cost.permute([0,2,3,1,4,5]).contiguous() 
        # (B, U, V, 2C, H, W) -> (BxUxV, 2C, H, W)
        cost = cost.view(x.size()[0]*sizeU*sizeV, c*2, x.size()[2], x.size()[3])
        cost = matchnet(cost)
        # (BxUxV, 2C, H, W) -> (BxUxV, 1, H, W)
        cost = cost.view(x.size()[0], sizeU, sizeV,1, x.size()[2], x.size()[3])
        cost = cost.permute([0,3,1,2,4,5]).contiguous() 

        # (B, U, V, H, W)
        return cost