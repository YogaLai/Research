import torch
import torch.nn as nn
from models.lib.dcn import DeformableConv2d
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

# class AttnCorrelation(nn.Module):
#     def __init__(self, in_channels, d=4, reduce_factor=8):
#         super(AttnCorrelation,self).__init__()
#         self.d = d
#         self.conv_q = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
#         self.conv_k = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
    
#     def forward(self, feat1, feat2):
#         h, w = feat1.size(2), feat1.size(3)
#         feat1 = self.conv_q(feat1)
#         feat2 = self.conv_k(feat2)
#         cv = []
#         feat2 = torch.nn.functional.pad(feat2, [self.d,self.d,self.d,self.d], "constant", 0)
#         for i in range(2*self.d+1):
#             for j in range(2*self.d+1):
#                 corr = torch.nn.functional.softmax(torch.mean(feat1*feat2[:,:,i:(i+h),j:(j+w)], dim=1, keepdim=True))
#                 cv.append(corr)
        
#         return torch.cat(cv, axis=1)

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
        return x

class MatchingNet(nn.Module):
    # Matching net with 2D conv as mentioned in the paper
    def __init__(self):
        super(MatchingNet, self).__init__()
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