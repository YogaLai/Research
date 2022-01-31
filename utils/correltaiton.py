import torch
import torch.nn as nn

# feat1 = torch.ones([8,196,7,16])
# feat2 = feat1.clone()

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
      
class AttnCorrelation(nn.Module):
    def __init__(self, in_channels, d=4, reduce_factor=8):
        super(AttnCorrelation,self).__init__()
        self.d = d
        self.conv_q = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
    
    def forward(self, feat1, feat2):
        h, w = feat1.size(2), feat1.size(3)
        feat1 = self.conv_q(feat1)
        feat2 = self.conv_k(feat2)
        cv = []
        feat2 = torch.nn.functional.pad(feat2, [self.d,self.d,self.d,self.d], "constant", 0)
        for i in range(2*self.d+1):
            for j in range(2*self.d+1):
                corr = torch.nn.functional.softmax(torch.mean(feat1*feat2[:,:,i:(i+h),j:(j+w)], dim=1, keepdim=True))
                cv.append(corr)
        
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
    def __init__(self, in_channels, out_channels, deconv=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if self.use_bn: self.bn = nn.BatchNorm2d(out_channels)
        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = torch.nn.functional.relu(x, inplace=True)
        return x

class MatchingNet(nn.Module):
    # Matching net with 2D conv as mentioned in the paper
    def __init__(self):
        super(MatchingNet, self).__init__()
        self.match = nn.Sequential(
                        BasicConv(64, 96, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(96, 128, kernel_size=3, stride=2,    padding=1),   # down by 1/2
                        BasicConv(128, 128, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(128, 64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64, 32, kernel_size=4, padding=1, stride=2, deconv=True), # up by 1/2 
                        nn.Conv2d(32, 1  , kernel_size=3, stride=1, padding=1, bias=True),
                    )

    def forward(self, x):
            x = self.match(x)
            return x

def DICL_cost(x,y,matching_net, max_disp=3):
    maxU = maxV = max_disp
    sizeU = 2*maxU+1
    sizeV = 2*maxV+1
    b,c,height,width = x.shape

    # with torch.cuda.device_of(x):
        # init cost as tensor matrix
    cost = torch.zeros([x.size()[0], 2*c, 2*maxU+1,2*maxV+1, height,  width], device=x.device).requires_grad_(False)


    # if cfg.CUDA_COST:
    #     # CUDA acceleration
    #     corr = SpatialCorrelationSampler(kernel_size=1,patch_size=(int(1+2*3),int(1+2*3)),stride=1,padding=0,dilation_patch=1)
    #     cost = corr(x, y)
    # else:
    for i in range(2*maxU+1):
        ind = i-maxU
        for j in range(2*maxV+1):
            indd = j-maxV
            # for each displacement hypothesis, we construct a feature map as the input of matching net
            # here we hold them together for parallel processing later
            cost[:,:c,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = x[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
            cost[:,c:,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = y[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]

    # if cfg.REMOVE_WARP_HOLE:
    #     # mitigate the effect of holes (may be raised by occ)
    #     valid_mask = cost[:,c:,...].sum(dim=1)!=0
    #     valid_mask = valid_mask.detach()
    #     cost = cost*valid_mask.unsqueeze(1).float()

    # (B, 2C, U, V, H, W) -> (B, U, V, 2C, H, W)
    cost = cost.permute([0,2,3,1,4,5]).contiguous() 
    # (B, U, V, 2C, H, W) -> (BxUxV, 2C, H, W)
    cost = cost.view(x.size()[0]*sizeU*sizeV,c*2, x.size()[2], x.size()[3])
    # (BxUxV, 2C, H, W) -> (BxUxV, 1, H, W)
    cost = matching_net(cost)
    # cost = cost.view(x.size()[0],sizeU,sizeV,1, x.size()[2],x.size()[3])
    # cost = cost.permute([0,3,1,2,4,5]).contiguous() 
    cost = cost.view(x.size(0), sizeU*sizeV, x.size(2), x.size(3)).contiguous()

    # (B, U, V, H, W)
    return cost