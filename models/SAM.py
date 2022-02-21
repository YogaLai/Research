import torch
import torch.nn as nn

from models.lib.modules import Aggregation


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


class SAM2(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM2, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if sa_type == 0:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            # self.pad = nn.ReflectionPad2d(kernel_size // 2)
            self.pad = nn.ZeroPad2d(kernel_size // 2)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=0)  # modify reflect pad to zero pad

    def forward(self, feat1, feat2):
        x1, x2, x3 = self.conv1(feat1), self.conv2(feat2), self.conv3(feat1)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))    # [B, 2, H, W]
            w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
            # w1 = self.subtraction2(x1, x2)  # [B, C, 9, H*W]
            # w2 = self.subtraction(p).repeat(x.shape[0], 1, 1, 1) # [B, 2, 9, H*W]
            # w = self.conv_w(torch.cat([w1, w2], 1))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(feat1.shape[0], -1, 1, feat1.shape[2]*feat1.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(feat1.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(feat1.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
            # w = self.conv_w(torch.cat([x1, x2], 1))
            # w = w.view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        return x

class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if sa_type == 0:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
            w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        return x

class VecAttn(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes=1, share_planes=8, kernel_size=3, stride=1, md=4):
        super(VecAttn, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        # self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.sam_layers = nn.ModuleList([SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride) for i in range(2)])
        # self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.md = md
        # self.stride = stride

    def forward(self, x):
        # _, c, height, width = feat1.shape
        # with torch.cuda.device_of(feat1):
        #     # init cost as tensor matrix
        #     cost = feat1.new().resize_(feat1.size()[0], 2*c, 2*self.md+1,2*self.md+1, height,  width).zero_().requires_grad_(False)

        # for i in range(2*self.md+1):
        #     ind = i-self.md
        #     for j in range(2*self.md+1):
        #         indd = j-self.md
        #         # for each displacement hypothesis, we construct a feature map as the input of matching net
        #         # here we hold them together for parallel processing later
        #         cost[:,:c,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = feat1[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
        #         cost[:,c:,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = feat2[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]
        
        # # (B, 2C, U, V, H, W) -> (B, U, V, 2C, H, W)
        # size_d = 2 * self.md + 1
        # cost = cost.permute([0,2,3,1,4,5]).contiguous() 
        # # (B, U, V, 2C, H, W) -> (BxUxV, 2C, H, W)
        # cost = cost.view(feat1.size()[0]*size_d*size_d, c*2, feat1.size()[2], feat1.size()[3])
        # # (BxUxV, 2C, H, W) -> (BxUxV, 1, H, W)
        # cost = self.sam(cost)
        # cost = self.conv(cost)
        # cost = cost.view(feat1.size()[0],size_d*size_d, feat1.size()[2],feat1.size()[3])
        # cost = cost.contiguous() 

        # return cost
        # h, w = feat1.size(2), feat1.size(3)
        # cv = []
        # feat2 = torch.nn.functional.pad(feat2, [self.md,self.md,self.md,self.md], "constant", 0)
        # for i in range(2*self.md+1):
        #     for j in range(2*self.md+1):
        #         out = self.relu(self.bn2(self.sam(feat1, feat2[:,:,i:(i+h),j:(j+w)])))
        #         out = self.conv(out)
        #         cv.append(out)
                
        # return torch.cat(cv, axis=1)
        for i in range(2):
            out = self.sam_layers[i](x)
        out = self.conv(out)
        return out


