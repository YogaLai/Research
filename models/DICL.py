from imp import SEARCH_ERROR
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import sys
import torch.nn.functional as F
import numpy as np
from config import cfg
import pdb

from models.SAM import VecAttn
from models.lib.Bottleneck import Bottleneck
# from spatial_correlation_sampler import SpatialCorrelationSampler


__all__ = ['dicl_wrapper']

def dicl_wrapper(data=None):
    # Wrapper for the DICL network
    model = DICL()
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model


class DICL_MODULE(nn.Module):
    # Matching net with 2D conv as mentioned in the paper
    def __init__(self):
        super(DICL_MODULE, self).__init__()

        self.match = nn.Sequential(
                        BasicConv(64, 96, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(96, 128, kernel_size=3, stride=2,    padding=1),   # down by 1/2
                        BasicConv(128, 128, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(128, 64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64, 32, kernel_size=4, padding=1, stride=2, deconv=True), # up by 1/2 
                        nn.Conv2d(32, 1  , kernel_size=3, stride=1, padding=1, bias=True),)

    def forward(self, x):
        x = self.match(x)
        return x
      

class FlowEntropy(nn.Module):
    # Compute entro from matching cost
    def __init__(self):
        super(FlowEntropy, self).__init__()

    def forward(self, x):
        x = torch.squeeze(x, 1)
        B,U,V,H,W = x.shape
        x = x.view(B,-1,H,W)
        x = F.softmax(x,dim=1).view(B,U,V,H,W)
        global_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
        global_entropy /= np.log(x.shape[1]*x.shape[2])
        return global_entropy

class FlowRegression(nn.Module):
    # 2D soft argmin/argmax
    def __init__(self, maxU, maxV, plot_prob=True):
        super(FlowRegression, self).__init__()
        self.maxU = maxU
        self.maxV = maxV
        self.plot_prob = plot_prob

    def forward(self, x):
        assert(x.is_contiguous() == True)
        sizeU = 2*self.maxU+1
        sizeV = 2*self.maxV+1
        x = x.squeeze(1)
        B,_,_,H,W = x.shape

        with torch.cuda.device_of(x):
            # displacement along u 
            dispU = torch.reshape(torch.arange(-self.maxU, self.maxU+1,device=torch.cuda.current_device(), dtype=torch.float32),[1,sizeU,1,1,1])
            dispU = dispU.expand(B, -1, sizeV, H,W).contiguous()
            dispU = dispU.view(B,sizeU*sizeV , H,W)

            # displacement along v
            dispV = torch.reshape(torch.arange(-self.maxV, self.maxV+1,device=torch.cuda.current_device(), dtype=torch.float32),[1,1,sizeV,1,1])
            dispV = dispV.expand(B,sizeU, -1, H,W).contiguous()
            dispV = dispV.view(B,sizeU*sizeV,H,W)
            
        x = x.view(B,sizeU*sizeV,H,W)


        if cfg.FLOW_REG_BY_MAX:
            x = F.softmax(x,dim=1)
        else:
            x = F.softmin(x,dim=1)
        if self.plot_prob:
            grid_x = np.linspace(-self.maxU, self.maxU, 7)
            grid_y = np.linspace(-self.maxV, self.maxV, 7)
            X, Y = np.meshgrid(grid_x,grid_y)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.gca(projection='3d')
            h = random.randint(0, x.size(2))
            w = random.randint(0, x.size(3))
            Z = x[0,:,h,w].view(sizeU, sizeV).cpu().data.numpy()
            ax.plot_surface(X, Y, Z,cmap='Reds',linewidth=0, antialiased=True, zorder = 0.5)
            plt.savefig('visualization/cost_prob_distribution')

        flowU = (x*dispU).sum(dim=1)
        flowV = (x*dispV).sum(dim=1)
        flow  = torch.cat((flowU.unsqueeze(1),flowV.unsqueeze(1)),dim=1)
        return flow

class DAP(nn.Module):
    def __init__(self, md=3):
        # Displacement-aware projection layer
        # implemented as a 1x1 2D conv
        super(DAP, self).__init__()
        dimC = (2*md+1)**2
        self.dap_layer = BasicConv(dimC, dimC, kernel_size=1, padding=0, stride=1, bn=False, relu=False)
        if cfg.DAP_BY_TEMPERATURE:
            self.dap_layer = BasicConv(dimC, 1, kernel_size=1, padding=0, stride=1, bn=False, relu=False)

    def forward(self, x):
        x = x.squeeze(1)
        bs,du,dv,h,w = x.shape
        x = x.view(bs,du*dv,h,w)

        if cfg.DAP_BY_TEMPERATURE:
            temp = self.dap_layer(x)+1e-6
            x = x*temp
        else:
            x = self.dap_layer(x)

        return x.view(bs,du,dv,h,w).unsqueeze(1)  



class DICL(nn.Module):
    def __init__(self,):
        super(DICL,self).__init__()
        self.feature = FeatureGA()
        # self.slices = slices
        # self.feature = FeatureExtractionPWC()

        # if cfg.DAP_LAYER:
        if True:
            self.dap_layer6 = DAP(md=cfg.SEARCH_RANGE[4])
            self.dap_layer5 = DAP(md=cfg.SEARCH_RANGE[3])
            self.dap_layer4 = DAP(md=cfg.SEARCH_RANGE[2])
            self.dap_layer3 = DAP(md=cfg.SEARCH_RANGE[1])
            self.dap_layer2 = DAP(md=cfg.SEARCH_RANGE[0])
        else:
            self.dap_layer6 = self.dap_layer5 = self.dap_layer4 = self.dap_layer3 = self.dap_layer2 = None

        self.entropy = FlowEntropy()

        # matching net, with the same arch at each pyramid level
        # level 6->2:  1/64,1/32,1/16,1/8,1/4
        self.matching6 = DICL_MODULE()
        self.matching5 = DICL_MODULE()
        self.matching4 = DICL_MODULE()
        self.matching3 = DICL_MODULE()
        self.matching2 = DICL_MODULE()

        # search range, e.g., [-3,3]
        # the search range for FlowRegression should be aligned with that when computing matching cost
        self.flow6 = FlowRegression(cfg.SEARCH_RANGE[4],cfg.SEARCH_RANGE[4])
        self.flow5 = FlowRegression(cfg.SEARCH_RANGE[3],cfg.SEARCH_RANGE[3])
        self.flow4 = FlowRegression(cfg.SEARCH_RANGE[2],cfg.SEARCH_RANGE[2])
        self.flow3 = FlowRegression(cfg.SEARCH_RANGE[1],cfg.SEARCH_RANGE[1])
        self.flow2 = FlowRegression(cfg.SEARCH_RANGE[0],cfg.SEARCH_RANGE[0])


        if cfg.CTF_CONTEXT: 
            # If you would use context network as introduced in PWCNet
            self.context_net2 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64, 128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 96 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(96,  64 , kernel_size=3, padding=16,  dilation=16),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
            self.context_net3 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64,  128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 96 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(96,  64 , kernel_size=3, padding=16,  dilation=16),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
            self.context_net4 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64,  128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 64 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
            self.context_net5 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64,  128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 64, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
            # We remove the last several layers on pyramid level 6,5,4 considering their inputs' resolution
            # though this does not have an obvious effect
            self.context_net6 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64 , 64, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)) or isinstance(m, nn.ConvTranspose2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if cfg.DAP_INIT_BY_ID:
            # init the dap layer kernel by identity matrix
            if self.dap_layer6 is not None:
                nn.init.eye_(self.dap_layer6.dap_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.dap_layer5 is not None:
                nn.init.eye_(self.dap_layer5.dap_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.dap_layer4 is not None:
                nn.init.eye_(self.dap_layer4.dap_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.dap_layer3 is not None:
                nn.init.eye_(self.dap_layer3.dap_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.dap_layer2 is not None:
                nn.init.eye_(self.dap_layer2.dap_layer.conv.weight.squeeze(-1).squeeze(-1))


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask, mask

    def upsample_flow(self, flow, h, w):
        flow = F.interpolate(flow, (h,w), mode='bilinear', align_corners=True)
        flow[:,0,:,:] = flow[:,0,:,:]*(w/flow.shape[3])
        flow[:,1,:,:] = flow[:,1,:,:]*(h/flow.shape[2])

        return flow

    def forward(self, images):
        # left and right images normalized
        x = images[:,:3,:,:]
        y = images[:,3:,:,:]
        x = (x-0.5)/0.5
        y = (y-0.5)/0.5
        h,w = images.shape[2], images.shape[3]
        
        # feature extraction
        _,x2,x3,x4,x5,x6 = self.feature(x)       
        _,y2,y3,y4,y5,y6 = self.feature(y)
        # x2,x3,x4,x5,x6 = self.feature(x)
        # y2,y3,y4,y5,y6 = self.feature(y)

        # compute flow for level 6
        # cost6 = self.compute_cost(x6,y6,self.matching2,cfg.SEARCH_RANGE[4],cfg.SEARCH_RANGE[4])
        cost6 = self.compute_cost(x6,y6,self.matching6,cfg.SEARCH_RANGE[4],cfg.SEARCH_RANGE[4])
        g6 = F.interpolate(images[:,:3,:,:],scale_factor=(1/64), mode='bilinear',align_corners=True)
        if self.dap_layer6 is not None: cost6 = self.dap_layer6(cost6)
        flow6 = self.flow6(cost6)
        
        if cfg.CTF_CONTEXT:
            if cfg.SUP_RAW_FLOW: raw_flow6 = flow6
            entro6 = self.entropy(cost6)
            # input to context network
            # including pred flow, entro, left img feat, left img
            feat6 = torch.cat((flow6.detach(),entro6.detach(),x6,g6),dim=1)
            flow6 = flow6 + self.context_net6(feat6)*cfg.SCALE_CONTEXT6
        up_flow6 = 2.0*F.interpolate(flow6, [x5.shape[2],x5.shape[3]], mode='bilinear',align_corners=True)
        up_flow6 = up_flow6.detach()

         # compute flow for level 5
        warp5,_ = self.warp(y5,up_flow6)
        cost5 = self.compute_cost(x5,warp5,self.matching5,cfg.SEARCH_RANGE[3],cfg.SEARCH_RANGE[3])
        g5 = F.interpolate(images[:,:3,:,:],scale_factor=(1/32), mode='bilinear',align_corners=True)
        if self.dap_layer5 is not None: cost5 = self.dap_layer5(cost5)
        flow5 = self.flow5(cost5) + up_flow6
        if cfg.CTF_CONTEXT:
            if cfg.SUP_RAW_FLOW: raw_flow5 = flow5
            entro5 = self.entropy(cost5)
            feat5 = torch.cat((flow5.detach(),entro5.detach(),x5,g5),dim=1) 
            flow5 = flow5 + self.context_net5(feat5)*cfg.SCALE_CONTEXT5
        up_flow5 = 2.0*F.interpolate(flow5, [x4.shape[2],x4.shape[3]], mode='bilinear',align_corners=True)
        up_flow5 = up_flow5.detach()

         # compute flow for level 4
        warp4,_ = self.warp(y4,up_flow5)
        cost4 = self.compute_cost(x4,warp4,self.matching4,cfg.SEARCH_RANGE[2],cfg.SEARCH_RANGE[2])
        g4 = F.interpolate(images[:,:3,:,:],scale_factor=(1/16), mode='bilinear',align_corners=True)
        if self.dap_layer4 is not None: cost4 = self.dap_layer4(cost4)
        flow4 = self.flow4(cost4) + up_flow5
        if cfg.CTF_CONTEXT:
            if cfg.SUP_RAW_FLOW: raw_flow4 = flow4
            entro4 = self.entropy(cost4)
            feat4 = torch.cat((flow4.detach(),entro4.detach(),x4,g4),dim=1) 
            flow4 = flow4 + self.context_net4(feat4)*cfg.SCALE_CONTEXT4
        up_flow4 = 2.0*F.interpolate(flow4, [x3.shape[2],x3.shape[3]], mode='bilinear',align_corners=True)
        up_flow4 = up_flow4.detach()

         # compute flow for level 3
        warp3,_ = self.warp(y3,up_flow4)
        cost3 = self.compute_cost(x3,warp3,self.matching3,cfg.SEARCH_RANGE[1],cfg.SEARCH_RANGE[1])
        g3 = F.interpolate(images[:,:3,:,:],scale_factor=(1/8), mode='bilinear',align_corners=True)
        if self.dap_layer3 is not None: cost3 = self.dap_layer3(cost3)
        flow3 = self.flow3(cost3) + up_flow4
        if cfg.CTF_CONTEXT:
            if cfg.SUP_RAW_FLOW: raw_flow3 = flow3
            entro3 = self.entropy(cost3)
            feat3 = torch.cat((flow3.detach(),entro3.detach(),x3,g3),dim=1) 
            flow3 = flow3 + self.context_net3(feat3)*cfg.SCALE_CONTEXT3
        up_flow3 = 2.0*F.interpolate(flow3, [x2.shape[2],x2.shape[3]], mode='bilinear',align_corners=True)
        up_flow3 = up_flow3.detach()

         # compute flow for level 2
        warp2,_ = self.warp(y2,up_flow3)
        cost2 = self.compute_cost(x2,warp2,self.matching2,cfg.SEARCH_RANGE[0],cfg.SEARCH_RANGE[0])
        g2 = F.interpolate(images[:,:3,:,:],scale_factor=(1/4), mode='bilinear',align_corners=True)
        if self.dap_layer2 is not None: cost2 = self.dap_layer2(cost2)
        flow2 = self.flow2(cost2) + up_flow3

        if cfg.CTF_CONTEXT or cfg.CTF_CONTEXT_ONLY_FLOW2:
            if cfg.SUP_RAW_FLOW: raw_flow2 = flow2
        entro2 = self.entropy(cost2)
        feat2 = torch.cat((flow2.detach(),entro2.detach(),x2,g2),dim=1) 
        flow2 = flow2 + self.context_net2(feat2)*cfg.SCALE_CONTEXT2


        flow2 = self.upsample_flow(flow2, h, w)
        flow3 = self.upsample_flow(flow3, h//2, w//2)
        flow4 = self.upsample_flow(flow4, h//4, w//4)
        flow5 = self.upsample_flow(flow5, h//8, w//8)
        

        # if cfg.SUP_RAW_FLOW: 
        #     return flow2, raw_flow2, flow3, raw_flow3, flow4, raw_flow4, flow5, raw_flow5, flow6,raw_flow6
        return flow2, flow3, flow4, flow5
        # return flow2, flow3, flow4, flow5, flow6


        
    def compute_cost(self, x,y,matchnet,maxU,maxV):
        sizeU = 2*maxU+1
        sizeV = 2*maxV+1
        b,c,height,width = x.shape

        if maxV == 0:
            with torch.cuda.device_of(x):
                # init cost as tensor matrix
                cost = x.new().resize_(x.size()[0], 2*c, 2*maxU+1, height, width).zero_().requires_grad_(False)
                for i in range(2*maxU+1):
                    ind = i-maxU
                    # for each displacement hypothesis, we construct a feature map as the input of matching net
                    # here we hold them together for parallel processing later
                    cost[:,:c,i,:,max(0,-ind):width-ind] = x[:,:,:,max(0,-ind):width-ind]
                    cost[:,c:,i,:,max(0,-ind):width-ind] = y[:,:,:,max(0,ind):width+ind]

            # (B, 2C, U, H, W) -> (B, U, 2C, H, W)
            cost = cost.permute([0,2,1,3,4]).contiguous() 
            # (B, U, V, 2C, H, W) -> (BxUxV, 2C, H, W)
            cost = cost.view(x.size()[0]*sizeU,c*2, x.size()[2], x.size()[3])
            # (BxUxV, 2C, H, W) -> (BxUxV, 1, H, W)
            cost = matchnet(cost)
            cost = cost.view(x.size()[0],sizeU,1, x.size()[2],x.size()[3])
            cost = cost.permute([0,2,1,3,4]).contiguous() 

        else:
            with torch.cuda.device_of(x):
                # init cost as tensor matrix
                cost = x.new().resize_(x.size()[0], 2*c, 2*maxU+1,2*maxV+1, height,  width).zero_().requires_grad_(False)


            for i in range(2*maxU+1):
                ind = i-maxU
                for j in range(2*maxV+1):
                    indd = j-maxV
                    # for each displacement hypothesis, we construct a feature map as the input of matching net
                    # here we hold them together for parallel processing later
                    cost[:,:c,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = x[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
                    cost[:,c:,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = y[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]

            if cfg.REMOVE_WARP_HOLE:
                # mitigate the effect of holes (may be raised by occ)
                valid_mask = cost[:,c:,...].sum(dim=1)!=0
                valid_mask = valid_mask.detach()
                cost = cost*valid_mask.unsqueeze(1).float()

            # (B, 2C, U, V, H, W) -> (B, U, V, 2C, H, W)
            cost = cost.permute([0,2,3,1,4,5]).contiguous() 
            # (B, U, V, 2C, H, W) -> (BxUxV, 2C, H, W)
            cost = cost.view(x.size()[0]*sizeU*sizeV,c*2, x.size()[2], x.size()[3])
            # (BxUxV, 2C, H, W) -> (BxUxV, 1, H, W)
            cost = matchnet(cost)
            cost = cost.view(x.size()[0],sizeU,sizeV,1, x.size()[2],x.size()[3])
            cost = cost.permute([0,3,1,2,4,5]).contiguous() 

        # (B, U, V, H, W)
        return cost



######################## Part of the Model #############################################


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        # To keep the shape of input and output same when dilation conv, we should compute the padding:
        # Reference:
        #   https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        # padding = [(o-1)*s+k+(k-1)*(d-1)-i]/2, here the i is input size, and o is output size.
        # set o = i, then padding = [i*(s-1)+k+(k-1)*(d-1)]/2 = [k+(k-1)*(d-1)]/2      , stride always equals 1
        # if dilation != 1:
        #     padding = (3+(3-1)*(dilation-1))/2
        padding = dilation

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        out = self.relu2(out)

        return out

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
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.deconv = deconv
        if deconv:
            kernel = 4
        else:
            kernel = 3

        self.conv1 = BasicConv(in_channels, out_channels, deconv, bn=False, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            self.conv2 = BasicConv(out_channels*2, out_channels, False, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert(x.size() == rem.size())

        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x

def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=True),
        nn.LeakyReLU(0.1)
    )

class FeatureExtractionPWC(nn.Module):
    def __init__(self):
        super(FeatureExtractionPWC, self).__init__()
        self.conv1a  = myconv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = myconv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = myconv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = myconv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = myconv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = myconv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = myconv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = myconv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = myconv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = myconv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = myconv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = myconv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = myconv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = myconv(128,128, kernel_size=3, stride=1)
        self.conv5b  = myconv(128,128, kernel_size=3, stride=1)
        self.conv6aa = myconv(128,192, kernel_size=3, stride=2)
        self.conv6a  = myconv(192,192, kernel_size=3, stride=1)
        self.conv6b  = myconv(192,192, kernel_size=3, stride=1)

        self.outconv_2 = myconv(32, 32, kernel_size=3,  padding=1)
        self.outconv_3 = myconv(64, 32, kernel_size=3,  padding=1)
        self.outconv_4 = myconv(96, 32, kernel_size=3,  padding=1)
        self.outconv_5 = myconv(128, 32, kernel_size=3,  padding=1)
        self.outconv_6 = myconv(192, 32, kernel_size=3,  padding=1)

    def forward(self, x):
        c11 = self.conv1b(self.conv1aa(self.conv1a(x)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))

        c12 = self.outconv_2(c12)
        c13 = self.outconv_3(c13)
        c14 = self.outconv_4(c14)
        c15 = self.outconv_5(c15)
        c16 = self.outconv_6(c16)
        
        return c12, c13, c14, c15, c16

class FeatureGA(nn.Module):
    def __init__(self):
        super(FeatureGA, self).__init__()
        # feature backbone
        # adopted from GANet (https://github.com/feihuzhang/GANet)
        self.conv_start = nn.Sequential(
            BasicConv(3, 16, kernel_size=3, padding=1),
            BasicConv(16, 16, kernel_size=3, stride=2, padding=1),
            BasicConv(16, 16, kernel_size=3, padding=1))
        self.conv1a = BasicConv(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)
        self.conv5a = BasicConv(128, 160, kernel_size=3, stride=2, padding=1)
        self.conv6a = BasicConv(160, 192, kernel_size=3, stride=2, padding=1)

        self.deconv6a = Conv2x(192, 160, deconv=True)
        self.deconv5a = Conv2x(160, 128, deconv=True)
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 32, deconv=True)
        self.deconv1a = Conv2x(32, 16, deconv=True)

        self.conv1b = Conv2x(16, 32)
        self.conv2b = Conv2x(32, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)
        self.conv5b = Conv2x(128, 160)
        self.conv6b = Conv2x(160, 192)

        self.deconv6b = Conv2x(192,160, deconv=True)
        self.outconv_6 = BasicConv(160, 32, kernel_size=3,  padding=1)

        self.deconv5b = Conv2x(160,128, deconv=True)
        self.outconv_5 = BasicConv(128, 32, kernel_size=3,  padding=1)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.outconv_4 = BasicConv(96, 32, kernel_size=3,  padding=1)

        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.outconv_3 = BasicConv(64, 32, kernel_size=3,  padding=1)

        self.deconv2b = Conv2x(64, 32, deconv=True)
        self.outconv_2 = BasicConv(32, 32, kernel_size=3,  padding=1)


    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.conv5a(x)
        rem5 = x
        x = self.conv6a(x)
        rem6 = x

        x = self.deconv6a(x,rem5)
        rem5 = x
        x = self.deconv5a(x,rem4)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        rem4 = x
        x = self.conv5b(x, rem5)
        rem5 = x
        x = self.conv6b(x, rem6)

        x = self.deconv6b(x, rem5)
        x6 = self.outconv_6(x)


        x = self.deconv5b(x, rem4)
        x5 = self.outconv_5(x)

        x = self.deconv4b(x, rem3)
        x4 = self.outconv_4(x)

        x = self.deconv3b(x, rem2)
        x3 = self.outconv_3(x)

        x = self.deconv2b(x, rem1)
        x2 = self.outconv_2(x)


        return None, x2, x3, x4,x5,x6

