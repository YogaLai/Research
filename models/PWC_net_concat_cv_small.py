import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os.path

from utils.correltaiton import MatchingNetSmall, MatchingNetSmallAttn, compute_cost
from .Dual_attention import DualAttention
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

# from .networks.correlation_package.correlation import Correlation
# from networks.correlation_package.correlation import Correlation

__all__ = [
    'pwc_dc_net'
    ]
def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=True),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class DAP(nn.Module):
    def __init__(self, md=4):
        # Displacement-aware projection layer
        # implemented as a 1x1 2D conv
        super(DAP, self).__init__()
        dimC = (2*md+1)**2
        self.dap_layer = nn.Conv2d(dimC, dimC, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = x.squeeze(1)
        bs,du,dv,h,w = x.shape
        x = x.view(bs,du*dv,h,w)
        x = self.dap_layer(x)

        return x.view(bs,du*dv,h,w) 

class AttnDAP(nn.Module):
    def __init__(self, md=4):
        # Displacement-aware projection layer
        # implemented as a 1x1 2D conv
        super(AttnDAP, self).__init__()
        in_channels = (2*md+1)**2
        reduction_ratio = 2*md+1
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.ca_conv1 = nn.Conv2d(in_channels, in_channels//reduction_ratio, 1)
        self.ca_conv2 = nn.Conv2d(in_channels//reduction_ratio, in_channels, 1)
    def forward(self, x):
        x = x.squeeze(1)
        bs,du,dv,h,w = x.shape
        x = x.view(bs,du*dv,h,w)
        ca = self.global_avg_pooling(x)
        ca = self.ca_conv1(ca)
        ca = self.ca_conv2(ca)
        x = x * ca
        return x

class PWCDCNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections
    """
    def __init__(self, md=3, attn_match=False, attn_dap=False):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(PWCDCNet,self).__init__()

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
        self.conv6aa = myconv(128,196, kernel_size=3, stride=2)
        self.conv6a  = myconv(196,196, kernel_size=3, stride=1)
        self.conv6b  = myconv(196,196, kernel_size=3, stride=1)

        self.conv_out2 = myconv(32, 16)
        self.conv_out3 = myconv(64, 16)
        self.conv_out4 = myconv(96, 16)
        self.conv_out5 = myconv(128, 16)
        self.conv_out6 = myconv(196, 16)
        if attn_match:
            attention_list = nn.ModuleList([
                DualAttention(48),
                DualAttention(96),
                DualAttention(96),
                DualAttention(48),
                DualAttention(32),
            ]) 
            self.matchnet2 = MatchingNetSmallAttn(attention_list)
            self.matchnet3 = MatchingNetSmallAttn(attention_list)
            self.matchnet4 = MatchingNetSmallAttn(attention_list)
            self.matchnet5 = MatchingNetSmallAttn(attention_list)
            self.matchnet6 = MatchingNetSmallAttn(attention_list)
        else:
            self.matchnet2 = MatchingNetSmall()
            self.matchnet3 = MatchingNetSmall()
            self.matchnet4 = MatchingNetSmall()
            self.matchnet5 = MatchingNetSmall()
            self.matchnet6 = MatchingNetSmall()

        if attn_dap:
            self.dap6 = AttnDAP(md=md)
            self.dap5 = AttnDAP(md=md)
            self.dap4 = AttnDAP(md=md)
            self.dap3 = AttnDAP(md=md)
            self.dap2 = AttnDAP(md=md)

        else:
            self.dap6 = DAP(md=md)
            self.dap5 = DAP(md=md)
            self.dap4 = DAP(md=md)
            self.dap3 = DAP(md=md)
            self.dap2 = DAP(md=md)

        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        # self.cascade_attn0 = DualAttention(128)
        # self.cascade_attn1 = DualAttention(128)
        # self.cascade_attn2 = DualAttention(96)
        # self.cascade_attn3 = DualAttention(64)
        # self.cascade_attn4 = DualAttention(32)

        od = nd
        self.conv6_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv6_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv6_4 = myconv(64,32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(32)
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(32, 2, kernel_size=4, stride=2, padding=1)
        
        od = nd+16+4
        # od = nd+128+4
        self.conv5_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv5_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv5_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv5_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(32) 
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(32, 2, kernel_size=4, stride=2, padding=1) 
        
        # od = nd+96+4
        self.conv4_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv4_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv4_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv4_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(32) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(32, 2, kernel_size=4, stride=2, padding=1) 
        
        # od = nd+64+4
        self.conv3_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv3_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv3_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv3_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(32) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(32, 2, kernel_size=4, stride=2, padding=1) 
        
        # od = nd+32+4
        self.conv2_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv2_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv2_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv2_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(32) 
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        
        self.dc_conv1 = myconv(32, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = myconv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = myconv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = myconv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = myconv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = myconv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    
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
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size()))
        if x.is_cuda:
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        #mask[mask<0.9999] = 0.0
        #mask[mask>0] = 1.0
        mask = torch.floor(torch.clamp(mask, 0 ,1))
        
        return output*mask


    def forward(self,x):
        im1 = x[:,:3,:,:]
        im2 = x[:,3:,:,:]
        H, W = im1.shape[2:4]
        
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))


        c12 = self.conv_out2(c12)
        c22 = self.conv_out2(c22)
        c13 = self.conv_out3(c13)
        c23 = self.conv_out3(c23)
        c14 = self.conv_out4(c14)
        c24 = self.conv_out4(c24)
        c15 = self.conv_out5(c15)
        c25 = self.conv_out5(c25)
        c16 = self.conv_out6(c16)
        c26 = self.conv_out6(c26)
       

        corr6 = compute_cost(c16, c26, self.matchnet6)
        corr6 = self.dap6(corr6)
        # corr6 = self.leakyRELU(corr6)  

        x = self.conv6_0(corr6)
        # x = self.cascade_attn0(x)
        x = torch.cat((x, corr6), 1)
        # x = torch.cat((self.conv6_0(corr6), corr6),1)
        # x = self.conv6_0(corr6)
        x = self.conv6_1(x)
        # x = self.cascade_attn1(x)
        x = self.conv6_2(x)
        # x = self.cascade_attn2(x)
        x = self.conv6_3(x)
        # x = self.cascade_attn3(x)
        x = self.conv6_4(x)
        # x = self.cascade_attn4(x)

        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        
        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = compute_cost(c15, warp5, self.matchnet5)
        corr5 = self.dap5(corr5)
        # corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = self.conv5_0(x)
        # x = self.cascade_attn0(x)
        x = self.conv5_1(x)
        # x = self.cascade_attn1(x)
        x = self.conv5_2(x)
        # x = self.cascade_attn2(x)
        x = self.conv5_3(x)
        # x = self.cascade_attn3(x)
        x = self.conv5_4(x)
        # x = self.cascade_attn4(x)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = compute_cost(c14, warp4, self.matchnet4)
        corr4 = self.dap4(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = self.conv4_0(x)
        # x = self.cascade_attn0(x)
        x = self.conv4_1(x)
        # x = self.cascade_attn1(x)
        x = self.conv4_2(x)
        # x = self.cascade_attn2(x)
        x = self.conv4_3(x)
        # x = self.cascade_attn3(x)
        x = self.conv4_4(x)
        # x = self.cascade_attn4(x)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = compute_cost(c13, warp3, self.matchnet3)
        corr3 = self.dap3(corr3) 
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = self.conv3_0(x)
        # x = self.cascade_attn0(x)
        x = self.conv3_1(x)
        # x = self.cascade_attn1(x)
        x = self.conv3_2(x)
        # x = self.cascade_attn2(x)
        x = self.conv3_3(x)
        # x = self.cascade_attn3(x)
        x = self.conv3_4(x)
        # x = self.cascade_attn4(x)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)


        warp2 = self.warp(c22, up_flow3*5.0) 
        corr2 = compute_cost(c12, warp2, self.matchnet2)
        corr2 = self.dap2(corr2) 
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = self.conv2_0(x)
        # x = self.cascade_attn0(x)
        x = self.conv2_1(x)
        # x = self.cascade_attn1(x)
        x = self.conv2_2(x)
        # x = self.cascade_attn2(x)
        x = self.conv2_3(x)
        # x = self.cascade_attn3(x)
        x = self.conv2_4(x)
        # x = self.cascade_attn4(x)
        flow2 = self.predict_flow2(x)
 
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        
        # flow0_enlarge = self.upsample_conv(flow2 * 4.0)
        # flow1_enlarge = self.upsample_conv(flow3 * 4.0)
        # flow2_enlarge = self.upsample_conv(flow4 * 4.0)
        # flow3_enlarge = self.upsample_conv(flow5 * 4.0)
        flow0_enlarge = nn.UpsamplingBilinear2d(size = (H, W))(flow2 * 4.0)
        flow1_enlarge = nn.UpsamplingBilinear2d(size = (H // 2, W // 2))(flow3 * 4.0)
        flow2_enlarge = nn.UpsamplingBilinear2d(size = (H // 4, W // 4))(flow4 * 4.0)
        flow3_enlarge = nn.UpsamplingBilinear2d(size = (H // 8, W // 8))(flow5 * 4.0)

        return [flow0_enlarge, flow1_enlarge, flow2_enlarge, flow3_enlarge]
        
        """
        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2
        """





def pwc_dc_net(path=None):

    model = PWCDCNet()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model




def pwc_dc_net_old(path=None):

    model = PWCDCNet_old()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model
