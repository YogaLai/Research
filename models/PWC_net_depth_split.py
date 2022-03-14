import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os.path
from .Dual_attention import DualAttention
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

# from .networks.correlation_package.correlation import Correlation
from networks.correlation_package.correlation import Correlation

__all__ = [
    'pwc_dc_net'
    ]
def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=True),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,3,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class PWCDCNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections
    """
    def __init__(self, md=4):
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

        self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        

        nd = (2*md+1)**2 + (2*md+1)
        dd = np.cumsum([128,128,96,64,32])

        self.cascade_attn0 = DualAttention(128)
        self.cascade_attn1 = DualAttention(128)
        self.cascade_attn2 = DualAttention(96)
        self.cascade_attn3 = DualAttention(64)
        self.cascade_attn4 = DualAttention(32)

        od = nd
        self.conv6_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv6_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv6_4 = myconv(64,32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(32)
        self.deconv6 = deconv(3, 3, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(32, 2, kernel_size=4, stride=2, padding=1)
        
        od = nd+128+5
        self.conv5_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv5_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv5_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv5_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(32) 
        self.deconv5 = deconv(3, 3, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(32, 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+5
        self.conv4_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv4_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv4_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv4_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(32) 
        self.deconv4 = deconv(3, 3, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(32, 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+5
        self.conv3_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv3_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv3_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv3_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(32) 
        self.deconv3 = deconv(3, 3, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(32, 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+5
        self.conv2_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv2_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv2_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv2_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(32) 
        
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

    
    def warp(self, x, disp, depth=False):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, _, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        if depth:
            zero = torch.zeros_like(disp)
            disp = torch.cat((disp, zero), 1)
        vgrid = grid + disp

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.ones(x.size())
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

    def disp_corr(self, refimg_fea, targetimg_fea):
        maxdisp=4
        b,c,h,w = refimg_fea.shape
        targetimg_fea = F.unfold(targetimg_fea, (1,2*maxdisp+1), padding=(0, maxdisp)) # b, c*(2*d+1), (h*w)
        targetimg_fea = targetimg_fea.view(b, c, 2*maxdisp+1, h, w)
        # targetimg_fea = F.unfold(targetimg_fea, (1,2*maxdisp+1), padding=maxdisp).view(b,c,2*maxdisp+1,h,w)
        cost = refimg_fea.view(b,c,h,w)[:,:,np.newaxis] * targetimg_fea # b,c,1,h,w * b,c,(2*d+1),h,w
        # cost = refimg_fea.view(b,c,h,w)[:,:,np.newaxis, np.newaxis]*targetimg_fea.view(b,c,2*maxdisp+1, 2*maxdisp+1**2,h,w)
        cost = cost.sum(1)

        # b, pw, h, w = cost.size()
        cost = cost/refimg_fea.size(1)
        # cost = cost.view(b, pw, h, w)/refimg_fea.size(1)
        return cost 

    def forward(self,x, slices):
        flow_slice = slices[0]+slices[-1]
        disp_slice = slices[1]+slices[2]
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


        flow_corr6 = self.corr(c16[flow_slice,:,:,:], c26[flow_slice,:,:,:]) 
        flow_corr6 = self.leakyRELU(flow_corr6)
        disp_corr6 = self.disp_corr(c16[disp_slice,:,:,:], c26[disp_slice,:,:,:])  
        disp_corr6 = self.leakyRELU(disp_corr6)
        corr6 = torch.cat((flow_corr6, disp_corr6), 1)

        x = self.conv6_0(corr6)
        x = self.cascade_attn0(x)
        x = torch.cat((x, corr6), 1)
        # x = torch.cat((self.conv6_0(corr6), corr6),1)
        # x = self.conv6_0(corr6)
        x = self.conv6_1(x)
        x = self.cascade_attn1(x)
        x = self.conv6_2(x)
        x = self.cascade_attn2(x)
        x = self.conv6_3(x)
        x = self.cascade_attn3(x)
        x = self.conv6_4(x)
        x = self.cascade_attn4(x)

        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25[flow_slice,:,:,:], up_flow6[:,:2,:,:]*0.625)
        flow_corr5 = self.corr(c15[flow_slice,:,:,:], warp5) 
        flow_corr5 = self.leakyRELU(flow_corr5)
        warp5 = self.warp(c25[disp_slice,:,:,:], up_flow6[:,2:,:,:]*0.625, depth=True)
        disp_corr5 = self.disp_corr(c15[disp_slice,:,:,:], warp5) 
        disp_corr5 = self.leakyRELU(disp_corr5)
        x = torch.cat((flow_corr5, disp_corr5, c15[flow_slice,:,:,:], up_flow6, up_feat6), 1)
        # x = torch.cat((flow_corr5, disp_corr5, c15, up_flow6, up_feat6), 1)
        x = self.conv5_0(x)
        x = self.cascade_attn0(x)
        x = self.conv5_1(x)
        x = self.cascade_attn1(x)
        x = self.conv5_2(x)
        x = self.cascade_attn2(x)
        x = self.conv5_3(x)
        x = self.cascade_attn3(x)
        x = self.conv5_4(x)
        x = self.cascade_attn4(x)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24[flow_slice,:,:,:], up_flow5[:,:2,:,:]*1.25)
        flow_corr4 = self.corr(c14[flow_slice,:,:,:], warp4)  
        flow_corr4 = self.leakyRELU(flow_corr4)
        warp4 = self.warp(c24[disp_slice,:,:,:], up_flow5[:,2:,:,:]*1.25, depth=True)
        disp_corr4 = self.disp_corr(c14[disp_slice,:,:,:], warp4) 
        disp_corr4 = self.leakyRELU(disp_corr4)
        x = torch.cat((flow_corr4, disp_corr4, c14[flow_slice,:,:,:], up_flow5, up_feat5), 1)
        x = self.conv4_0(x)
        x = self.cascade_attn0(x)
        x = self.conv4_1(x)
        x = self.cascade_attn1(x)
        x = self.conv4_2(x)
        x = self.cascade_attn2(x)
        x = self.conv4_3(x)
        x = self.cascade_attn3(x)
        x = self.conv4_4(x)
        x = self.cascade_attn4(x)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23[flow_slice,:,:,:], up_flow4[:,:2,:,:]*2.5)
        flow_corr3 = self.corr(c13[flow_slice,:,:,:], warp3)  
        flow_corr3 = self.leakyRELU(flow_corr3)
        warp3 = self.warp(c23[disp_slice,:,:,:], up_flow4[:,2:,:,:]*2.5, depth=True)
        disp_corr3 = self.disp_corr(c13[disp_slice,:,:,:], warp3) 
        disp_corr3 = self.leakyRELU(disp_corr3)
        

        x = torch.cat((flow_corr3, disp_corr3, c13[flow_slice,:,:,:], up_flow4, up_feat4), 1)
        x = self.conv3_0(x)
        x = self.cascade_attn0(x)
        x = self.conv3_1(x)
        x = self.cascade_attn1(x)
        x = self.conv3_2(x)
        x = self.cascade_attn2(x)
        x = self.conv3_3(x)
        x = self.cascade_attn3(x)
        x = self.conv3_4(x)
        x = self.cascade_attn4(x)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22[flow_slice,:,:,:], up_flow3[:,:2,:,:]*5.0)
        flow_corr2 = self.corr(c12[flow_slice,:,:,:], warp2)  
        flow_corr2 = self.leakyRELU(flow_corr2)
        warp2 = self.warp(c22[disp_slice,:,:,:], up_flow3[:,2:,:,:]*5.0, depth=True)
        disp_corr2 = self.disp_corr(c12[disp_slice,:,:,:], warp2) 
        disp_corr2 = self.leakyRELU(disp_corr2)
        x = torch.cat((flow_corr2, disp_corr2, c12[flow_slice,:,:,:], up_flow3, up_feat3), 1)
        x = self.conv2_0(x)
        x = self.cascade_attn0(x)
        x = self.conv2_1(x)
        x = self.cascade_attn1(x)
        x = self.conv2_2(x)
        x = self.cascade_attn2(x)
        x = self.conv2_3(x)
        x = self.cascade_attn3(x)
        x = self.conv2_4(x)
        x = self.cascade_attn4(x)
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
