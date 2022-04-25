import collections
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
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

def normalize_features(feature_list, normalize, center, moments_across_channels=True, moments_across_images=False):
    """Normalizes feature tensors (e.g., before computing the cost volume).
    Args:
        feature_list: list of torch tensors, (feat and warping feat), each with dimensions [b, c, h, w]
        normalize: bool flag, divide features by their standard deviation
        center: bool flag, subtract feature mean
        moments_across_channels: bool flag, compute mean and std across channels, 看到UFlow默认是True
        moments_across_images: bool flag, compute mean and std across images, 看到UFlow默认是True
    Returns:
        list, normalized feature_list
    """

    # Compute feature statistics.

    statistics = collections.defaultdict(list)
    axes = [1, 2, 3] if moments_across_channels else [2, 3]  # [b, c, h, w]
    for feature_image in feature_list:
        mean = torch.mean(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
        variance = torch.var(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
        statistics['mean'].append(mean)
        statistics['var'].append(variance)

    if moments_across_images:
        # statistics['mean'] = ([tf.reduce_mean(input_tensor=statistics['mean'])] *
        #                       len(feature_list))
        # statistics['var'] = [tf.reduce_mean(input_tensor=statistics['var'])
        #                      ] * len(feature_list)
        statistics['mean'] = ([torch.mean(torch.stack(statistics['mean'], dim=0), dim=(0,))] * len(feature_list))
        statistics['var'] = ([torch.var(torch.stack(statistics['var'], dim=0), dim=(0,))] * len(feature_list))

    statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]

    # Center and normalize features.

    if center:
        feature_list = [
            f - mean for f, mean in zip(feature_list, statistics['mean'])
        ]
    if normalize:
        feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]

    return feature_list

def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=True):
    _, _, h, w = target_as.size()
    res = nn.functional.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    if if_rate:
        _, _, h_, w_ = inputs.size()
        # inputs[:, 0, :, :] *= (w / w_)
        # inputs[:, 1, :, :] *= (h / h_)
        u_scale = (w / w_)
        v_scale = (h / h_)
        u, v = res.chunk(2, dim=1)
        u = u * u_scale
        v = v * v_scale
        res = torch.cat([u, v], dim=1)
    return res

class PWCDCNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections
    """
    def __init__(self, md=4):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(PWCDCNet,self).__init__()

        self.conv1a = myconv(3,   16, kernel_size=3, stride=2)
        self.conv1b = myconv(16,  16, kernel_size=3, stride=1)
        self.conv2a = myconv(16,  32, kernel_size=3, stride=2)
        self.conv2b = myconv(32,  32, kernel_size=3, stride=1)
        self.conv3a = myconv(32,  64, kernel_size=3, stride=2)
        self.conv3b = myconv(64,  64, kernel_size=3, stride=1)
        self.conv4a = myconv(64,  96, kernel_size=3, stride=2)
        self.conv4b = myconv(96,  96, kernel_size=3, stride=1)
        self.conv5a = myconv(96, 128, kernel_size=3, stride=2)
        self.conv5b = myconv(128,128, kernel_size=3, stride=1)
        self.conv6a = myconv(128,196, kernel_size=3, stride=2)
        self.conv6b = myconv(196,196, kernel_size=3, stride=1)

        self.conv_out2 = myconv(32, 32, kernel_size=1, padding=0)
        self.conv_out3 = myconv(64, 32, kernel_size=1, padding=0)
        self.conv_out4 = myconv(96, 32, kernel_size=1, padding=0)
        self.conv_out5 = myconv(128, 32, kernel_size=1, padding=0)
        self.conv_out6 = myconv(196, 32, kernel_size=1, padding=0)

        self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        nd = (2*md+1)**2

        # self.cascade_attn0 = DualAttention(128)
        # self.cascade_attn1 = DualAttention(128)
        # self.cascade_attn2 = DualAttention(96)
        # self.cascade_attn3 = DualAttention(64)
        # self.cascade_attn4 = DualAttention(32)

        od = nd+128
        self.conv6_0 = myconv(nd,      128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od,128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv6_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv6_4 = myconv(64,32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(32)
        # self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(32, 2, kernel_size=4, stride=2, padding=1)
        
        od = nd+32+4
        self.conv5_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv5_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv5_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv5_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(32) 
        # self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(32, 2, kernel_size=4, stride=2, padding=1) 
        
        # od = nd+96+4
        self.conv4_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv4_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv4_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv4_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(32) 
        # self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(32, 2, kernel_size=4, stride=2, padding=1) 
        
        # od = nd+64+4
        self.conv3_0 = myconv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = myconv(128,128, kernel_size=3, stride=1)
        self.conv3_2 = myconv(128,96,  kernel_size=3, stride=1)
        self.conv3_3 = myconv(96,64,  kernel_size=3, stride=1)
        self.conv3_4 = myconv(64,32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(32) 
        # self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
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
        
        c11 = self.conv1b(self.conv1a(im1))
        c21 = self.conv1b(self.conv1a(im2))
        c12 = self.conv2b(self.conv2a(c11))
        c22 = self.conv2b(self.conv2a(c21))
        c13 = self.conv3b(self.conv3a(c12))
        c23 = self.conv3b(self.conv3a(c22))
        c14 = self.conv4b(self.conv4a(c13))
        c24 = self.conv4b(self.conv4a(c23))
        c15 = self.conv5b(self.conv5a(c14))
        c25 = self.conv5b(self.conv5a(c24))
        c16 = self.conv6b(self.conv6a(c15))
        c26 = self.conv6b(self.conv6a(c25))

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

        c16, c26 = normalize_features([c16, c26], normalize=True, center=True)
        corr6 = self.corr(c16, c26) 
        corr6 = self.leakyRELU(corr6)
        x = self.conv6_0(corr6)
        # x = self.cascade_attn0(x)
        x = torch.cat((x, corr6), 1)
        x = self.conv6_1(x)
        # x = self.cascade_attn1(x)
        x = self.conv6_2(x)
        # x = self.cascade_attn2(x)
        x = self.conv6_3(x)
        # x = self.cascade_attn3(x)
        x = self.conv6_4(x)
        # x = self.cascade_attn4(x)
        flow6 = self.predict_flow6(x)
        up_flow6 = upsample2d_flow_as(flow6, c15)
        # up_flow6 = self.deconv6(flow6)*2
        up_feat6 = self.upfeat6(x)

        
        warp5 = self.warp(c25, up_flow6)
        c15, warp5 = normalize_features([c15, warp5], normalize=True, center=True)
        corr5 = self.corr(c15, warp5) 
        corr5 = self.leakyRELU(corr5)
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
        flow5 = flow5 + up_flow6 # residual
        up_flow5 = upsample2d_flow_as(flow5, c14)
        # up_flow5 = self.deconv5(flow5)*2
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24, up_flow5)
        c14, warp4 = normalize_features([c14, warp4], normalize=True, center=True)
        corr4 = self.corr(c14, warp4)  
        corr4 = self.leakyRELU(corr4)
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
        flow4 = flow4 + up_flow5
        up_flow4 = upsample2d_flow_as(flow4, c13)
        # up_flow4 = self.deconv4(flow4)*2
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23, up_flow4)
        c13, warp3 = normalize_features((c13, warp3), normalize=True, center=True)
        corr3 = self.corr(c13, warp3) 
        corr3 = self.leakyRELU(corr3)
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
        up_flow3 = upsample2d_flow_as(flow3, c12)
        # up_flow3 = self.deconv3(flow3)*2
        up_feat3 = self.upfeat3(x)


        warp2 = self.warp(c22, up_flow3)
        c12, warp2 = normalize_features((c12, warp2), normalize=True, center=True)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
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
        flow2 = self.predict_flow2(x) + up_flow3
 
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        
        flow2 = upsample2d_flow_as(flow2, im1)

        return flow2, [flow3, flow4, flow5]
        
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