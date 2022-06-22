import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from networks.resample2d_package.resample2d import Resample2d
import time

def gradient_x(img):
    gx = torch.add(img[:,:,:-1,:], -1, img[:,:,1:,:])
    return gx

def gradient_y(img):
    gy = torch.add(img[:,:,:,:-1], -1, img[:,:,:,1:])
    return gy

def get_disparity_smoothness(disp, pyramid):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]
    
    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]
    
    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
    
    return smoothness_x + smoothness_y

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)
    
    #(input, kernel, stride, padding)
    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y
    
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    SSIM = SSIM_n / SSIM_d
    
    return torch.clamp((1 - SSIM) / 2, 0, 1)

def cal_grad2_error(flo, image, beta, edge_weight=10.0):
    """
    Calculate the image-edge-aware second-order smoothness loss for flo 
    """

    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy
    
    
    img_grad_x, img_grad_y = gradient(image)
    weights_x = torch.exp(-edge_weight * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
    weights_y = torch.exp(-edge_weight * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (torch.mean(beta*weights_x[:,:, :, 1:]*torch.abs(dx2)) + torch.mean(beta*weights_y[:, :, 1:, :]*torch.abs(dy2))) / 2.0

def warp_2(est, img, occ_mask, args):
    l1_warp2 = torch.abs(est - img) * occ_mask
    l1_reconstruction_loss_warp2 = torch.mean(l1_warp2) / torch.mean(occ_mask)
    ssim_warp2 = SSIM(est * occ_mask, img * occ_mask)
    ssim_loss_warp2 = torch.mean(ssim_warp2) / torch.mean(occ_mask)
    image_loss_warp2  = args.alpha_image_loss * ssim_loss_warp2 + (1 - args.alpha_image_loss) * l1_reconstruction_loss_warp2
    return image_loss_warp2

def create_mask(tensor, paddings):
    shape = tensor.shape
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner = Variable(torch.ones((inner_height, inner_width)).cuda())
    
    mask2d = nn.ZeroPad2d((paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]))(inner)
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    mask4d = mask3d.unsqueeze(1)
    return mask4d.detach()

def create_border_mask(tensor, border_ratio = 0.1):
    num_batch, _, height, width = tensor.shape
    sz = np.ceil(height * border_ratio).astype(np.int).item(0)
    border_mask = create_mask(tensor, [[sz, sz], [sz, sz]])
    return border_mask.detach()

def length_sq(x):
    return torch.sum(x**2, 1, keepdim=True)

def create_outgoing_mask(flow):
    num_batch, channel, height, width = flow.shape
    
    grid_x = torch.arange(width).view(1, 1, width)
    grid_x = grid_x.repeat(num_batch, height, 1)
    grid_y = torch.arange(height).view(1, height, 1)
    grid_y = grid_y.repeat(num_batch, 1, width)
    
    flow_u, flow_v = torch.unbind(flow, 1)
    pos_x = grid_x.type(torch.FloatTensor) + flow_u.data.cpu()
    pos_y = grid_y.type(torch.FloatTensor) + flow_v.data.cpu()
    inside_x = (pos_x <= (width - 1)) & (pos_x >= 0.0)
    inside_y = (pos_y <= (height - 1)) & (pos_y >= 0.0)
    inside = inside_x & inside_y
    return inside.type(torch.FloatTensor).unsqueeze(1)

def get_mask(forward, backward, border_mask):
    flow_fw = forward
    flow_bw = backward
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    
    flow_bw_warped = Resample2d()(flow_bw, flow_fw)
    flow_fw_warped = Resample2d()(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh =  0.01 * mag_sq + 0.5
    # fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh).type(torch.cuda.FloatTensor)
    # fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh).type(torch.cuda.FloatTensor)
    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh).to(forward.device).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh).to(backward.device).float()
    
    if border_mask == None:
        mask_fw = create_outgoing_mask(flow_fw).to(forward.device)
        mask_bw = create_outgoing_mask(flow_bw).to(forward.device)
    else:
        mask_fw = border_mask
        mask_bw = border_mask
    fw = mask_fw * (1 - fb_occ_fw)
    bw = mask_bw * (1 - fb_occ_bw)

    return fw, bw, fb_occ_fw, fb_occ_bw

def get_mix_mask(forward, backward, border_mask, flow_slices):
    flow_fw = forward
    flow_bw = backward
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    
    flow_bw_warped = Resample2d()(flow_bw, flow_fw)
    flow_fw_warped = Resample2d()(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh =  0.01 * mag_sq + 0.5
    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh).to(forward.device).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh).to(backward.device).float()
    
    mask_fw = create_outgoing_mask(flow_fw).to(forward.device)
    mask_bw = create_outgoing_mask(flow_bw).to(forward.device)
    mask_fw[flow_slices] = border_mask[flow_slices]
    mask_bw[flow_slices] = border_mask[flow_slices]

    fw = mask_fw * (1 - fb_occ_fw)
    bw = mask_bw * (1 - fb_occ_bw)

    return fw, bw, flow_diff_fw, flow_diff_bw

def get_dilated_warp_mask(forward, backward):
    def get_obj_occ_check(valid_mask, out_occ):
            outgoing_mask = torch.zeros_like(valid_mask)
            if valid_mask.is_cuda:
                outgoing_mask = outgoing_mask.cuda()
            outgoing_mask[valid_mask == 1] = 1
            outgoing_mask[out_occ == 0] = 1
            return outgoing_mask

    flow_fw = forward
    flow_bw = backward
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    
    flow_bw_warped = Resample2d()(flow_bw, flow_fw)
    flow_fw_warped = Resample2d()(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh =  0.01 * mag_sq + 0.5
    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh).to(forward.device).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh).to(backward.device).float()
    
    outgoing_mask_fw = create_outgoing_mask(flow_fw).to(forward.device)
    outgoing_mask_bw = create_outgoing_mask(flow_bw).to(backward.device)

    obj_fw_mask = get_obj_occ_check(1 - fb_occ_fw, outgoing_mask_fw)
    obj_bw_mask = get_obj_occ_check(1 - fb_occ_bw, outgoing_mask_bw)

    return obj_fw_mask, obj_bw_mask

class boundary_dilated_warp():

        @classmethod
        def get_grid(cls, batch_size, H, W, start):
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
            ones = torch.ones_like(xx)
            grid = torch.cat((xx, yy, ones), 1).float()
            if torch.cuda.is_available():
                grid = grid.cuda()
            # print("grid", grid.shape)
            # print("start", start.repeat(batch_size//start.size(0), 1, 1, 1).shape)
            grid[:, :2, :, :] = grid[:, :2, :, :] + start.repeat(batch_size // start.size(0), 1, 1, 1)  # 加上patch在原图内的偏移量

            return grid

        @classmethod
        def transformer(cls, I, vgrid, train=True):
            # I: Img, shape: batch_size, 1, full_h, full_w
            # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
            # outsize: (patch_h, patch_w)

            def _repeat(x, n_repeats):

                rep = torch.ones([n_repeats, ]).unsqueeze(0)
                rep = rep.int()
                x = x.int()

                x = torch.matmul(x.reshape([-1, 1]), rep)
                return x.reshape([-1])

            def _interpolate(im, x, y, out_size, scale_h):
                # x: x_grid_flat
                # y: y_grid_flat
                # out_size: same as im.size
                # scale_h: True if normalized
                # constants
                num_batch, num_channels, height, width = im.size()

                out_height, out_width = out_size[0], out_size[1]
                # zero = torch.zeros_like([],dtype='int32')
                zero = 0
                max_y = height - 1
                max_x = width - 1
                if scale_h:
                    # scale indices from [-1, 1] to [0, width or height]
                    # print('--Inter- scale_h:', scale_h)
                    x = (x + 1.0) * (height) / 2.0
                    y = (y + 1.0) * (width) / 2.0

                # do sampling
                x0 = torch.floor(x).int()
                x1 = x0 + 1
                y0 = torch.floor(y).int()
                y1 = y0 + 1

                x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
                x1 = torch.clamp(x1, zero, max_x)
                y0 = torch.clamp(y0, zero, max_y)
                y1 = torch.clamp(y1, zero, max_y)

                dim1 = torch.from_numpy(np.array(width * height))
                dim2 = torch.from_numpy(np.array(width))

                base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)  # 其实就是单纯标出batch中每个图的下标位置
                # base = torch.arange(0,num_batch) * dim1
                # base = base.reshape(-1, 1).repeat(1, out_height * out_width).reshape(-1).int()
                # 区别？expand不对数据进行拷贝 .reshape(-1,1).expand(-1,out_height * out_width).reshape(-1)
                if torch.cuda.is_available():
                    dim2 = dim2.cuda()
                    dim1 = dim1.cuda()
                    y0 = y0.cuda()
                    y1 = y1.cuda()
                    x0 = x0.cuda()
                    x1 = x1.cuda()
                    base = base.cuda()
                base_y0 = base + y0 * dim2
                base_y1 = base + y1 * dim2
                idx_a = base_y0 + x0
                idx_b = base_y1 + x0
                idx_c = base_y0 + x1
                idx_d = base_y1 + x1

                # use indices to lookup pixels in the flat image and restore
                # channels dim
                im = im.permute(0, 2, 3, 1)
                im_flat = im.reshape([-1, num_channels]).float()

                idx_a = idx_a.unsqueeze(-1).long()
                idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
                Ia = torch.gather(im_flat, 0, idx_a)

                idx_b = idx_b.unsqueeze(-1).long()
                idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
                Ib = torch.gather(im_flat, 0, idx_b)

                idx_c = idx_c.unsqueeze(-1).long()
                idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
                Ic = torch.gather(im_flat, 0, idx_c)

                idx_d = idx_d.unsqueeze(-1).long()
                idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
                Id = torch.gather(im_flat, 0, idx_d)

                # and finally calculate interpolated values
                x0_f = x0.float()
                x1_f = x1.float()
                y0_f = y0.float()
                y1_f = y1.float()

                wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
                wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
                wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
                wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
                output = wa * Ia + wb * Ib + wc * Ic + wd * Id

                return output

            def _transform(I, vgrid, scale_h):

                C_img = I.shape[1]
                B, C, H, W = vgrid.size()

                x_s_flat = vgrid[:, 0, ...].reshape([-1])
                y_s_flat = vgrid[:, 1, ...].reshape([-1])
                out_size = vgrid.shape[2:]
                input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size, scale_h)

                output = input_transformed.reshape([B, H, W, C_img])
                return output

            # scale_h = True
            output = _transform(I, vgrid, scale_h=False)
            if train:
                output = output.permute(0, 3, 1, 2)
            return output

        @classmethod
        def warp_im(cls, I_nchw, flow_nchw, start_n211):
            batch_size, _, img_h, img_w = I_nchw.size()
            _, _, patch_size_h, patch_size_w = flow_nchw.size()
            patch_indices = cls.get_grid(batch_size, patch_size_h, patch_size_w, start_n211)
            vgrid = patch_indices[:, :2, ...]
            # grid_warp = vgrid - flow_nchw
            grid_warp = vgrid + flow_nchw
            pred_I2 = cls.transformer(I_nchw, grid_warp)
            return pred_I2

def get_mask_wo_resample(forward, backward, border_mask):
    flow_fw = forward
    flow_bw = backward
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    
    flow_bw_warped = flow_warp(flow_bw, flow_fw)
    flow_fw_warped = flow_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh =  0.01 * mag_sq + 0.5
    # fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh).type(torch.cuda.FloatTensor)
    # fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh).type(torch.cuda.FloatTensor)
    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh).to(forward.device).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh).to(backward.device).float()
    
    if border_mask is None:
        mask_fw = create_outgoing_mask(flow_fw)
        mask_bw = create_outgoing_mask(flow_bw)
    else:
        mask_fw = border_mask
        mask_bw = border_mask
    # print(mask_fw.device)
    # print(fb_occ_bw.device)
    fw = mask_fw * (1 - fb_occ_fw)
    bw = mask_bw * (1 - fb_occ_bw)

    return fw, bw, flow_diff_fw, flow_diff_bw

def make_pyramid(image, num_scales):
    scale_image = [Variable(image.cuda())]
    height, width = image.shape[2:]

    for i in range(num_scales - 1):
        new = []
        for j in range(image.shape[0]):
            ratio = 2 ** (i+1)
            nh = height // ratio
            nw = width // ratio
            tmp = transforms.ToPILImage()(image[j]).convert('RGB')
            tmp = transforms.Resize([nh, nw])(tmp)
            tmp = transforms.ToTensor()(tmp)
            new.append(tmp.unsqueeze(0))
        this = torch.cat(new, 0)
        scale_image.append(Variable(this.cuda()))
        
    return scale_image

def evaluate_flow(flow, flow_gt, valid_mask=None, noc_mask=None):

    if valid_mask is None:
        tmp = np.multiply(flow_gt[0,:,:], flow_gt[1,:,:])
        valid_mask = np.ceil(np.clip(np.abs(tmp), 0, 1))
        
    N = np.sum(valid_mask)

    u = flow[0, :, :]
    v = flow[1, :, :]

    u_gt = flow_gt[0, :, :]
    v_gt = flow_gt[1, :, :]

    ### compute_EPE
    du = u - u_gt
    dv = v - v_gt

    du2 = np.multiply(du, du)
    dv2 = np.multiply(dv, dv)

    EPE = np.sqrt(du2 + dv2)
    EPE_all = np.sum(np.multiply(EPE, valid_mask)) / N

    EPE_noc = np.multiply(EPE, noc_mask)
    EPE_noc = np.sum(EPE_noc) / np.sum(noc_mask)
    occ_idx = (valid_mask.astype(np.uint8)-(noc_mask).astype(np.uint8))
    EPE_occ = np.multiply(EPE, occ_idx) 
    EPE_occ = np.sum(EPE_occ) / max(np.sum(occ_idx), 1.0)

    # epe_all = np.sum(epe_map*mask)/np.sum(mask)
    # epe_noc = np.sum(epe_map*noc_mask)/np.sum(noc_mask)
    # idx = (mask.astype(np.uint8)-(noc_mask).astype(np.uint8))
    # epe_occ = np.sum(epe_map*idx) / max(np.sum(idx), 1.0)

    
    ### compute FL
    EPE = np.multiply(EPE, valid_mask)
    bad_pixels = np.logical_and(
        EPE > 3,
        (EPE / np.sqrt(np.sum(np.square(flow_gt), axis=0)) + 1e-5) > 0.05)
    FL_avg = bad_pixels.sum() / valid_mask.sum()

    return EPE_all, EPE_noc, EPE_occ, FL_avg

def flow_warp(tenInput, tenFlow):
    tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
    tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
    backwarp_tenGrid = torch.cat([ tenHor, tenVer ], 1).cuda()
    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)

def repeat(x, n_repeats):
    x = x.type(torch.int32)
    rep = torch.ones((1, n_repeats)).type(torch.int32)
    x = x.reshape(-1,1)@rep
    return x.reshape(-1)

def transformer_fwd(im0, flow):
    zero = torch.zeros([]).to(flow.device)
    B,C,H,W = im0.shape
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(flow.device)
    grid = grid + flow
   
    x0 = torch.floor(grid[:,0:1,:,:])
    x1 = x0 + 1
    y0 = torch.floor(grid[:,1:,:,:])
    y1 = y0 + 1

    base = repeat(torch.arange(B)*W*H, W*H).to(flow.device)
    base_y0 = base + torch.clamp(y0.reshape(-1), 0, H-1) * W
    base_y1 = base + torch.clamp(y1.reshape(-1), 0, H-1) * W
    idx_a = (base_y0 + torch.clamp(x0.reshape(-1), 0, W-1)).type(torch.int64)
    idx_b = (base_y1 + torch.clamp(x0.reshape(-1), 0, W-1)).type(torch.int64)
    idx_c = (base_y0 + torch.clamp(x1.reshape(-1), 0, W-1)).type(torch.int64)
    idx_d = (base_y1 + torch.clamp(x1.reshape(-1), 0, W-1)).type(torch.int64)

    wa = ((x1-grid[:,0:1,:,:]) * (y1-grid[:,1:,:,:]))
    wb = ((x1-grid[:,0:1,:,:]) * (grid[:,1:,:,:]-y0))
    wc = ((grid[:,0:1,:,:]-x0) * (y1-grid[:,1:,:,:]))
    wd = ((grid[:,0:1,:,:]-x0) * (grid[:,1:,:,:]-y0))

    """ valid range not done"""
    cond_x0 = torch.logical_and(x0>=0, x0<W)
    cond_x1 = torch.logical_and(x1>=0, x1<W)
    cond_y0 = torch.logical_and(y0>=0, y0<H)
    cond_y1 = torch.logical_and(y1>=0, y1<H)
    wa = torch.where(torch.logical_and(cond_x0, cond_y0), wa, zero).reshape(-1,1)
    wb = torch.where(torch.logical_and(cond_x0, cond_y1), wb, zero).reshape(-1,1)
    wc = torch.where(torch.logical_and(cond_x1, cond_y0), wc, zero).reshape(-1,1)
    wd = torch.where(torch.logical_and(cond_x1, cond_y1), wd, zero).reshape(-1,1)

    im0 = im0.reshape(-1, C)
    im1 = torch.zeros_like(im0)
    im1 = im1.scatter_add(0, idx_a.reshape(-1,1), wa*im0)
    im1 = im1.scatter_add(0, idx_b.reshape(-1,1), wb*im0)
    im1 = im1.scatter_add(0, idx_c.reshape(-1,1), wc*im0)
    im1 = im1.scatter_add(0, idx_d.reshape(-1,1), wd*im0)

    im1 = im1.reshape(B,C,H,W)
    return im1

def get_soft_mask(backward_flow, fw, slices, eps=1e-3, neg_disp=False):
    one = torch.ones([backward_flow.size(0), 1, backward_flow.size(2), backward_flow.size(3)]).to(backward_flow.device).contiguous()
    if neg_disp:
        disp = backward_flow.clone().detach()
        disp[slices[1]+slices[2]] = -disp[slices[1]+slices[2]]
    else:
        disp = backward_flow

    disp = disp.permute(0,2,3,1).contiguous()
    # disp = torch.cat([-backward_flow[[0,1]], backward_flow[[2,3,4,5]], -backward_flow[[6,7]]])
    # mask = forward_warp(one, backward_flow)
    mask = fw(one, disp)
    mask += eps
    mask = torch.clamp(mask, 0, 1)
    return mask

def disparity_to_flow(disp):
    zero = torch.zeros_like(disp)
    flow = torch.cat((disp, zero), 1)
    return flow

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    plt.figure(figsize=(40,15))
    for n, p in named_parameters:
        if 'deconv2' in n:
            continue
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().data.numpy())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('visualization/grad_flow.png')

def photo_loss_function(diff, mask, q, charbonnier_or_abs_robust, averge=True):
    if charbonnier_or_abs_robust:
        p = ((diff) ** 2 + 1e-6).pow(q)
        p = p * mask
        if averge:
            p = p.mean()
            ap = mask.mean()
        else:
            p = p.sum()
            ap = mask.sum()
        loss_mean = p / (ap * 2 + 1e-6)
        
    else:
        diff = (torch.abs(diff) + 0.01).pow(q)
        diff = diff * mask
        diff_sum = torch.sum(diff)
        loss_mean = diff_sum / (torch.sum(mask) * 2 + 1e-6)
       
    return loss_mean

def census_loss(img1, img1_warp, mask, q=0.45, charbonnier_or_abs_robust=True, averge=True, max_disp=3):
    patch_size = 2 * max_disp + 1

    def _ternary_transform_torch(image):
        R, G, B = torch.split(image, 1, 1)
        intensities_torch = (0.2989 * R + 0.5870 * G + 0.1140 * B) * 255  # * 255  # convert to gray
        # intensities = tf.image.rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))  # h,w,1,out_c
        w_ = np.transpose(w, (3, 2, 0, 1))  # 1,out_c,h,w
        weight = torch.from_numpy(w_).float()
        if image.is_cuda:
            weight = weight.cuda()
        patches_torch = torch.conv2d(input=intensities_torch, weight=weight, bias=None, stride=[1, 1], padding=[max_disp, max_disp])
        transf_torch = patches_torch - intensities_torch
        transf_norm_torch = transf_torch / torch.sqrt(0.81 + transf_torch ** 2)
        return transf_norm_torch

    def _hamming_distance_torch(t1, t2):
        dist = (t1 - t2) ** 2
        dist = torch.sum(dist / (0.1 + dist), 1, keepdim=True)
        return dist

    def create_mask_torch(tensor, paddings):
        shape = tensor.shape  # N,c, H,W
        inner_width = shape[2] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[3] - (paddings[1][0] + paddings[1][1])
        inner_torch = torch.ones([shape[0], shape[1], inner_width, inner_height]).float()
        if tensor.is_cuda:
            inner_torch = inner_torch.cuda()
        mask2d = F.pad(inner_torch, [paddings[0][0], paddings[0][1], paddings[1][0], paddings[1][1]])
        return mask2d

    img1 = _ternary_transform_torch(img1)
    img1_warp = _ternary_transform_torch(img1_warp)
    dist = _hamming_distance_torch(img1, img1_warp)
    transform_mask = create_mask_torch(mask, [[max_disp, max_disp],
                                                [max_disp, max_disp]])
    census_loss = photo_loss_function(diff=dist, mask=mask * transform_mask, q=q,
                                            charbonnier_or_abs_robust=charbonnier_or_abs_robust, averge=averge)
    return census_loss, dist

def upsample_flow(inputs, target_size=None, target_flow=None, mode="bilinear"):
    if target_size is not None:
        h, w = target_size
    elif target_flow is not None:
        _, _, h, w = target_flow.size()
    else:
        raise ValueError('wrong input')
    _, _, h_, w_ = inputs.size()
    res = torch.nn.functional.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    res[:, 0, :, :] *= (w / w_)
    res[:, 1, :, :] *= (h / h_)
    return res

def photo_loss_abs_robust(diff, occ_mask, photo_loss_delta=0.4):
    loss_diff = (torch.abs(diff) + 0.01).pow(photo_loss_delta)
    photo_loss = torch.sum(loss_diff * occ_mask) / (torch.sum(occ_mask) + 1e-6)
    return photo_loss

def get_selfsup_transformations(args, crop_size=32):
    return transforms.Compose([
        transforms.CenterCrop((args.input_height - 2 * crop_size, args.input_width - 2 * crop_size)),
        # transforms.Resize([args.input_height, args.input_width]),
    ])