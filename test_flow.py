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
from matplotlib import colors
import cv2
# from models.PWC_net_small_attn import PWCDCNet
from models.UFlow_wo_residual import PWCDCNet
# from models.dc_cost_attn_uflow_wo_residual import PWCDCNet
# from models.PWC_net_dc_cost_uflow import PWCDCNet
from utils.scene_dataloader import *
from collections import OrderedDict
from utils.utils import *
from utils.flow_visualize import save_flow

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name',                type=str,   help='model name', default='pwc')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=256)
    parser.add_argument('--input_width',               type=int,   help='input width', default=512)
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', required=True)
    parser.add_argument('--vis_rgb_flow',                  help='use gpu', action='store_true')
    args = parser.parse_args()
    return args

args = get_args()

net = PWCDCNet().cuda()
args.input_width = 896
args.input_height = 320
checkpoint = torch.load(args.checkpoint_path)
if any('module' in s for s in checkpoint['state_dict'].keys()):
    state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        state_dict[name] = v
else:
    state_dict = checkpoint['state_dict']
net.load_state_dict(state_dict)
net = net.eval()

noc_filename = args.filenames_file.replace('occ_', '')
former_test, latter_test, flow = get_flow_data(args.filenames_file, args.data_path)
former_test, latter_test, noc_flow = get_flow_data(noc_filename, args.data_path)
TestFlowLoader = torch.utils.data.DataLoader(
        myImageFolder(former_test, latter_test, flow, args, noc_flow=noc_flow),
        batch_size = 1, shuffle = False, num_workers = 1, drop_last = False)
# noc_flows, noc_masks = get_gt_flows(noc_flow_fn)

total_error = 0
total_epe_noc = 0
total_epe_occ = 0
fl_error = 0
num_test = 0
for batch_idx, (left, right, gt, noc_gt, mask, noc_mask, h, w) in enumerate(TestFlowLoader, 0):
    
    left_batch = torch.cat((left, torch.from_numpy(np.flip(left.numpy(), 3).copy())), 0)
    right_batch = torch.cat((right, torch.from_numpy(np.flip(right.numpy(), 3).copy())), 0)
    
    left = Variable(left_batch.cuda())
    right = Variable(right_batch.cuda())

    model_input = torch.cat((left, right), 1)
    if args.model_name == 'monodepth':
        disp_est_scale, disp_est = net(model_input)
    elif args.model_name == 'pwc':
        disp_est_scale = net(model_input)

    # mask = np.ceil(np.clip(np.abs(gt[0,0]), 0, 1))
    # noc_mask = np.ceil(np.clip(np.abs(noc_gt[0,0]), 0, 1))
    mask = mask.numpy()
    noc_mask = noc_mask.numpy()

    disp_ori_scale = nn.UpsamplingBilinear2d(size=(int(h), int(w)))(disp_est_scale[0][:1])
    disp_ori_scale[0,0] = disp_ori_scale[0,0] * int(w) / args.input_width
    disp_ori_scale[0,1] = disp_ori_scale[0,1] * int(h) / args.input_height

    if args.vis_rgb_flow:
        if batch_idx % 10 == 0:
            plt_flow = disp_ori_scale[0].permute(1,2,0).cpu().data.numpy()
            plt_flow[:,:,0] = plt_flow[:,:,0] * noc_mask
            plt_flow[:,:,1] = plt_flow[:,:,1] * noc_mask
            occ_mask = (mask.astype(np.uint8)-(noc_mask).astype(np.uint8))
            save_flow(f'visualization/rgb_flows/{batch_idx}.png', plt_flow)
            # plt.imsave(f'visualization/rgb_flows/noc_mask_{batch_idx}.png', noc_mask[0])
            # save_flow(f'visualization/rgb_flows/noc_gt_{batch_idx}.png', noc_gt[0].permute(1,2,0).cpu().data.numpy())
            # save_flow(f'visualization/rgb_flows/occ_gt_{batch_idx}.png', (gt[0]*occ_mask).permute(1,2,0).cpu().data.numpy())

    # epe_all, epe_noc, epe_occ, fl = evaluate_flow(disp_ori_scale[0].data.cpu().numpy(), gt[0].numpy(), mask, noc_mask)
    epe_all, epe_noc, epe_occ, fl = evaluate_flow(disp_ori_scale[0].data.cpu().numpy(), gt[0].numpy(), mask[0], noc_mask[0])
    total_error += epe_all
    total_epe_noc += epe_noc
    total_epe_occ += epe_occ
    fl_error += fl
    num_test += 1
    
total_error /= num_test
total_epe_noc /= num_test
total_epe_occ /= num_test
fl_error /= num_test
print("The average EPE is : ", total_error)
print("The EPE-noc is : ", total_epe_noc)
print("The EPE-occ is : ", total_epe_occ)
print("The average Fl is : ", fl_error)