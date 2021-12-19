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
from models.MonodepthModel import *
from models.PWC_net_my_correlation import *
from utils.scene_dataloader import *
from utils.utils import *
from models.RT_stereov4 import HRstereoNet

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
    parser.add_argument('--split',                     type=str,   help='validation set', default='kitti')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=256)
    parser.add_argument('--input_width',               type=int,   help='input width', default=512)
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', required=True)
    parser.add_argument('--exp_name',                  type=str,   help='experiment name',required=True)
    parser.add_argument('--cuda',                  help='use gpu', action='store_true')
    args = parser.parse_args()
    return args

args = get_args()

torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)

checkpoint = torch.load(args.checkpoint_path)
if args.model_name == 'monodepth':
    net = MonodepthNet()
elif args.model_name == 'pwc':
    net = pwc_dc_net()
    args.input_width = 832 
elif args.model_name == 'realtime_stereo':
    net = HRstereoNet(maxdisp=192)
    net = nn.DataParallel(net)
    args.input_width = 832
   
if args.cuda:
    net = net.cuda()
net.load_state_dict(checkpoint['state_dict'])
net.eval()

left_image_test, right_image_test = get_data(args.filenames_file, args.data_path)
TestImageLoader = torch.utils.data.DataLoader(
         myImageFolder(left_image_test, right_image_test, None, args),
         batch_size = 1, shuffle = False, drop_last=False)

if args.split == 'kitti':
    disparities = np.zeros((200, args.input_height, args.input_width), dtype=np.float32)
elif args.split == 'eigen':
    disparities = np.zeros((697, args.input_height, args.input_width), dtype=np.float32)

for batch_idx, (left, right) in enumerate(TestImageLoader, 0):
    print(batch_idx)
    if args.model_name == 'realtime_stereo':
        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**normal_mean_var)
        left, right = normalize(left[0]).unsqueeze(0), normalize(right[0]).unsqueeze(0)
        if args.cuda:
            left, right = left.cuda(), right.cuda()
        disp_est = net(left, right)[0] / 832
        disparities[batch_idx] = disp_est.data.cpu().numpy()
        continue
  
    left_batch = torch.cat((left, torch.from_numpy(np.flip(left.numpy(), 3).copy())), 0)
    right_batch = torch.cat((right, torch.from_numpy(np.flip(right.numpy(), 3).copy())), 0)
    
    left_pyramid = make_pyramid(left_batch, 4)
    
    model_input = Variable(torch.cat((left_batch, right_batch), 1))
    if args.cuda:
        model_input = model_input.cuda()

    if args.model_name == 'monodepth':
        disp_est_scale, disp_est= net(model_input)
    elif args.model_name == 'pwc':
        disp_est_scale = net(model_input)
        disp_est = [torch.cat((disp_est_scale[i][:,0,:,:].unsqueeze(1) / disp_est_scale[i].shape[3], disp_est_scale[i][:,1,:,:].unsqueeze(1) / disp_est_scale[i].shape[2]), 1) for i in range(4)]
    
    # pred_disp = (pred_disp*256).astype('uint16')
    # img = Image.fromarray(pred_disp)
    # img.save(f'visualization/evaluate/realtime_stereo/{batch_idx}.png')
    # plt.imsave(f'visualization/evaluate/{batch_idx}.png', pred_disp, cmap='jet')

    disparities[batch_idx] = -disp_est[0][0,0,:,:].data.cpu().numpy()
print('done')
np.save(f'./out_npy/disparities_{args.exp_name}.npy', disparities)
