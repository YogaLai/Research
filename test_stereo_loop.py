import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt
from models.PWC_net_concat_cv import *
# from models.PWC_net_attn_bigger import *
# from models.PWC_net_small_attn import *
from utils.scene_dataloader import *
from utils.utils import *
# from models.RT_stereov4 import HRstereoNet
from collections import OrderedDict
from models.DICL import dicl_wrapper
from config import cfg_from_file

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name',                type=str,   help='model name', default='pwc')
    parser.add_argument('--split',                     type=str,   help='validation set', default='kitti')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=256)
    parser.add_argument('--input_width',               type=int,   help='input width', default=512)
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', required=True)
    parser.add_argument('--cuda',                  help='use gpu', action='store_true')
    parser.add_argument('--epoch_thres',               type=int,   help='use gpu', default=0)
    parser.add_argument('--exp_name',                  type=str,   help='experiment name',required=True)
    args = parser.parse_args()
    return args

args = get_args()

torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)

if args.model_name == 'monodepth':
    net = MonodepthNet()
elif args.model_name == 'pwc':
    net = pwc_dc_net()
    args.input_width = 768
elif args.model_name == 'dicl':
    cfg_from_file('cfgs/dicl5_kitti.yml')
    net = dicl_wrapper()
    args.input_width = 768
   
if args.cuda:
    net = net.cuda()

left_image_test, right_image_test = get_data(args.filenames_file, args.data_path)
TestImageLoader = torch.utils.data.DataLoader(
         myImageFolder(left_image_test, right_image_test, None, args),
         batch_size = 1, shuffle = False, drop_last=False)

ckt_dict = {}
for ckt_name in os.listdir(args.checkpoint_path):
    if 'model_epoch' in ckt_name:
        epoch = int(ckt_name[11:])
        if epoch < args.epoch_thres:
            continue
        ckt_dict[epoch] = os.path.join(args.checkpoint_path, ckt_name)

ckt_dict = sorted(ckt_dict.items(), key=lambda x: int(x[0])) # sort by key
for epoch, ckt in ckt_dict:
    print('epoch: ', epoch)
    checkpoint = torch.load(ckt)
    # if any("abc" in s for s in some_list):
    if any('module' in s for s in checkpoint['state_dict'].keys()):
        state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove `module.`
            state_dict[name] = v
    else:
        state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
    net.eval()

    if args.split == 'kitti':
        disparities = np.zeros((200, args.input_height, args.input_width), dtype=np.float32)
    elif args.split == 'eigen':
        disparities = np.zeros((697, args.input_height, args.input_width), dtype=np.float32)

    for batch_idx, (left, right) in enumerate(TestImageLoader, 0):
        left_batch = torch.cat((left, torch.from_numpy(np.flip(left.numpy(), 3).copy())), 0)
        right_batch = torch.cat((right, torch.from_numpy(np.flip(right.numpy(), 3).copy())), 0)
        
        model_input = Variable(torch.cat((left_batch, right_batch), 1))
        if args.cuda:
            model_input = model_input.cuda()

        disp_est_scale = net(model_input)
        disp_est = [torch.cat((disp_est_scale[i][:,0,:,:].unsqueeze(1) / disp_est_scale[i].shape[3], disp_est_scale[i][:,1,:,:].unsqueeze(1) / disp_est_scale[i].shape[2]), 1) for i in range(4)]

        disparities[batch_idx] = -disp_est[0][0,0,:,:].data.cpu().numpy()
    print('done')
    if not os.path.isdir('out_npy/' + args.exp_name):
        os.makedirs('out_npy/' + args.exp_name)
    np.save(f'./out_npy/{args.exp_name}/disparities_{args.exp_name}_{epoch}.npy', disparities)
