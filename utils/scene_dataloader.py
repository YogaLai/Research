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
import os

def get_kitti_cycle_data(file_path_train, path):
    f_train = open(file_path_train)
    former_left_image_train = list()
    latter_left_image_train = list()
    former_right_image_train = list()
    latter_right_image_train = list()
    
    for line in f_train:
        former_left_image_train.append(path+line.split()[0])
        latter_left_image_train.append(path+line.split()[2])
        former_right_image_train.append(path+line.split()[1])
        latter_right_image_train.append(path+line.split()[3])
        
    return former_left_image_train, latter_left_image_train, former_right_image_train, latter_right_image_train

def get_data(file_path_test, path):
    f_test = open(file_path_test)
    left_image_test = list()
    right_image_test = list()

    for line in f_test:
        left_image_test.append(path+line.split()[0])
        right_image_test.append(path+line.split()[1])
        
    return left_image_test, right_image_test

def get_middleburry_data(path, is_perfect=True):
    if is_perfect:
        path = os.path.join(path, 'perfect')
    else:
        path = os.path.join(path, 'imperfect')

    left = []
    right = []
    for subdir in os.listdir(path):
        scene = os.path.join(path, subdir)
        left.append(os.path.join(scene, 'im0.png'))
        right.append(os.path.join(scene, 'im1.png'))
    
    return left, right

def get_flow_data(file_path_test, path):
    f_test = open(file_path_test)
    flow_test = list()
    former_image_test = list()
    latter_image_test = list()

    for line in f_test:
        former_image_test.append(path+line.split()[0])
        latter_image_test.append(path+line.split()[1])
        flow_test.append(path+line.split()[2])
        
    return former_image_test, latter_image_test, flow_test

def get_transform(param, resize):
    if resize:
        return transforms.Compose([
            transforms.Resize([param.input_height, param.input_width]),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor()
        ])
    
class myCycleImageFolder(data.Dataset):
    def __init__(self, left1, left2, right1, right2, training, param, resize=True):
        self.right1 = right1
        self.left1 = left1
        self.right2 = right2
        self.left2 = left2
        self.training = training
        self.param = param
        self.resize = resize
        
    def __getitem__(self, index):
        left1 = self.left1[index]
        right1 = self.right1[index]
        left2 = self.left2[index]
        right2 = self.right2[index]
        param = self.param
        try:
            left_image_1 = Image.open(left1).convert('RGB')
            right_image_1 = Image.open(right1).convert('RGB')
            left_image_2 = Image.open(left2).convert('RGB')
            right_image_2 = Image.open(right2).convert('RGB')

        except:
            print('\read image error: \n', left1)
            exit()

        
        #augmentation
        if self.training:
            
            #randomly flip
            if random.uniform(0, 1) > 0.5:
                left_image_1 = left_image_1.transpose(Image.FLIP_LEFT_RIGHT)
                right_image_1 = right_image_1.transpose(Image.FLIP_LEFT_RIGHT)
                left_image_2 = left_image_2.transpose(Image.FLIP_LEFT_RIGHT)
                right_image_2 = right_image_2.transpose(Image.FLIP_LEFT_RIGHT)
                
            #randomly shift gamma
            if random.uniform(0, 1) > 0.5:
                gamma = random.uniform(0.8, 1.2)
                left_image_1 = Image.fromarray(np.clip((np.array(left_image_1) ** gamma), 0, 255).astype('uint8'), 'RGB')
                right_image_1 = Image.fromarray(np.clip((np.array(right_image_1) ** gamma), 0, 255).astype('uint8'), 'RGB')
                left_image_2 = Image.fromarray(np.clip((np.array(left_image_2) ** gamma), 0, 255).astype('uint8'), 'RGB')
                right_image_2 = Image.fromarray(np.clip((np.array(right_image_2) ** gamma), 0, 255).astype('uint8'), 'RGB')
            
            #randomly shift brightness
            if random.uniform(0, 1) > 0.5:
                brightness = random.uniform(0.5, 2.0)
                left_image_1 = Image.fromarray(np.clip((np.array(left_image_1) * brightness), 0, 255).astype('uint8'), 'RGB')
                right_image_1 = Image.fromarray(np.clip((np.array(right_image_1) * brightness), 0, 255).astype('uint8'), 'RGB')
                left_image_2 = Image.fromarray(np.clip((np.array(left_image_2) * brightness), 0, 255).astype('uint8'), 'RGB')
                right_image_2 = Image.fromarray(np.clip((np.array(right_image_2) * brightness), 0, 255).astype('uint8'), 'RGB')
            
            #randomly shift color
            if random.uniform(0, 1) > 0.5:
                colors = [random.uniform(0.8, 1.2) for i in range(3)]
                shape = np.array(left_image_1).shape
                white = np.ones((shape[0], shape[1]))
                color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
                left_image_1 = Image.fromarray(np.clip((np.array(left_image_1) * color_image), 0, 255).astype('uint8'), 'RGB')
                right_image_1 = Image.fromarray(np.clip((np.array(right_image_1) * color_image), 0, 255).astype('uint8'), 'RGB')
                left_image_2 = Image.fromarray(np.clip((np.array(left_image_2) * color_image), 0, 255).astype('uint8'), 'RGB')
                right_image_2 = Image.fromarray(np.clip((np.array(right_image_2) * color_image), 0, 255).astype('uint8'), 'RGB')
                
        
        #transforms
        process = get_transform(param, self.resize)
        left_image_1 = process(left_image_1)
        right_image_1 = process(right_image_1)
        left_image_2 = process(left_image_2)
        right_image_2 = process(right_image_2)
        
        return left_image_1, left_image_2, right_image_1, right_image_2
    def __len__(self):
        return len(self.left1)
    
class myImageFolder(data.Dataset):
    def __init__(self, left, right, flow, param, noc_flow=None, disp_gt=None, resize=True):
        self.right = right
        self.left = left
        self.flow = flow
        self.noc_flow = noc_flow
        self.param = param
        self.disp_gt = disp_gt
        self.resize = resize
        
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        param = self.param
        try:
            left_image = Image.open(left).convert('RGB')
            right_image = Image.open(right).convert('RGB')
        except:
            print('Erro open PNG: ', left)
            print('Erro open PNG: ', right)
            exit()

        # w,h = left_image.size
        # th = int(int(h / 64) * 64)
        # tw = int(int(w / 64) * 64)
        # param.input_width = tw
        # param.input_height = th
 
        process = get_transform(param, self.resize)
        left_image = process(left_image)
        right_image = process(right_image)
        
        if self.flow is not None:
            flow = self.flow[index]
            flow_image = cv2.imread(flow, -1)
            h, w, _ = flow_image.shape
            flo_img = flow_image[:,:,2:0:-1].astype(np.float32)
            invalid = (flow_image[:,:,0] == 0)

            flo_img = (flo_img - 32768) / 64
            flo_img[np.abs(flo_img) < 1e-10] = 1e-10
            flo_img[invalid, :] = 0

            f = torch.from_numpy(flo_img.transpose((2,0,1)))
            mask = torch.from_numpy((flow_image[:,:,0] == 1).astype(np.float32)).type(torch.FloatTensor)

            if self.noc_flow != None:
                flow = self.noc_flow[index]
                flow_image = cv2.imread(flow, -1)
                h, w, _ = flow_image.shape
                flo_img = flow_image[:,:,2:0:-1].astype(np.float32)
                invalid = (flow_image[:,:,0] == 0)

                flo_img = (flo_img - 32768) / 64
                flo_img[np.abs(flo_img) < 1e-10] = 1e-10
                flo_img[invalid, :] = 0

                noc_f = torch.from_numpy(flo_img.transpose((2,0,1))).type(torch.FloatTensor)
                # mask = torch.from_numpy((flow_image[:,:,0] == 1).astype(np.float32)).type(torch.FloatTensor)

                return left_image, right_image, f.type(torch.FloatTensor), noc_f, mask, h, w
                
            return left_image, right_image, f.type(torch.FloatTensor), mask, h, w
        
        if self.disp_gt is not None:
            disp_gt = self.disp_gt[index]
            disp_gt = Image.open(disp_gt)
            w, h = disp_gt.size
            # disp_gt = disp_gt.resize([1232, 368])
            disp_gt = disp_gt.crop((w-1232, h-368, w, h))
            disp_gt = np.ascontiguousarray(disp_gt,dtype=np.float32)/256

            return left_image, right_image, disp_gt

        return left_image, right_image
    
    def __len__(self):
        return len(self.left)    

def get_kitti_2015(dataset_path):
    left_image = []
    right_image = []
    gt = []
    total_num = len(os.listdir(os.path.join(dataset_path, "RGB_left")))//2
    
    for i in range(total_num):
        left_image.append(os.path.join(dataset_path, "RGB_left", str(i).zfill(6) + "_10.png"))
        right_image.append(os.path.join(dataset_path, "RGB_right", str(i).zfill(6) + "_10.png"))
        gt.append(os.path.join(dataset_path, "disp_occ_0", str(i).zfill(6) + "_10.png"))
    
    return left_image, right_image, gt