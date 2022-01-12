#!/bin/bash 
cmd="python utils/evaluate_kitti.py --gt_path D:/yoga/BSP_lab/temporal_stereo/xiaojie_sample/dataset_kitti/test/kitti_2015/ --exp_name mix_mask --predicted_disp_path $1" 
echo ${cmd}; 
eval ${cmd};