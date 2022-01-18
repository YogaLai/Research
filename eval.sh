#!/bin/bash 
cmd="python utils/evaluate_kitti.py --gt_path D:/yoga/BSP_lab/temporal_stereo/xiaojie_sample/dataset_kitti/test/kitti_2015/ --exp_name try_pwc_small --predicted_disp_path $1" 
# cmd="python utils/eval_and_show_curve.py --gt_path D:/yoga/BSP_lab/temporal_stereo/xiaojie_sample/dataset_kitti/test/kitti_2015/ --exp_name $1 $2 $3" 
echo ${cmd}; 
eval ${cmd};