#!/bin/bash
cmd="python utils/evaluate_kitti.py --gt_path ../dataset_kitti/test/kitti_2015/ --exp_name pretrain_soft_mask_wo_lr --predicted_disp_path $1"
echo ${cmd};
eval ${cmd};