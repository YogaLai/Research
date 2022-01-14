#!/bin/bash
cmd="python test_stereo_loop.py --model_name pwc --epoch_thres 4 --data_path ../dataset_kitti/test/kitti_2015/ --filenames_file utils/filenames/kitti_stereo_2015_rgb.txt --checkpoint_path savemodel/pretrain_soft_mask_wo_lr --exp_name pretrain_soft_mask_wo_lr $1"
echo ${cmd}
eval ${cmd}