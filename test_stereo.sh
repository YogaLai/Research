#!/bin/bash

# cmd="python ./test_stereo_loop.py --model_name pwc --data_path D:/yoga/BSP_lab/temporal_stereo/xiaojie_sample/dataset_kitti/test/kitti_2015/ --checkpoint_path ./savemodel/mix_mask --filenames_file ./utils/filenames/kitti_stereo_2015_rgb.txt --epoch_thres 11 --exp_name mix_mask $1"
cmd="python ./test_stereo.py --model_name pwc --data_path D:/yoga/BSP_lab/temporal_stereo/xiaojie_sample/dataset_kitti/test/kitti_2015/ --checkpoint_path ./savemodel/2warp_cross_mask_after_pretrain/model_iter316499 --filenames_file ./utils/filenames/kitti_stereo_2015_rgb.txt --exp_name 2warp_continue_cross_mask $1"
echo ${cmd}
eval ${cmd}