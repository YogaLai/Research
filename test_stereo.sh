#!/bin/bash

# cmd="python ./test_stereo_loop.py --model_name pwc --data_path D:/yoga/BSP_lab/temporal_stereo/xiaojie_sample/dataset_kitti/test/kitti_2015/ --checkpoint_path ./savemodel/sparse_volume_pretrain --filenames_file ./utils/filenames/kitti_stereo_2015_rgb.txt --epoch_thres 12 --exp_name sparse_volume_pretrain $1"
cmd="python ./test_stereo.py --model_name pwc --data_path D:/yoga/BSP_lab/temporal_stereo/xiaojie_sample/dataset_kitti/test/kitti_2015/ --checkpoint_path savemodel/try_pwc_small_v2/model_iter7499 --filenames_file ./utils/filenames/kitti_stereo_2015_rgb.txt --exp_name try_pwc_small $1"
echo ${cmd}
eval ${cmd}