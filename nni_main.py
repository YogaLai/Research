from nni.experiment import Experiment
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'
search_space = {
    'rec_loss_weight': {'_type': 'choice', '_value': [1, 0.5]},
    'smooth_loss_weight': {'_type': 'choice', '_value': [10, 0.05]},
    'lr_loss_weight': {'_type': 'choice', '_value': [0.1, 0.5]},
    'msd_loss_weight': {'_type': 'choice', '_value': [0.01, 0.05]},
}

experiment = Experiment('local')
experiment.config.trial_command = "python nni_model.py \
                                   --data_path /media/tiffanygpu/ImageNet/kitti_raw/ \
                                   --gt_path //media/tiffanygpu/ImageNet/kitti_raw/ \
                                   --filenames_file utils/filenames/eigen_cycle_8000.txt \
                                   --test_filenames_file utils/filenames/eigen_test_files.txt \
                                   --type_of_2warp -1 \
                                   --exp_name uflow_census_intensity255_nni_tuning \
                                   --batch_size 2 \
                                   --split eigen"
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
# experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
# Configure how many trials to run
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we evaluate 10 sets of hyperparameters in total, and concurrently evaluate 2 sets at a time.
experiment.config.max_trial_number = 20
experiment.config.trial_concurrency = 1
experiment.config.trial_gpu_number = 4
experiment.config.training_service.use_active_gpu = True
experiment.config.experiment_working_directory = '/media/tiffanygpu/ImageNet/yoga/Research/nni_experiments'

experiment.run(8081)
# experiment.stop()