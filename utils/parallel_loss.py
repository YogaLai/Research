import torch
import torch.nn as nn
from networks.resample2d_package.resample2d import Resample2d
from utils.utils import census_loss

class ParallelLoss(nn.Module):
    def __init__(self):
        super(ParallelLoss, self).__init__()
        # self.former = former
        # self.latter = latter

    def forward(self, former, latter, disp_est_scale, disp_est_scale_2, fw_mask, bw_mask):
        left_est = Resample2d()(latter, disp_est_scale)
        image_loss = census_loss(former, left_est, fw_mask)
        right_est = Resample2d()(former, disp_est_scale_2)
        image_loss_2 = census_loss(latter, right_est, bw_mask) 

        return image_loss, image_loss_2