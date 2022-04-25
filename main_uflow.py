import torch
import torch.optim as optim
import argparse
from models.UFlow import PWCDCNet
# from models.PWC_net_small_sparse import *
# from models.PWC_net_small_attn import *
# from models.PWC_stereo_concat_cv_small import PWCDCNet
# from models.PWC_net_concat_cv_small import PWCDCNet
from models.DICL import dicl_wrapper
from utils.scene_dataloader import *
from utils.utils import *
from networks.resample2d_package.resample2d import Resample2d
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from config import cfg, cfg_from_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',                type=str,   help='model name', default='pwc')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=256)
    parser.add_argument('--input_width',               type=int,   help='input width', default=512)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=2)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=80)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=0.5)
    parser.add_argument('--msd_loss_weight',           type=float, help='multi scale distillation weight', default=0.01)
    parser.add_argument('--smooth_loss_weight',        type=float, help='smooth loss weight', default=0.05)
    parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
    parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='savemodel/')
    parser.add_argument('--type_of_2warp',             type=int,   help='2warp type', default=0)
    parser.add_argument('--num_scales',                type=int,   help='number of scales', default=4)
    parser.add_argument('--exp_name',                  type=str,   help='experiment name')
    parser.add_argument('--loadmodel',                 type=str,   help='the path of model weight')
    args = parser.parse_args()
    return args


args = get_args()
writer = SummaryWriter('logs/' + args.exp_name)
iter = 0
start_epoch = 0

if not os.path.isdir('savemodel/' + args.exp_name):
    os.makedirs('savemodel/' + args.exp_name)

if args.model_name == 'pwc':
    net = PWCDCNet().cuda()
    args.input_width = 768
elif args.model_name == 'dicl':
    cfg_from_file('cfgs/dicl5_kitti.yml')
    net = dicl_wrapper().cuda()
    args.input_width = 768


left_image_1, left_image_2, right_image_1, right_image_2 = get_kitti_cycle_data(args.filenames_file, args.data_path)
cycle_loader_aug = torch.utils.data.DataLoader(
    myCycleImageFolder(left_image_1, left_image_2, right_image_1, right_image_2, True, args),
    batch_size=args.batch_size, shuffle=True, drop_last=False)
cycle_loader = torch.utils.data.DataLoader(
    myCycleImageFolder(left_image_1, left_image_2, right_image_1, right_image_2, False, args),
    batch_size=args.batch_size, shuffle=True, drop_last=False)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7, 10, 13], gamma=0.5)

slices = []
for i in range(4):
    slices.append(list(range(i * args.batch_size, (i + 1) * args.batch_size)))

# if torch.cuda.device_count() >= 2:
#     net = nn.DataParallel(net)

if args.loadmodel:
    checkpoint = torch.load(args.loadmodel)
    net.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    scheduler = checkpoint['scheduler']
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter = int(start_epoch * len(cycle_loader.dataset) / args.batch_size) + 1

for epoch in range(start_epoch, args.num_epochs):
    print("Epoch :", epoch)
    total_image_loss = 0
    total_image_loss_2 = 0
    total_disp_gradient_loss = 0
    total_disp_gradient_loss_2 = 0
    total_lr_loss = 0
    total_msd_loss = 0
    if args.type_of_2warp > 0:
        total_warp2loss = 0
        total_warp2loss_2 = 0

    CycleLoader = cycle_loader_aug if epoch >= 2 else cycle_loader
    with tqdm(total=len(CycleLoader.dataset)) as pbar:
        for batch_idx, (left_image_1, left_image_2, right_image_1, right_image_2) in enumerate(CycleLoader):
            optimizer.zero_grad()

            former = torch.cat((left_image_2, left_image_1, right_image_1, left_image_1), 0).cuda()
            latter = torch.cat((right_image_2, left_image_2, right_image_2, right_image_1), 0).cuda()

            model_input = torch.cat((former, latter), 1)
            model_input_2 = torch.cat((latter, former), 1)

            disp_est_scale, flows = net(model_input)
            disp_est = torch.cat((disp_est_scale[:, 0, :, :].unsqueeze(1) / disp_est_scale.shape[3],
                                   disp_est_scale[:, 1, :, :].unsqueeze(1) / disp_est_scale.shape[2]), 1)
            disp_est_scale_2, flows_2 = net(model_input_2)
            disp_est_2 = torch.cat((disp_est_scale_2[:, 0, :, :].unsqueeze(1) / disp_est_scale_2.shape[3],
                                     disp_est_scale_2[:, 1, :, :].unsqueeze(1) / disp_est_scale_2.shape[2]), 1)

            border_mask = create_border_mask(former, 0.1)
            fw, bw, diff_fw, diff_bw = get_mask(disp_est_scale, disp_est_scale_2, border_mask)
            fw += 1e-3
            bw += 1e-3
            fw[slices[0] + slices[-1]] = fw[slices[0] + slices[-1]] * 0 + 1
            bw[slices[0] + slices[-1]] = bw[slices[0] + slices[-1]] * 0 + 1
            fw_mask = fw.clone().detach()
            bw_mask = bw.clone().detach()

            # try census
            # left_est = [Resample2d()(right_pyramid[i], disp_est_scale[i]) for i in range(4)]
            # closs_left = [census_loss(left_pyramid[i], left_est[i], fw_mask[i]) for i in range(4)]
            # image_loss = sum(closs_left)

            # try census
            # right_est = [Resample2d()(left_pyramid[i], disp_est_scale_2[i]) for i in range(4)]
            # closs_right = [census_loss(right_pyramid[i], right_est[i], bw_mask[i]) for i in range(4)]
            # image_loss_2 = sum(closs_right)

            # Reconstruction from right to left
            left_est = Resample2d()(latter, disp_est_scale)
            l1_left = torch.abs(left_est - former) * fw_mask 
            l1_reconstruction_loss_left = torch.mean(l1_left) / torch.mean(fw_mask) 
            ssim_left = SSIM(left_est * fw_mask, former * fw_mask) 
            ssim_loss_left = torch.mean(ssim_left) / torch.mean(fw_mask) 
            image_loss = args.alpha_image_loss * ssim_loss_left + (1 - args.alpha_image_loss) * l1_reconstruction_loss_left

            # Reconstruction from left to right
            right_est = Resample2d()(former, disp_est_scale_2) 
            l1_right = torch.abs(right_est - latter) * bw_mask 
            l1_reconstruction_loss_right = torch.mean(l1_right) / torch.mean(bw_mask) 
            ssim_right = SSIM(right_est * bw_mask, latter * bw_mask) 
            ssim_loss_right = torch.mean(ssim_right) / torch.mean(bw_mask) 
            image_loss_2 =  args.alpha_image_loss * ssim_loss_right + (1 - args.alpha_image_loss) * l1_reconstruction_loss_right
            
            # Smooth loss
            disp_gradient_loss = cal_grad2_error(disp_est_scale / 20, former, 1.0) 
            disp_gradient_loss_2 = cal_grad2_error(disp_est_scale_2 / 20, latter, 1.0)

            # LR consistency
            right_to_left_disp = -Resample2d()(disp_est_2, disp_est_scale) 
            left_to_right_disp = -Resample2d()(disp_est, disp_est_scale_2)
            lr_left_loss = torch.mean(torch.abs(right_to_left_disp[slices[0] + slices[-1]] - disp_est[slices[0] + slices[-1]])) 
            lr_right_loss = torch.mean(torch.abs(left_to_right_disp[slices[0] + slices[-1]] - disp_est_2[slices[0] + slices[-1]]))
            lr_loss = lr_left_loss + lr_right_loss

            # Pyramid distillation
            flow_fw_label = disp_est_scale.clone().detach()
            flow_bw_label = disp_est_scale_2.clone().detach()
            msd_loss = []
            for i in range(len(flows)):
                scale_fw, scale_bw = flows[i], flows_2[i]
                flow_fw_label_sacle = upsample_flow(flow_fw_label, target_flow=scale_fw)
                occ_scale_fw = F.interpolate(fw_mask, [scale_fw.size(2), scale_fw.size(3)], mode='nearest')
                flow_bw_label_sacle = upsample_flow(flow_bw_label, target_flow=scale_bw)
                occ_scale_bw = F.interpolate(bw_mask, [scale_bw.size(2), scale_bw.size(3)], mode='nearest')
                msd_loss_scale_fw = photo_loss_abs_robust(x=scale_fw, y=flow_fw_label_sacle, occ_mask=occ_scale_fw)
                msd_loss_scale_bw = photo_loss_abs_robust(x=scale_bw, y=flow_bw_label_sacle, occ_mask=occ_scale_bw) 
                msd_loss.append(msd_loss_scale_fw)
                msd_loss.append(msd_loss_scale_bw)
            msd_loss = sum(msd_loss)

            # total_loss
            loss = image_loss + image_loss_2 + args.smooth_loss_weight * (disp_gradient_loss + disp_gradient_loss_2) + args.lr_loss_weight * lr_loss + args.msd_loss_weight * msd_loss
            """
            ##########################################################################################
            #                                                                                        #
            #   batch              7,8                mask for the direction of the reconstruction   #
            #   forward   L_t ------------> R_t                                                      #
            #              |                 |        mask   : L_t+1 ---> L_t   ---> R_t             #
            #          3,4 |                 | 5,6    mask_2 : L_t+1 ---> R_t+1 ---> R_t             #
            #              |                 |        mask_3 : R_t+1 ---> R_t   ---> L_t             #
            #              v                 v        mask_4 : R_t+1 ---> L_t+1 ---> L_t             #
            #             L_t+1 ----------> R_t+1     mask_5 : R_t   ---> L_t   ---> L_t+1           #
            #                      1,2                                                               #
            #                                                                                        #
            ##########################################################################################
            """

            if args.type_of_2warp == 1:
                mask_4 = [fw_mask[i][[2,3]] for i in range(4)]
                warp2_est_4 = [Resample2d()(left_est[i][[0,1]], disp_est_scale[i][[2,3]]) for i in range(4)]
                loss += 0.1 * sum([warp_2(warp2_est_4[i], left_pyramid[i][[6,7]], mask_4[i], args) for i in range(4)])
                mask_5 = [bw_mask[i][[2,3]] for i in range(4)]
                warp2_est_5 = [Resample2d()(left_est[i][[6,7]], disp_est_scale_2[i][[2,3]]) for i in range(4)]
                loss += 0.1 * sum([warp_2(warp2_est_5[i], left_pyramid[i][[0,1]], mask_5[i], args) for i in range(4)])

            elif args.type_of_2warp == 2:
                # mask = [Resample2d()(fw_mask[i][[2,3]], disp_est_scale_2[i][[0,1]]) for i in range(4)]
                mask = [Resample2d()(fw_mask[i][slices[1]], disp_est_scale_2[i][slices[0]]) for i in range(4)]
                # warp2_est = [Resample2d()(left_est[i][[2,3]], disp_est_scale_2[i][[6,7]]) for i in range(4)]
                warp2_est = [Resample2d()(left_est[i][slices[1]], disp_est_scale_2[i][slices[-1]]) for i in range(4)]
                # warp2loss = sum([warp_2(warp2_est[i], right_est[i][[6,7]], mask[i], args) for i in range(4)])
                warp2loss = sum([warp_2(warp2_est[i], right_est[i][slices[-1]], mask[i], args) for i in range(4)])
                loss += 0.1 * warp2loss
                # mask_3 = [Resample2d()(fw_mask[i][[4,5]], disp_est_scale[i][[0,1]]) for i in range(4)]
                mask_3 = [Resample2d()(fw_mask[i][slices[2]], disp_est_scale[i][slices[0]]) for i in range(4)]
                # warp2_est_3 = [Resample2d()(left_est[i][[4,5]], disp_est_scale[i][[6,7]]) for i in range(4)]
                warp2_est_3 = [Resample2d()(left_est[i][slices[2]], disp_est_scale[i][slices[-1]]) for i in range(4)]
                # warp2loss_2 = sum([warp_2(warp2_est_3[i], left_est[i][[6,7]], mask_3[i], args) for i in range(4)])
                warp2loss_2 = sum([warp_2(warp2_est_3[i], left_est[i][slices[-1]], mask_3[i], args) for i in range(4)])
                loss += 0.1 * warp2loss_2

            elif args.type_of_2warp == 3:
                mask = [Resample2d()(fw_mask[i][[2, 3]], disp_est_scale_2[i][[0, 1]]) for i in range(4)]
                warp2_est = [Resample2d()(left_est[i][[2, 3]], disp_est_scale_2[i][[6, 7]]) for i in range(4)]
                loss += 0.1 * sum([warp_2(warp2_est[i], right_pyramid[i][[6, 7]], mask[i], args) for i in range(4)])
                mask_2 = [fw_mask[i][[4, 5]] for i in range(4)]
                warp2_est_2 = [Resample2d()(right_est[i][[0, 1]], disp_est_scale[i][[4, 5]]) for i in range(4)]
                loss += 0.1 * sum([warp_2(warp2_est_2[i], right_pyramid[i][[6, 7]], mask_2[i], args) for i in range(4)])

            loss.backward()
            optimizer.step()

            writer.add_scalar('iter/rec_loss', image_loss.data, iter)
            writer.add_scalar('iter/rec_loss_2', image_loss_2.data, iter)
            writer.add_scalar('iter/smooth_loss', disp_gradient_loss.data, iter)
            writer.add_scalar('iter/smooth_loss_2', disp_gradient_loss_2.data, iter)
            writer.add_scalar('iter/lr_consistency', lr_loss.data, iter)
            writer.add_scalar('iter/msd_loss', msd_loss.data, iter)
            if args.type_of_2warp > 0:
                writer.add_scalar('iter/warp2loss', warp2loss.data, iter)
                writer.add_scalar('iter/warp2loss_2', warp2loss_2.data, iter)

            total_image_loss += float(image_loss)
            total_image_loss_2 += float(image_loss_2)
            total_disp_gradient_loss += float(disp_gradient_loss)
            total_disp_gradient_loss_2 += float(disp_gradient_loss_2)
            total_lr_loss += float(lr_loss)
            total_msd_loss += float(msd_loss)
            if args.type_of_2warp > 0:
                total_warp2loss += float(warp2loss)
                total_warp2loss_2 += float(warp2loss_2)

            if iter % 100 == 0:
                writer.add_images('fw_mask', fw_mask, iter)
                writer.add_images('bw_mask', bw_mask, iter)
                writer.add_images('rec_left', left_est, iter)
                writer.add_images('rec_right', right_est, iter)
                writer.add_images('left_rgb', left_image_1, iter)
                writer.add_images('right_rgb', right_image_1, iter)

            if (iter + 1) % 600 == 0:
                state = {'iter': iter, 'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}
                torch.save(state, "savemodel/" + args.exp_name + "/model_iter" + str(iter))
                print("The model of iter ", iter, "has been saved.")

            iter += 1
            pbar.set_description(
                f"loss: {loss.item():.5f}"
            )
            pbar.update(left_image_1.size(0))

    scheduler.step()

    writer.add_scalar('epoch/rec_loss', total_image_loss / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/rec_loss_2', total_image_loss_2 / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/smooth_loss', total_disp_gradient_loss / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/smooth_loss_2', total_disp_gradient_loss_2 / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/lr_consistency', total_lr_loss / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/msd_loss', total_msd_loss / len(CycleLoader.dataset), epoch)
    if args.type_of_2warp > 0:
        writer.add_scalar('epoch/warp2loss', total_warp2loss / len(CycleLoader.dataset), epoch)
        writer.add_scalar('epoch/warp2loss_2', total_warp2loss_2 / len(CycleLoader.dataset), epoch)

    # if epoch % 1 == 0:
    state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}
    torch.save(state, "savemodel/" + args.exp_name + "/model_epoch" + str(epoch))
    print("The model of epoch ", epoch, "has been saved.")
