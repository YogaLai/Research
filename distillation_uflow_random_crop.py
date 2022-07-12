import torch
import torch.optim as optim
import argparse
# from models.PWC_net import *
# from models.PWC_net_concat_cv import PWCDCNet
# from models.PWC_net_small_attn import PWCDCNet
from models.UFlow_wo_residual import PWCDCNet
# from models.PWC_net_dc_cost_uflow import PWCDCNet
# from models.PWC_net_small_attn_LD_biup import PWCDCNet
from utils.scene_dataloader import *
from utils.utils import *
from networks.resample2d_package.resample2d import Resample2d
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from torchvision.transforms.functional import crop
from utils.evaluation_utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',                     type=str,   help='data split, kitti or eigen',         default='kitti')
    parser.add_argument('--model_name',                type=str,   help='model name', default='pwc')
    parser.add_argument('--gt_path',                   type=str,   help='path to ground truth disparities',   required=True)
    parser.add_argument('--data_path',                 type=str,   help='path to the stereo data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--test_filenames_file',       type=str,   help='path to the testing filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=256)
    parser.add_argument('--input_width',               type=int,   help='input width', default=512)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=2)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=80)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=0.5)
    parser.add_argument('--msd_loss_weight',           type=float, help='multi scale distillation weight', default=0.05)
    parser.add_argument('--smooth_loss_weight',        type=float, help='smooth loss weight', default=10)
    parser.add_argument('--census_loss_weight',        type=float, help='census loss weight', default=0.5)
    parser.add_argument('--selfsup_loss_weight',       type=float, help='self-supervised distillation weight', default=0.1)
    parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
    parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='savemodel/')
    parser.add_argument('--type_of_2warp',             type=int,   help='2warp type', default=0)
    parser.add_argument('--num_scales',                type=int,   help='number of scales', default=4)
    parser.add_argument('--exp_name',                  type=str,   help='experiment name')
    parser.add_argument('--loadmodel',                 type=str,   help='the path of model weight')
    parser.add_argument('--min_depth',                 type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth for evaluation',  default=80)
    parser.add_argument('--use_census_loss',           help='enable census loss calculation', action="store_true")
    args = parser.parse_args()
    return args

# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
torch.backends.cudnn.benchmark = True
torch.manual_seed(71)
torch.cuda.manual_seed(71)
random.seed(71)
np.random.seed(71)
args = get_args()
epoch = 0
writer = SummaryWriter('logs/' + args.exp_name)
torch.backends.cudnn.benchmark = True

if not os.path.isdir('savemodel/' + args.exp_name):
    os.makedirs('savemodel/' + args.exp_name)

net = PWCDCNet().cuda()
teacher_net = PWCDCNet().cuda().eval()
args.input_width = 640
args.input_height = 640

left_image_1, left_image_2, right_image_1, right_image_2 = get_kitti_cycle_data(args.filenames_file, args.data_path)
left_image_test, right_image_test = get_data(args.test_filenames_file, args.gt_path)
CycleLoader = torch.utils.data.DataLoader(
    myCycleImageFolder(left_image_1, left_image_2, right_image_1, right_image_2, True, args),
    batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=4)
TestImageLoader = torch.utils.data.DataLoader(
    myImageFolder(left_image_test, right_image_test, None, args),
    shuffle = False, drop_last=False)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7, 10, 13], gamma=0.5)

flow_filenames_file = 'utils/filenames/kitti_flow_val_files_occ_200.txt'
flow_noc_filename = flow_filenames_file.replace('occ_', '')
former_test, latter_test, flow = get_flow_data(flow_filenames_file, args.gt_path)
former_test, latter_test, noc_flow = get_flow_data(flow_noc_filename, args.gt_path)
TestFlowLoader = torch.utils.data.DataLoader(
        myImageFolder(former_test, latter_test, flow, args, noc_flow=noc_flow),
        batch_size = 1, shuffle = False, drop_last = False)

if args.loadmodel:
    checkpoint = torch.load(args.loadmodel)
    net.load_state_dict(checkpoint['state_dict'])
    teacher_net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    # scheduler = checkpoint['scheduler']
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter = int(epoch * len(CycleLoader.dataset) / args.batch_size) + 1

if torch.cuda.device_count() >= 2:
    net = nn.DataParallel(net)
    teacher_net = nn.DataParallel(teacher_net)

def train(epoch, dataloader, net, optimizer, scheduler, writer, args):
    slices = []
    for i in range(4):
        slices.append(list(range(i * args.batch_size, (i + 1) * args.batch_size)))

    total_image_loss = 0
    total_image_loss_2 = 0
    total_disp_gradient_loss = 0
    total_disp_gradient_loss_2 = 0
    total_lr_loss = 0
    total_msd_loss = 0
    total_distillation_loss = 0
    total_distillation_loss_2 = 0
    if args.use_census_loss:
        total_census_image_loss = 0
        total_census_image_loss_2 = 0
    if args.type_of_2warp > 0:
        total_warp2loss = 0
        total_warp2loss_2 = 0
    iter = int(epoch * len(CycleLoader.dataset) / args.batch_size) + 1

    net.train()
    with tqdm(total=len(dataloader.dataset)) as pbar:
        for batch_idx, (left_image_1, left_image_2, right_image_1, right_image_2) in enumerate(dataloader):
            optimizer.zero_grad()

            h, w = left_image_1.size(2), left_image_1.size(3)
            crop_x = random.randint(0, w - (args.input_width-64))
            crop_y = random.randint(0, h - (args.input_height-64))
            full_former = torch.cat((left_image_2, left_image_1, right_image_1, left_image_1), 0).cuda()
            full_latter = torch.cat((right_image_2, left_image_2, right_image_2, right_image_1), 0).cuda()

            former = crop(full_former, crop_y, crop_x, args.input_height-64, args.input_width-64)
            latter = crop(full_latter, crop_y, crop_x, args.input_height-64, args.input_width-64)

            model_input = torch.cat((full_former, full_latter), 1)
            model_input_2 = torch.cat((full_latter, full_former), 1)

            crop_model_input = torch.cat((former, latter), 1)
            crop_model_input_2 = torch.cat((latter, former), 1)

            with torch.no_grad():
                teacher_disp_est_scale, teacher_flows = teacher_net(model_input)
                teacher_disp_est_scale_2, teacher_flows_2 = teacher_net(model_input_2)

            disp_est_scale, flows = net(crop_model_input)
            disp_est = torch.cat((disp_est_scale[:, 0, :, :].unsqueeze(1) / disp_est_scale.shape[3],
                                   disp_est_scale[:, 1, :, :].unsqueeze(1) / disp_est_scale.shape[2]), 1)
            disp_est_scale_2, flows_2 = net(crop_model_input_2)
            disp_est_2 = torch.cat((disp_est_scale_2[:, 0, :, :].unsqueeze(1) / disp_est_scale_2.shape[3],
                                     disp_est_scale_2[:, 1, :, :].unsqueeze(1) / disp_est_scale_2.shape[2]), 1)

            border_mask = create_border_mask(former, 0.1)
            teacher_border_mask = create_border_mask(full_former, 0.1)
            fw, bw, _, _ = get_mix_mask(disp_est_scale, disp_est_scale_2, border_mask, slices[1]+slices[2])
            # fw += 1e-3
            # bw += 1e-3
            occ_fw = 1 - fw
            occ_bw = 1 - bw
            fw_mask = fw.clone().detach()
            bw_mask = bw.clone().detach()

            student_occ_fw = occ_fw.clone().detach()
            student_occ_bw = occ_bw.clone().detach()
            fw, bw, _, _ = get_mix_mask(teacher_disp_est_scale, teacher_disp_est_scale_2, teacher_border_mask, slices[1]+slices[2])
            # fw += 1e-3
            # bw += 1e-3
            occ_fw = 1 - fw
            occ_bw = 1 - bw
            full_occ_fw = occ_fw.clone().detach()
            full_occ_bw = occ_bw.clone().detach()

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

            # Census
            if args.use_census_loss:
                census_image_loss = census_loss(former, left_est, fw_mask)
                census_image_loss_2 = census_loss(latter, right_est, bw_mask)
             
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
                msd_loss_scale_fw = photo_loss_abs_robust(scale_fw - flow_fw_label_sacle, occ_mask=occ_scale_fw)
                msd_loss_scale_bw = photo_loss_abs_robust(scale_bw - flow_bw_label_sacle, occ_mask=occ_scale_bw) 
                msd_loss.append(msd_loss_scale_fw)
                msd_loss.append(msd_loss_scale_bw)
            msd_loss = sum(msd_loss)

            # Distillation
            crop_teacher_disp_est_scale = crop(teacher_disp_est_scale, crop_y, crop_x, args.input_height-64, args.input_width-64)
            crop_teacher_disp_est_scale_2 = crop(teacher_disp_est_scale_2, crop_y, crop_x, args.input_height-64, args.input_width-64)
            crop_teacher_occ_fw = crop(full_occ_fw, crop_y, crop_x, args.input_height-64, args.input_width-64)
            crop_teacher_occ_bw = crop(full_occ_bw, crop_y, crop_x, args.input_height-64, args.input_width-64)
            valid_mask_fw = torch.clamp(student_occ_fw - crop_teacher_occ_fw, 0, 1) # stduent occluded region and teacher non-occluded region 
            valid_mask_bw = torch.clamp(student_occ_bw - crop_teacher_occ_bw, 0, 1) # stduent occluded region and teacher non-occluded region 
            distillation_loss = photo_loss_abs_robust(disp_est_scale - crop_teacher_disp_est_scale, valid_mask_fw)
            distillation_loss_2 = photo_loss_abs_robust(disp_est_scale_2 - crop_teacher_disp_est_scale_2, valid_mask_bw)

            loss =  2 * (image_loss + image_loss_2) + args.smooth_loss_weight * (disp_gradient_loss + disp_gradient_loss_2) + args.lr_loss_weight * lr_loss + args.msd_loss_weight * msd_loss + args.selfsup_loss_weight * (distillation_loss + distillation_loss_2)
            if args.use_census_loss:
                loss += args.census_loss_weight * (census_image_loss + census_image_loss_2)

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
            writer.add_scalar('iter/distillation', distillation_loss.data, iter)
            writer.add_scalar('iter/distillation_2', distillation_loss_2.data, iter)
            if args.use_census_loss:
                writer.add_scalar('iter/census_loss', census_image_loss.data, iter)
                writer.add_scalar('iter/census_loss_2', census_image_loss_2.data, iter)
            if args.type_of_2warp > 0:
                writer.add_scalar('iter/warp2loss', warp2loss.data, iter)
                writer.add_scalar('iter/warp2loss_2', warp2loss_2.data, iter)

            total_image_loss += float(image_loss)
            total_image_loss_2 += float(image_loss_2)
            total_disp_gradient_loss += float(disp_gradient_loss)
            total_disp_gradient_loss_2 += float(disp_gradient_loss_2)
            total_lr_loss += float(lr_loss)
            total_msd_loss += float(msd_loss)
            total_distillation_loss += float(distillation_loss)
            total_distillation_loss_2 += float(distillation_loss_2)
            if args.use_census_loss:
                total_census_image_loss += float(census_image_loss)
                total_census_image_loss_2 += float(census_image_loss_2)
            if args.type_of_2warp > 0:
                total_warp2loss += float(warp2loss)
                total_warp2loss_2 += float(warp2loss_2)

            if iter % 100 == 0:
                writer.add_images('fw_mask', fw_mask, iter)
                writer.add_images('bw_mask', bw_mask, iter)
                writer.add_images('left_rgb', left_image_1, iter)
                writer.add_images('right_rgb', right_image_1, iter)
                writer.add_images('rec_left', left_est, iter)

            # if (iter + 1) % 600 == 0:
            #     # state = {'iter': iter, 'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}
            #     state = {'iter': iter, 'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            #     torch.save(state, "savemodel/" + args.exp_name + "/model_iter" + str(iter))
            #     print("The model of iter ", iter, "has been saved.")

            iter += 1
            pbar.set_description(
                f"loss: {loss.item():.5f}"
            )
            pbar.update(left_image_1.size(0))

    # scheduler.step()

    writer.add_scalar('epoch/rec_loss', total_image_loss / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/rec_loss_2', total_image_loss_2 / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/smooth_loss', total_disp_gradient_loss / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/smooth_loss_2', total_disp_gradient_loss_2 / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/lr_consistency', total_lr_loss / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/msd_loss', total_msd_loss / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/distillation_loss', total_distillation_loss / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/distillation_loss_2', total_distillation_loss_2 / len(CycleLoader.dataset), epoch)
    if args.use_census_loss:
        writer.add_scalar('epoch/census_loss', total_census_image_loss / len(CycleLoader.dataset), epoch)
        writer.add_scalar('epoch/census_loss_2', total_census_image_loss_2 / len(CycleLoader.dataset), epoch)
    if args.type_of_2warp > 0:
        writer.add_scalar('epoch/warp2loss', total_warp2loss / len(CycleLoader.dataset), epoch)
        writer.add_scalar('epoch/warp2loss_2', total_warp2loss_2 / len(CycleLoader.dataset), epoch)

    # state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}
    state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, "savemodel/" + args.exp_name + "/model_epoch" + str(epoch))
    print("The model of epoch ", epoch, "has been saved.")

def test_stereo(dataloader, net, gt, writer, epoch):
    pred_disparities = []
    net.eval()
    with torch.no_grad():
        for left, right in dataloader:
            model_input = torch.cat((left, right), 1).cuda()

            disp_est_scale = net(model_input)
            if isinstance(disp_est_scale, tuple):
                disp_est_scale = disp_est_scale[0]
                disp_est =  torch.cat((disp_est_scale[:,0,:,:].unsqueeze(1) / disp_est_scale.shape[3], disp_est_scale[:,1,:,:].unsqueeze(1) / disp_est_scale.shape[2]), 1)
                pred_disparities.append(-disp_est[0,0,:,:].data.cpu().numpy())
            else:
                disp_est = [torch.cat((disp_est_scale[i][:,0,:,:].unsqueeze(1) / disp_est_scale[i].shape[3], disp_est_scale[i][:,1,:,:].unsqueeze(1) / disp_est_scale[i].shape[2]), 1) for i in range(1)]
                pred_disparities.append(-disp_est[0][0,0,:,:].data.cpu().numpy())

    abs_rel = np.zeros(len(gt))
    if args.split == 'kitti':
        gt_disparities = gt
        gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_disparities)
        for i in range(len(gt)):
            gt_disp = gt_disparities[i]
            mask = gt_disp > 0
            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]

            pred_depth[pred_depth < args.min_depth] = args.min_depth
            pred_depth[pred_depth > args.max_depth] = args.max_depth

            abs_rel[i] = np.mean(np.abs(gt_depth[mask] - pred_depth[mask]) / gt_depth[mask])

    elif args.split == 'eigen':
        pred_depths = []
        gt_depths = gt
        for i in range(len(gt_depths)):
            disp_pred = cv2.resize(pred_disparities[i], (im_sizes[i][1], im_sizes[i][0]), interpolation=cv2.INTER_LINEAR)
            disp_pred = disp_pred * disp_pred.shape[1]

            # need to convert from disparity to depth
            focal_length, baseline = get_focal_length_baseline(gt_calib[i], cams[i])
            depth_pred = (baseline * focal_length) / disp_pred
            depth_pred[np.isinf(depth_pred)] = 0

            pred_depths.append(depth_pred)

            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]

            pred_depth[pred_depth < args.min_depth] = args.min_depth
            pred_depth[pred_depth > args.max_depth] = args.max_depth

            mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)
            gt_height, gt_width = gt_depth.shape
            crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                                0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

           
            # sq_rel[i] = np.mean(((gt_depth[mask] - pred_depth[mask])**2) / gt_depth[mask])
            abs_rel[i] = np.mean(np.abs(gt_depth[mask] - pred_depth[mask]) / gt_depth[mask])

    abs_rel = abs_rel.mean()
    print('Abs_rel: ', abs_rel)
    writer.add_scalar('Test/abs_rel', abs_rel, epoch)
    return abs_rel

def test_flow(dataloader, net, writer, epoch):
    total_error = 0
    total_epe_noc = 0
    total_epe_occ = 0
    fl_error = 0
    num_test = 0
    for left, right, gt, noc_gt, mask, h, w in dataloader:
        left_batch = torch.cat((left, torch.from_numpy(np.flip(left.numpy(), 3).copy())), 0)
        right_batch = torch.cat((right, torch.from_numpy(np.flip(right.numpy(), 3).copy())), 0)
        
        with torch.no_grad():
            left = left_batch.cuda()
            right = right_batch.cuda()
            model_input = torch.cat((left, right), 1)
            disp_est_scale = net(model_input)

            mask = np.ceil(np.clip(np.abs(gt[0,0]), 0, 1))
            noc_mask = np.ceil(np.clip(np.abs(noc_gt[0,0]), 0, 1))

            disp_ori_scale = nn.UpsamplingBilinear2d(size=(int(h), int(w)))(disp_est_scale[0][:1])
            disp_ori_scale[0,0] = disp_ori_scale[0,0] * int(w) / args.input_width
            disp_ori_scale[0,1] = disp_ori_scale[0,1] * int(h) / args.input_height

            epe_all, epe_noc, epe_occ, fl = evaluate_flow(disp_ori_scale[0].data.cpu().numpy(), gt[0].numpy(), mask.numpy(), noc_mask.numpy())
            total_error += epe_all
            total_epe_noc += epe_noc
            total_epe_occ += epe_occ
            fl_error += fl
            num_test += 1

    total_error /= num_test 
    total_epe_noc /= num_test
    total_epe_occ /= num_test
    fl_error /= num_test
    print("EPE-noc: ", total_epe_noc)
    print("EPE-all: ", total_error)
    print("EPE-occ: ", total_epe_occ)
    print("Fl: ", fl_error)

    writer.add_scalar('Test/EPE-noc', total_epe_noc, epoch)
    writer.add_scalar('Test/EPE-all', total_error, epoch)
    writer.add_scalar('Test/EPE-occ', total_epe_occ, epoch)
    writer.add_scalar('Test/Fl', fl_error, epoch)

    return total_error


best_stereo_metric = 10000
best_flow_metric = 10000
gt_disparities = load_gt_disp_kitti(args.gt_path)
start_epoch = epoch
for epoch in range(start_epoch, args.num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(epoch, CycleLoader, net, optimizer, scheduler, writer, args)
    if args.split == 'kitti':
        stereo_err = test_stereo(TestImageLoader, net, gt_disparities, writer, epoch)
        flow_err = test_flow(TestFlowLoader, net, writer, epoch)
    if stereo_err < best_stereo_metric:
        state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, "savemodel/" + args.exp_name + "/stereo_best")
    if flow_err < best_flow_metric:
        # state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}
        state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, "savemodel/" + args.exp_name + "/flow_best")