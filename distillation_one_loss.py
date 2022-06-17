import torch
import torch.optim as optim
import argparse
# from models.PWC_net import *
# from models.PWC_net_concat_cv import PWCDCNet
# from models.PWC_net_small_attn import PWCDCNet
from models.UFlow_wo_residual import PWCDCNet
# from models.PWC_net_small_attn_LD_biup import PWCDCNet
from utils.scene_dataloader import *
from utils.utils import *
from networks.resample2d_package.resample2d import Resample2d
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
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
    parser.add_argument('--msd_loss_weight',           type=float, help='multi scale distillation weight', default=0.01)
    parser.add_argument('--smooth_loss_weight',        type=float, help='smooth loss weight', default=0.05)
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

os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
args.input_width = 832

left_image_1, left_image_2, right_image_1, right_image_2 = get_kitti_cycle_data(args.filenames_file, args.data_path)
left_image_test, right_image_test = get_data(args.test_filenames_file, args.gt_path)
CycleLoader = torch.utils.data.DataLoader(
    myCycleImageFolder(left_image_1, left_image_2, right_image_1, right_image_2, True, args),
    batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=8)
TestImageLoader = torch.utils.data.DataLoader(
         myImageFolder(left_image_test, right_image_test, None, args),
         batch_size = 1, shuffle = False, drop_last=False)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5, 7, 9], gamma=0.5)

flow_filenames_file = 'utils/filenames/kitti_flow_val_files_occ_200.txt'
flow_noc_filename = flow_filenames_file.replace('occ_', '')
former_test, latter_test, flow = get_flow_data(flow_filenames_file, args.gt_path)
former_test, latter_test, noc_flow = get_flow_data(flow_noc_filename, args.gt_path)
TestFlowLoader = torch.utils.data.DataLoader(
        myImageFolder(former_test, latter_test, flow, args, noc_flow=noc_flow),
        batch_size = 1, shuffle = False, num_workers = 1, drop_last = False)

if args.loadmodel:
    checkpoint = torch.load(args.loadmodel)
    net.load_state_dict(checkpoint['state_dict'])
    teacher_net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    scheduler = checkpoint['scheduler']
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter = int(epoch * len(CycleLoader.dataset) / args.batch_size) + 1

if torch.cuda.device_count() >= 2:
    net = nn.DataParallel(net)
    teacher_net = nn.DataParallel(teacher_net)

def gaussian_noise(x, mean=0, sigma=0.1):
    noise = torch.normal(mean, sigma, x.shape).to(x.device)
    gaussian_out = x + noise
    gaussian_out = torch.clamp(gaussian_out, 0, 1)
    return gaussian_out

def train(epoch, dataloader, net, optimizer, scheduler, writer, args):
    slices = []
    for i in range(4):
        slices.append(list(range(i * args.batch_size, (i + 1) * args.batch_size)))

    total_distillation_loss = 0
    total_distillation_loss_2 = 0
    iter = int(epoch * len(CycleLoader.dataset) / args.batch_size) + 1

    net.train()
    selfsup_transformations = get_selfsup_transformations(args, crop_size=32)
    with tqdm(total=len(dataloader.dataset)) as pbar:
        for batch_idx, (left_image_1, left_image_2, right_image_1, right_image_2) in enumerate(dataloader):
            optimizer.zero_grad()

            full_former = torch.cat((left_image_2, left_image_1, right_image_1, left_image_1), 0).cuda()
            full_latter = torch.cat((right_image_2, left_image_2, right_image_2, right_image_1), 0).cuda()

            former = selfsup_transformations(full_former)
            latter = selfsup_transformations(full_latter)

            model_input = torch.cat((full_former, full_latter), 1)
            model_input_2 = torch.cat((full_latter, full_former), 1)

            noise_latter = gaussian_noise((latter))
            noise_former = gaussian_noise((latter))
            crop_model_input = torch.cat((former, noise_latter), 1)
            crop_model_input_2 = torch.cat((latter, noise_former), 1)

            with torch.no_grad():
                teacher_disp_est_scale, teacher_flows = teacher_net(model_input)
                teacher_disp_est_scale_2, teacher_flows_2 = teacher_net(model_input_2)

            disp_est_scale, flows = net(crop_model_input)
            disp_est_scale_2, flows_2 = net(crop_model_input_2)

            teacher_border_mask = create_border_mask(full_former, 0.1)
            fw, bw, occ_fw, occ_bw = get_mix_mask(teacher_disp_est_scale, teacher_disp_est_scale_2, teacher_border_mask, slices[1]+slices[2])
            # fw += 1e-3
            # bw += 1e-3
            fw_mask = fw.clone().detach()
            bw_mask = bw.clone().detach()

            # Distillation
            crop_teacher_disp_est_scale = selfsup_transformations(teacher_disp_est_scale)
            crop_teacher_disp_est_scale_2 = selfsup_transformations(teacher_disp_est_scale_2)
            crop_teacher_fw_mask = selfsup_transformations(fw_mask)
            crop_teacher_bw_mask = selfsup_transformations(bw_mask)
            distillation_loss = photo_loss_abs_robust(disp_est_scale, crop_teacher_disp_est_scale, crop_teacher_fw_mask)
            distillation_loss_2 = photo_loss_abs_robust(disp_est_scale_2, crop_teacher_disp_est_scale_2, crop_teacher_bw_mask)

            loss =  args.selfsup_loss_weight * (distillation_loss + distillation_loss_2)
            loss.backward()
            optimizer.step()

            writer.add_scalar('iter/distillation', distillation_loss.data, iter)
            writer.add_scalar('iter/distillation_2', distillation_loss_2.data, iter)

            total_distillation_loss += float(distillation_loss)
            total_distillation_loss_2 += float(distillation_loss_2)

            if iter % 100 == 0:
                writer.add_images('fw_mask', fw_mask, iter)
                writer.add_images('bw_mask', bw_mask, iter)
                writer.add_images('left_rgb', left_image_1, iter)
                writer.add_images('right_rgb', right_image_1, iter)

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

    scheduler.step()

    writer.add_scalar('epoch/distillation_loss', total_distillation_loss / len(CycleLoader.dataset), epoch)
    writer.add_scalar('epoch/distillation_loss_2', total_distillation_loss_2 / len(CycleLoader.dataset), epoch)

    state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}
    # state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
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
        best_stereo_metric = stereo_err
        str_err = "{:.4f}".format(stereo_err)
        # state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}
        state = {'epoch': epoch, 'state_dict': net.state_dict()}
        torch.save(state, "savemodel/" + args.exp_name + "/best_" + str_err + "_epoch" + str(epoch))
    if flow_err < best_flow_metric:
        best_flow_metric = flow_err
        str_err = "{:.4f}".format(flow_err)
        # state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}
        state = {'epoch': epoch, 'state_dict': net.state_dict()}
        torch.save(state, "savemodel/" + args.exp_name + "/flow_best_" + str_err + "_epoch" + str(epoch))