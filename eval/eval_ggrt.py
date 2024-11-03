import sys
import hydra
from omegaconf import DictConfig

# sys.path.append('./')
# sys.path.append('../')
import visdom
import imageio
import lpips

from torch.utils.data import DataLoader

from ggrt.base.checkpoint_manager import CheckPointManager
from ggrt.config import config_parser
from ggrt.sample_ray import RaySamplerSingleImage
from ggrt.render_image import render_single_image
from ggrt.model.dbarf import DBARFModel
from utils_loc import *
from ggrt.projection import Projector
from ggrt.data_loaders import dataset_dict
from ggrt.loss.ssim_torch import ssim as ssim_torch
from ggrt.geometry.depth import inv2depth
from ggrt.model.pixelsplat.decoder import get_decoder
from ggrt.model.pixelsplat.encoder import get_encoder
from ggrt.model.pixelsplat.pixelsplat import PixelSplat

from ggrt.pose_util import Pose
from ggrt.geometry.align_poses import align_ate_c2b_use_a2b
from ggrt.pose_util import rotation_distance
from eval_dbarf import compose_state_dicts
from ggrt.visualization.pose_visualizer import visualize_cameras
from einops import rearrange, repeat
from matplotlib import cm
from torchvision.utils import save_image
from math import ceil
import copy

mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)






@torch.no_grad()
def get_predicted_training_poses(pred_poses):
    target_pose = torch.eye(4, device=pred_poses.device, dtype=torch.float).repeat(1, 1, 1)

    # World->camera poses.
    pred_poses = Pose.from_vec(pred_poses) # [n_views, 4, 4]
    pred_poses = torch.cat([target_pose, pred_poses], dim=0)

    # Convert camera poses to camera->world.
    pred_poses = pred_poses.inverse()

    return pred_poses


@torch.no_grad()
def align_predicted_training_poses(pred_poses, data, device='cpu'):
    target_pose_gt = data['camera'][..., -16:].reshape(1, 4, 4)
    src_poses_gt = data['src_cameras'][..., -16:].reshape(-1, 4, 4)
    poses_gt = torch.cat([target_pose_gt, src_poses_gt], dim=0).to(device).float()
    
    pred_poses = get_predicted_training_poses(pred_poses)

    aligned_pred_poses = align_ate_c2b_use_a2b(pred_poses, poses_gt)

    return aligned_pred_poses, poses_gt


@torch.no_grad()
def evaluate_camera_alignment(aligned_pred_poses, poses_gt):
    # measure errors in rotation and translation
    R_aligned, t_aligned = aligned_pred_poses.split([3, 1], dim=-1)
    R_gt, t_gt = poses_gt.split([3, 1], dim=-1)
    
    R_error = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
    t_error = (t_aligned - t_gt)[..., 0].norm(dim=-1)
    
    return R_error, t_error

def random_crop(data,size=[160,224] ,center=None):
    _,_,_,h, w = data['context']['image'].shape
    # size=torch.from_numpy(size)
    batch=copy.deepcopy(data)
    out_h, out_w = size[0], size[1]

    if center is not None:
        center_h, center_w = center
    else:
        center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
        center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)
    batch['context']['image'] = batch['context']['image'][:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2]
    batch['target']['image'] = batch['target']['image'][:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2]

    batch['context']['intrinsics'][:,:,0,0]=batch['context']['intrinsics'][:,:,0,0]*w/out_w
    batch['context']['intrinsics'][:,:,1,1]=batch['context']['intrinsics'][:,:,1,1]*h/out_h
    batch['context']['intrinsics'][:,:,0,2]=(batch['context']['intrinsics'][:,:,0,2]*w-center_w+out_w // 2)/out_w
    batch['context']['intrinsics'][:,:,1,2]=(batch['context']['intrinsics'][:,:,1,2]*h-center_h+out_h // 2)/out_h

    batch['target']['intrinsics'][:,:,0,0]=batch['target']['intrinsics'][:,:,0,0]*w/out_w
    batch['target']['intrinsics'][:,:,1,1]=batch['target']['intrinsics'][:,:,1,1]*h/out_h
    batch['target']['intrinsics'][:,:,0,2]=(batch['target']['intrinsics'][:,:,0,2]*w-center_w+out_w // 2)/out_w
    batch['target']['intrinsics'][:,:,1,2]=(batch['target']['intrinsics'][:,:,1,2]*h-center_h+out_h // 2)/out_h
    return batch,center_h,center_w











def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def img2mse(x, y, mask=None):
    '''
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    '''

    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())


def img2ssim(gt_image, pred_image):
    """
    Args:
        gt_image: [B, 3, H, W]
        pred_image: [B, 3, H, W]
    """
    return ssim_torch(gt_image, pred_image).item()


def img2lpips(lpips_loss, gt_image, pred_image):
    return lpips_loss(gt_image * 2 - 1, pred_image * 2 - 1).item()


def compose_state_dicts(model) -> dict:
    state_dicts = dict()
    
    state_dicts['net_coarse'] = model.net_coarse
    state_dicts['feature_net'] = model.feature_net
    if model.net_fine is not None:
        state_dicts['net_fine'] = model.net_fine
    state_dicts['pose_learner'] = model.pose_learner

    return state_dicts



def apply_color_map(
    x,
    color_map,
):
    cmap = cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)
    
def apply_color_map_to_image(
    image,
    color_map ,
) :
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")
def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="pretrain_dgaussian",
)
def eval(cfg_dict: DictConfig):
    args = cfg_dict
    args.distributed = False
    # args.pixelsplat['encoder']['epipolar_transformer']['num_context_views'] = args.num_source_views
    
    
    # Create IBRNet model
    model = DBARFModel(args, load_scheduler=False, load_opt=False, pretrained=False)
    state_dicts = compose_state_dicts(model=model)
    ckpt_manager = CheckPointManager()
    start_step = ckpt_manager.load(config=args, models=state_dicts)
    print(f'start_step: {start_step}')
    
    encoder, encoder_visualizer = get_encoder(args.pixelsplat.encoder)
    decoder = get_decoder(args.pixelsplat.decoder)
    gaussian_model = PixelSplat(encoder, decoder, encoder_visualizer)
    gaussian_model.load_state_dict(torch.load(args.ckpt_path)['gaussian'])
    # gaussian_model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
    gaussian_model.cuda()
    
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(args.rootdir, args.expname)
    print("saving results to {}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)


    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, start_step))
    os.makedirs(out_scene_dir, exist_ok=True)

    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    results_dict = {scene_name: {}}
    sum_coarse_psnr = 0
    sum_fine_psnr = 0
    running_mean_coarse_psnr = 0
    running_mean_fine_psnr = 0
    sum_coarse_lpips = 0
    sum_fine_lpips = 0
    running_mean_coarse_lpips = 0
    running_mean_fine_lpips = 0
    sum_coarse_ssim = 0
    sum_fine_ssim = 0
    running_mean_coarse_ssim = 0
    running_mean_fine_ssim = 0

    lpips_loss = lpips.LPIPS(net="alex").cuda()
    R_errors = []
    t_errors = []
    video_rgb_pred = []
    video_depth_pred = []
    visdom_ins = visdom.Visdom(server='localhost', port=8097, env='splatam')
    for i, data in enumerate(test_loader):
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        src_rgbs = data['src_rgbs'][0].cpu().numpy()

        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)

        model.switch_to_eval()
        with torch.no_grad():

            # if args.render_video is not True:
            pred_inv_depth, pred_rel_poses, _, _ = model.correct_poses(
                fmaps=None,
                target_image=data['rgb'].cuda(),
                ref_imgs=data['src_rgbs'].cuda(),
                target_camera=data['camera'],
                ref_cameras=data['src_cameras'],
                min_depth=data['depth_range'][0][0],
                max_depth=data['depth_range'][0][1],
                scaled_shape=data['scaled_shape'])
            pred_inv_depth = pred_inv_depth.squeeze(0).squeeze(0).detach().cpu()
            pred_depth = inv2depth(pred_inv_depth)
            pred_rel_poses = pred_rel_poses.detach().cpu()
            aligned_pred_poses, poses_gt = align_predicted_training_poses(pred_rel_poses, data)
            pose_error = evaluate_camera_alignment(aligned_pred_poses, poses_gt)
            visualize_cameras(visdom_ins, step=i, poses=[aligned_pred_poses, poses_gt], cam_depth=0.1)
            
            R_errors.append(pose_error[0])
            t_errors.append(pose_error[1])

            batch = data_shim(data, device="cuda:0")
            batch = gaussian_model.data_shim(batch)       
            # out_h=640
            # out_w=960
            # _, _, _, h, w = batch["target"]["image"].shape
            # gt_rgb_full = batch["target"]["image"]
            # row=ceil(h/out_h)
            # col=ceil(w/out_w)
            # crop_list = []
            # for i in range(row):
            #     for j in range(col):
            #         if i==row-1 and j==col-1:
            #             data_crop,center_h,center_w=random_crop( batch,size=[out_h,out_w],center=(int(h-out_h//2),int(w-out_w//2)))
            #         elif i==row-1:#最后一行
            #             data_crop,center_h,center_w=random_crop( batch,size=[out_h,out_w],center=(int(h-out_h//2),int(out_w//2+j*out_w)))
            #         elif j==col-1:#z最后一列
            #             data_crop,center_h,center_w=random_crop( batch,size=[out_h,out_w],center=(int(out_h//2+i*out_h),int(w-out_w//2)))
            #         else:
            #             data_crop,center_h,center_w=random_crop(batch,size=[out_h,out_w],center=(int(out_h//2+i*out_h),int(out_w//2+j*out_w)))  
            #         output, gt_rgb = gaussian_model(data_crop, i)
            #         crop_list.append(output['rgb'].detach().cpu()[0][0].permute(1, 2, 0))
            #         coarse_pred_rgb = output['rgb'].detach().cpu()[0][0].permute(1, 2, 0)
            #         coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            #         imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_coarse.png'.format(i*2+j)), coarse_pred_rgb)
            # full_rgb = torch.cat([torch.cat([crop_list[0],crop_list[1]], dim=1),torch.cat([crop_list[2],crop_list[3]], dim=1)],dim = 0)
            # full_pred_rgb = (255 * np.clip(full_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            # imageio.imwrite(os.path.join(out_scene_dir, 'full_pred_coarse.png'), full_pred_rgb)
            # gt_rgb = gt_rgb_full.detach().cpu()[0][0].permute(1, 2, 0)
            # gt_rgb_np = torch.from_numpy(gt_rgb.numpy()[None, ...]).cuda()
            # full_pred_rgb_np = torch.from_numpy(np.clip(full_rgb.numpy()[None, ...], a_min=0., a_max=1.)).cuda()
            # coarse_psnr = img2psnr(gt_rgb_np, full_pred_rgb_np)


            output, gt_rgb = gaussian_model(batch, i)
            # depth = depth_map(output['depth'][0][0])
            # depth = depth.detach().cpu().permute(1, 2, 0)
            depth = depth_map(output['depth'][0])
            depth = depth.detach().cpu().permute(0,2, 3, 1)
            gt_rgb = gt_rgb['rgb'].detach().cpu()[0][0].permute(1, 2, 0)
            coarse_pred_rgb = output['rgb'].detach().cpu()[0][0].permute(1, 2, 0)
            # coarse_err_map = torch.sum((coarse_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
            # coarse_err_map_colored = (colorize_np(coarse_err_map, range=(0., 1.)) * 255).astype(np.uint8)

            coarse_pred_rgb_np = torch.from_numpy(np.clip(coarse_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)).cuda()
            gt_rgb_np = torch.from_numpy(gt_rgb.numpy()[None, ...]).cuda()

            coarse_psnr = img2psnr(gt_rgb_np, coarse_pred_rgb_np)
            coarse_lpips = img2lpips(lpips_loss, gt_rgb_np.permute(0, 3, 1, 2), coarse_pred_rgb_np.permute(0, 3, 1, 2))
            coarse_ssim = img2ssim(gt_rgb_np.permute(0, 3, 1, 2), coarse_pred_rgb_np.permute(0, 3, 1, 2))

            coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_coarse.png'.format(file_id)), coarse_pred_rgb)

            gt_rgb_np_uint8 = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_gt_rgb.png'.format(file_id)), gt_rgb_np_uint8)
            
            if args.render_video is True:
                sum_coarse_psnr += coarse_psnr
                running_mean_coarse_psnr = sum_coarse_psnr / (i + 1)
                sum_coarse_lpips += coarse_lpips
                running_mean_coarse_lpips = sum_coarse_lpips / (i + 1)
                sum_coarse_ssim += coarse_ssim
                running_mean_coarse_ssim = sum_coarse_ssim / (i + 1)

                fine_ssim = fine_lpips = fine_psnr = 0.

                sum_fine_psnr += fine_psnr
                running_mean_fine_psnr = sum_fine_psnr / (i + 1)
                sum_fine_lpips += fine_lpips
                running_mean_fine_lpips = sum_fine_lpips / (i + 1)
                sum_fine_ssim += fine_ssim
                running_mean_fine_ssim = sum_fine_ssim / (i + 1)

                print("==================\n"
                    "{}, curr_id: {} \n"
                    "current coarse psnr: {:03f}, current fine psnr: {:03f} \n"
                    "running mean coarse psnr: {:03f}, running mean fine psnr: {:03f} \n"
                    "current coarse ssim: {:03f}, current fine ssim: {:03f} \n"
                    "running mean coarse ssim: {:03f}, running mean fine ssim: {:03f} \n" 
                    "current coarse lpips: {:03f}, current fine lpips: {:03f} \n"
                    "running mean coarse lpips: {:03f}, running mean fine lpips: {:03f} \n"
                    # "R Error {:.03f}, t Error {:.03f} \n"
                    "===================\n"
                    .format(scene_name, file_id,
                            coarse_psnr, fine_psnr,
                            running_mean_coarse_psnr, running_mean_fine_psnr,
                            coarse_ssim, fine_ssim,
                            running_mean_coarse_ssim, running_mean_fine_ssim,
                            coarse_lpips, fine_lpips,
                            running_mean_coarse_lpips, running_mean_fine_lpips,
                            # pose_error[0], pose_error[1]
                            ))
                results_dict[scene_name][file_id] = {'coarse_psnr': coarse_psnr,
                                                    'fine_psnr': fine_psnr,
                                                    'coarse_ssim': coarse_ssim,
                                                    'fine_ssim': fine_ssim,
                                                    'coarse_lpips': coarse_lpips,
                                                    'fine_lpips': fine_lpips,
                                                    }
                if i>2:      
                    video_rgb_pred.append(coarse_pred_rgb)
                    video_depth_pred.append(depth)
                # v,_,_,_ = depth.shape
                # for i in range(v): 
                #     video_depth_pred.append(depth[i])
                # imageio.mimwrite(os.path.join(out_scene_dir, 'video_depth_ref.mp4'), video_depth_pred, fps=10, quality=8)
                
            # saving outputs ...
            imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(file_id)),averaged_img)

            coarse_pred_depth = output['depth'].detach().cpu()[0][0]
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_coarse.png'.format(file_id)),
                            (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
            coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                    range=tuple(data['depth_range'].squeeze().cpu().numpy()))


            save_image(depth[0].permute(2,0,1), os.path.join(out_scene_dir, '{}_depth_vis_coarse.png'.format(file_id)))

            imageio.imwrite(os.path.join(out_scene_dir, f'{file_id}_pose_optimizer_gray_depth.png'),
                            (pred_depth.numpy() * 255.).astype(np.uint8))
            coarse_pred_depth = colorize(coarse_pred_depth, cmap_name='jet', append_cbar=True)
            imageio.imwrite(os.path.join(out_scene_dir, f'{file_id}_dgassin_color_depth.png'),
                            (coarse_pred_depth.numpy() * 255.).astype(np.uint8))

            pred_depth = colorize_1(pred_depth, cmap_name='jet', append_cbar=True)
            imageio.imwrite(os.path.join(out_scene_dir, f'{file_id}_pose_optimizer_color_depth.png'),
                            (pred_depth.numpy() * 255.).astype(np.uint8))
            # imageio.imwrite(os.path.join(out_scene_dir, '{}_err_map_coarse.png'.format(file_id)),
            #                 coarse_err_map_colored)
            
            sum_coarse_psnr += coarse_psnr
            running_mean_coarse_psnr = sum_coarse_psnr / (i + 1)
            sum_coarse_lpips += coarse_lpips
            running_mean_coarse_lpips = sum_coarse_lpips / (i + 1)
            sum_coarse_ssim += coarse_ssim
            running_mean_coarse_ssim = sum_coarse_ssim / (i + 1)

            fine_ssim = fine_lpips = fine_psnr = 0.

            sum_fine_psnr += fine_psnr
            running_mean_fine_psnr = sum_fine_psnr / (i + 1)
            sum_fine_lpips += fine_lpips
            running_mean_fine_lpips = sum_fine_lpips / (i + 1)
            sum_fine_ssim += fine_ssim
            running_mean_fine_ssim = sum_fine_ssim / (i + 1)

            print("==================\n"
                  "{}, curr_id: {} \n"
                  "current coarse psnr: {:03f}, current fine psnr: {:03f} \n"
                  "running mean coarse psnr: {:03f}, running mean fine psnr: {:03f} \n"
                  "current coarse ssim: {:03f}, current fine ssim: {:03f} \n"
                  "running mean coarse ssim: {:03f}, running mean fine ssim: {:03f} \n" 
                  "current coarse lpips: {:03f}, current fine lpips: {:03f} \n"
                  "running mean coarse lpips: {:03f}, running mean fine lpips: {:03f} \n"
                  "===================\n"
                  .format(scene_name, file_id,
                          coarse_psnr, fine_psnr,
                          running_mean_coarse_psnr, running_mean_fine_psnr,
                          coarse_ssim, fine_ssim,
                          running_mean_coarse_ssim, running_mean_fine_ssim,
                          coarse_lpips, fine_lpips,
                          running_mean_coarse_lpips, running_mean_fine_lpips
                          ))

            results_dict[scene_name][file_id] = {'coarse_psnr': coarse_psnr,
                                                 'fine_psnr': fine_psnr,
                                                 'coarse_ssim': coarse_ssim,
                                                 'fine_ssim': fine_ssim,
                                                 'coarse_lpips': coarse_lpips,
                                                 'fine_lpips': fine_lpips,
                                                 }
    if args.render_video is True:
        imageio.mimwrite(os.path.join(out_scene_dir, 'video_rgb_pred.mp4'), video_rgb_pred, fps=10, quality=8)
        imageio.mimwrite(os.path.join(out_scene_dir, 'video_depth_pred.mp4'), video_depth_pred, fps=10, quality=8)
    
    R_errors = np.concatenate(R_errors, axis=0)
    t_errors = np.concatenate(t_errors, axis=0)

    mean_rotation_error = np.rad2deg(R_errors.mean())
    mean_position_error = t_errors.mean()
    med_rotation_error = np.rad2deg(np.median(R_errors))
    med_position_error = np.median(t_errors)

    metrics_dict = {'R_error_mean': str(mean_rotation_error),
                    't_error_mean': str(mean_position_error),
                    'R_error_med': str(med_rotation_error),
                    't_error_med': str(med_position_error)
                    }
    print("Rel Pose Error: ", metrics_dict)
    mean_coarse_psnr = sum_coarse_psnr / total_num
    mean_fine_psnr = sum_fine_psnr / total_num
    mean_coarse_lpips = sum_coarse_lpips / total_num
    mean_fine_lpips = sum_fine_lpips / total_num
    mean_coarse_ssim = sum_coarse_ssim / total_num
    mean_fine_ssim = sum_fine_ssim / total_num

    print('------{}-------\n'
          'final coarse psnr: {}, final fine psnr: {}\n'
          'fine coarse ssim: {}, final fine ssim: {} \n'
          'final coarse lpips: {}, fine fine lpips: {} \n'
          .format(scene_name, mean_coarse_psnr, mean_fine_psnr,
                  mean_coarse_ssim, mean_fine_ssim,
                  mean_coarse_lpips, mean_fine_lpips,
                  ))

    results_dict[scene_name]['coarse_mean_psnr'] = mean_coarse_psnr
    results_dict[scene_name]['fine_mean_psnr'] = mean_fine_psnr
    results_dict[scene_name]['coarse_mean_ssim'] = mean_coarse_ssim
    results_dict[scene_name]['fine_mean_ssim'] = mean_fine_ssim
    results_dict[scene_name]['coarse_mean_lpips'] = mean_coarse_lpips
    results_dict[scene_name]['fine_mean_lpips'] = mean_fine_lpips

    f = open("{}/psnr_{}_{}.txt".format(extra_out_dir, save_prefix, start_step), "w")
    f.write(str(results_dict))
    f.close()


if __name__ == '__main__':
    eval()