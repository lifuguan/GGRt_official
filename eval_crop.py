from copy import copy
from logging import root
from math import ceil
import sys
from tkinter import CENTER
import hydra
from omegaconf import DictConfig
import copy

# sys.path.append('./')
# sys.path.append('../')

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
from concat import concat
from compare import compare
mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)


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



    return batch
@hydra.main(
    version_base=None,
    config_path="./configs",
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
    projector = Projector(device="cuda:0")

    for i, data in enumerate(test_loader):
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        src_rgbs = data['src_rgbs'][0].cpu().numpy()

        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(file_id)),
                        averaged_img)

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()

            images = torch.cat([data['rgb'], data['src_rgbs'].squeeze(0)], dim=0).cuda().permute(0, 3, 1, 2)
            all_feat_maps = model.feature_net(images)
            
            feat_maps = (all_feat_maps[0][1:, :32, ...], None) if args.coarse_only else \
                        (all_feat_maps[0][1:, :32, ...], all_feat_maps[1][1:, ...])

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
            
            # if True:
            #     num_views = data['src_cameras'].shape[1]
            #     target_pose = data['camera'][0,-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1).to("cuda:0")
            #     context_poses = projector.get_train_poses(target_pose, pred_rel_poses)
            #     data['context']['extrinsics'] = context_poses.unsqueeze(0)
            root_1='/home/gyy/Downloads/dgaussian'
            w=504
            h=378
            crop_h=160
            crop_w=224
            row=ceil(h/crop_h)
            col=ceil(w/crop_w)
            gt_full=data['target']['image'].squeeze(0).squeeze(0)
            for i in range(row):
                for j in range(col):
                    if i==row-1 and j==col-1:
                        data_crop=random_crop(data,size=[crop_h,crop_w],center=(int(h-crop_h//2),int(w-crop_w//2)))
                    elif i==row-1:#最后一行
                        data_crop=random_crop(data,size=[crop_h,crop_w],center=(int(h-crop_h//2),int(crop_w//2+j*crop_w)))
                    elif j==col-1:#z最后一列
                        data_crop=random_crop(data,size=[crop_h,crop_w],center=(int(crop_h//2+i*crop_h),int(w-crop_w//2)))
                    else:
                        data_crop=random_crop(data,size=[crop_h,crop_w],center=(int(crop_h//2+i*crop_h),int(crop_w//2+j*crop_w)))
                        

                    batch_ = data_shim(data_crop, device="cuda:0")

                    batch = gaussian_model.data_shim(batch_)       
                    output, gt_rgb = gaussian_model(batch, i)

                   
                    coarse_pred_rgb = output['rgb'].detach().cpu()[0][0].permute(1, 2, 0)


                    coarse_pred_rgb_np = torch.from_numpy(np.clip(coarse_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)).cuda()
            # gt_rgb_np = torch.from_numpy(gt_rgb.numpy()[None, ...]).cuda()
            # coarse_psnr = img2psnr(gt_rgb_np, coarse_pred_rgb_np)

            # saving outputs ...
                    coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                    imageio.imwrite(f'{root_1}/pic_{file_id[-3:]}/{i}{j}.png', coarse_pred_rgb)

            gt_rgb = gt_full.detach().cpu().permute(1, 2, 0)
            gt_rgb_np_uint8 = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(f'{root_1}/pic_{file_id[-3:]}/gt.png', gt_rgb_np_uint8)
            concat(root_1,file_id,crop_h,crop_w)
            # sum_coarse_psnr += coarse_psnr
            # running_mean_coarse_psnr = sum_coarse_psnr / (i + 1)
    compare(root_1)


if __name__ == '__main__':
    eval()