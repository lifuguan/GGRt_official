# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import imageio
import time
import sys
sys.path.append('../')

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ggrt.base.checkpoint_manager import CheckPointManager
from ggrt.config import config_parser
from ggrt.sample_ray import RaySamplerSingleImage
from ggrt.render_image import render_single_image
from ggrt.model.dbarf import DBARFModel
from utils_loc import *
from ggrt.projection import Projector
from ggrt.data_loaders import get_nearest_pose_ids
from ggrt.data_loaders.llff_data_utils import load_llff_data, batch_parse_llff_poses
from ggrt.geometry.depth import inv2depth



class LLFFRenderDataset(Dataset):
    def __init__(self, args,
                 scenes='fern',  # 'fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'
                 **kwargs):

        self.folder_path = os.path.join(args.rootdir, 'nerf_llff_data/')
        self.num_source_views = args.num_source_views

        print(f"loading {scenes} for rendering")

        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []
        self.h = []
        self.w = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False, factor=4)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)
            h, w = poses[0][:2, -1]
            render_intrinsics, render_c2w_mats = batch_parse_llff_poses(render_poses)

            i_test = [i_test]
            i_val = i_test
            i_train = np.array([i for i in np.arange(len(rgb_files)) if
                                (i not in i_test and i not in i_val)])

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())

            num_render = len(render_intrinsics)
            self.render_rgb_files.extend(np.array(rgb_files)[i_train].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in render_intrinsics])
            self.render_poses.extend([c2w_mat for c2w_mat in render_c2w_mats])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)
            self.h.extend([int(h)]*num_render)
            self.w.extend([int(w)]*num_render)

    def __len__(self):
        return len(self.render_poses)

    def __getitem__(self, idx):
        # rgb_file = self.render_rgb_files[idx]
        # rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        h, w = self.h[idx], self.w[idx]
        camera = np.concatenate(([h, w], intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        id_render = -1
        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                self.num_source_views,
                                                tar_id=id_render,
                                                angular_dist_method='dist')

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

        return {#'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': '',
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                # 'scaled_shape': (0, 0)
                }


def compose_state_dicts(model) -> dict:
    state_dicts = dict()
    
    state_dicts['net_coarse'] = model.net_coarse
    state_dicts['feature_net'] = model.feature_net
    if model.net_fine is not None:
        state_dicts['net_fine'] = model.net_fine
    state_dicts['pose_learner'] = model.pose_learner

    return state_dicts


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    # Create IBRNet model
    model = DBARFModel(args, load_scheduler=False, load_opt=False, pretrained=False)
    state_dicts = compose_state_dicts(model=model)
    ckpt_manager = CheckPointManager()
    print(f'checkpoint path: {args.ckpt_path}')
    start_step = ckpt_manager.load(config=args, models=state_dicts)
    if start_step == 0:
        start_step = int(args.ckpt_path[-10:-4])
    print(f'start_step: {start_step}')

    eval_dataset_name = args.eval_dataset
    extra_out_dir = os.path.join(args.rootdir, args.expname)
    print(f"saving results to {extra_out_dir}...")
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, start_step), 'videos')
    print(f'saving results to {out_scene_dir}')

    os.makedirs(out_scene_dir, exist_ok=True)

    test_dataset = LLFFRenderDataset(args, scenes=args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    out_frames = []
    out_depth_frames = []
    out_color_depth_frames = []
    crop_ratio = 0.075

    for i, data in enumerate(test_loader):
        start = time.time()
        src_rgbs = data['src_rgbs'][0].cpu().numpy()

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()
            # feat_maps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))

            # images = torch.cat([data['rgb'], data['src_rgbs'].squeeze(0)], dim=0).cuda().permute(0, 3, 1, 2)
            # all_feat_maps = model.feature_net(images)

            all_feat_maps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
            # print(f'all_feat_maps shape: {all_feat_maps.shape}')
            feat_maps = (all_feat_maps[0][:, :32, ...], None) if args.coarse_only else \
                        (all_feat_maps[0][:, :32, ...], all_feat_maps[1][:, ...])

            ret = render_single_image(ray_sampler=ray_sampler,
                                      ray_batch=ray_batch,
                                      model=model,
                                      projector=projector,
                                      chunk_size=args.chunk_size,
                                      det=True,
                                      N_samples=args.N_samples,
                                      inv_uniform=args.inv_uniform,
                                      N_importance=args.N_importance,
                                      white_bkgd=args.white_bkgd,
                                      feat_maps=feat_maps
                                      )
            torch.cuda.empty_cache()

        coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
        coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, f'{i}_pred_coarse.png'), coarse_pred_rgb)

        coarse_pred_depth = ret['outputs_coarse']['depth'].detach().cpu()
        imageio.imwrite(os.path.join(out_scene_dir, f'{i}_depth_coarse.png'),
                        (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
        coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                range=tuple(data['depth_range'].squeeze().numpy()))
        imageio.imwrite(os.path.join(out_scene_dir, f'{i}_depth_vis_coarse.png'),
                        (255 * coarse_pred_depth_colored).astype(np.uint8))

        if ret['outputs_fine'] is not None:
            fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
            fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, f'{i}_pred_fine.png'), fine_pred_rgb)
            fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
            imageio.imwrite(os.path.join(out_scene_dir, f'{i}_depth_fine.png'),
                            (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
            fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                  range=tuple(data['depth_range'].squeeze().cpu().numpy()))
            imageio.imwrite(os.path.join(out_scene_dir, f'{i}_depth_vis_fine.png'),
                            (255 * fine_pred_depth_colored).astype(np.uint8))
        else:
            fine_pred_rgb = None

        out_frame = fine_pred_rgb if fine_pred_rgb is not None else coarse_pred_rgb
        h, w = coarse_pred_rgb.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        # crop out image boundaries
        out_frame = out_frame[crop_h:h - crop_h, crop_w:w - crop_w, :]
        out_frames.append(out_frame)

        out_depth_frame = coarse_pred_depth[crop_h:h - crop_h, crop_w:w - crop_w]
        out_depth_frames.append(out_depth_frame)

        out_color_depth_frame = (255 * coarse_pred_depth_colored).astype(np.uint8)[crop_h:h - crop_h, crop_w:w - crop_w, :]
        out_color_depth_frames.append(out_color_depth_frame)

        print(f'frame {i} completed, {time.time() - start}')

    imageio.mimwrite(os.path.join(extra_out_dir, f'{scene_name}_{start_step}.mp4'), out_frames, fps=30, quality=8)
    imageio.mimwrite(os.path.join(extra_out_dir, f'{scene_name}_depth_{start_step}.mp4'), out_depth_frames, fps=30, quality=8)
    imageio.mimwrite(os.path.join(extra_out_dir, f'{scene_name}_color_depth_{start_step}.mp4'), out_color_depth_frames, fps=30, quality=8)
