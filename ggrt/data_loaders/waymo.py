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

import enum
import os
import numpy as np
import imageio
import torch
import sys
import cv2
sys.path.append('../')
from torch.utils.data import Dataset
from .data_utils import get_nearby_view_ids, random_crop, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses
from ..pose_util import PoseInitializer

import random
from .base_utils import downsample_gaussian_blur
import math
def loader_resize(rgb, camera, src_rgbs, src_cameras, size=(400, 600)):
    h, w = rgb.shape[:2]
    out_h, out_w = size[0], size[1]
    intrinsics = camera[2:18].reshape(4, 4)
    src_intrinsics = src_cameras[:, 2:18].reshape(-1, 4, 4)
    if out_w >= w or out_h >= h:
        return rgb, camera, src_rgbs, src_cameras, intrinsics[..., :3, :3], src_intrinsics[..., :3, :3]

    ratio_y = out_h / h
    ratio_x = out_w / w
    camera[0] = out_h
    camera[1] = out_w
    camera[2:18] = intrinsics.flatten()
    src_cameras[:, 0] = out_h
    src_cameras[:, 1] = out_w
    src_cameras[:, 2:18] = src_intrinsics.reshape(-1, 16)
    rgb = cv2.resize(downsample_gaussian_blur(
                rgb, ratio_y), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    src_rgbs = [cv2.resize(downsample_gaussian_blur(
                src_rgb, ratio_y), (out_w, out_h), interpolation=cv2.INTER_LINEAR) for src_rgb in src_rgbs]
    src_rgbs = np.stack(src_rgbs, axis=0)
    return rgb, camera, src_rgbs, src_cameras, intrinsics[..., :3, :3], src_intrinsics[..., :3, :3]

class WaymoStaticDataset(Dataset):
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        if mode == 'test' or len(scenes) > 0:
            if scenes[0] == '016':
                self.folder_path = os.path.join('data/waymo_dynamic/waymo/processed/training') # for dynamic dataset
            else:
                self.folder_path = args.dataset_root_eval
        else:
            self.folder_path = args.dataset_root_train
        self.dataset_name = 'waymo' 
        self.pose_noise_level = 0 

        self.args = args
        self.mode = mode # if args.render_video is not True else "train"  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        self.idx_to_node_id_list = []
        self.node_id_to_idx_list = []
        self.train_view_graphs = []

        if mode == 'train':
            self.image_size = (352, 480)
        else:
            self.image_size = (640, 960)     
        all_scenes = os.listdir(self.folder_path)
    
        ############     Wamyo Parameters     ############
        self.num_cams, self.camera_list =1, [0]
        
        self.start_timestep, self.end_timestep = 0, 197
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        print(f'[INFO] num scenes: {len(scenes)}')
        
        for i, scene in enumerate(scenes):
            if scene == 126 :
                scene = '126'
            scene_path = os.path.join(self.folder_path, scene)
            
            rgb_files = []
            i_test,count = [], 0
            for t in range(self.start_timestep, self.end_timestep):
                for cam_idx in self.camera_list:
                    if cam_idx == 0:
                        i_test.append(count)
                    rgb_files.append(os.path.join(scene_path, "images", f"{t:03d}_{cam_idx}.jpg"))
                    count += 1
            
            intrinsics, c2w_mats = self.load_calibrations(scene_path)
            
            
            near_depth = 0.1
            far_depth = 100
            
            i_test = i_test[::8] if mode != 'eval_pose' else []
            i_train = np.array([j for j in np.arange(len(rgb_files)) if
                                (j not in i_test and j not in i_test)])
            
            idx_to_node_id, node_id_to_idx = {}, {}
            for j in range(i_train.shape[0]):
                idx_to_node_id[j] = i_train[j]
                node_id_to_idx[i_train[j]] = j
            self.idx_to_node_id_list.append(idx_to_node_id)
            self.node_id_to_idx_list.append(node_id_to_idx)

            if self.mode == 'train' or self.mode == 'eval_pose':
                i_render = i_train
            else:
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend(intrinsics[i_train])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)


    def load_calibrations(self, scene_path):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        # to store per-camera intrinsics and extrinsics
        _intrinsics = []
        cam_to_egos = []
        for i in range(self.num_cams):
            # load camera intrinsics
            # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
            # ====!! we did not use distortion parameters for simplicity !!====
            # to be improved!!
            intrinsic = np.loadtxt(
                os.path.join(scene_path, "intrinsics", f"{i}.txt")
            )
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            # scale intrinsics w.r.t. load size
            fx, fy = (
                fx * self.image_size[1] / self.ORIGINAL_SIZE[i][1],
                fy * self.image_size[0] / self.ORIGINAL_SIZE[i][0],
            )
            cx, cy = (
                cx * self.image_size[1] / self.ORIGINAL_SIZE[i][1],
                cy * self.image_size[0] / self.ORIGINAL_SIZE[i][0],
            )
            # intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            intrinsic = np.array([[fx, 0, cx,0], [0, fy, cy,0], [0, 0, 1, 0], [0, 0, 0, 1]])
            _intrinsics.append(intrinsic)

            # load camera extrinsics
            cam_to_ego = np.loadtxt(
                os.path.join(scene_path, "extrinsics", f"{i}.txt")
            )
            # because we use opencv coordinate system to generate camera rays,
            # we need a transformation matrix to covnert rays from opencv coordinate
            # system to waymo coordinate system.
            # opencv coordinate system: x right, y down, z front
            # waymo coordinate system: x front, y left, z up
            cam_to_egos.append(cam_to_ego @ self.OPENCV2DATASET)

        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, cam_ids = [], []
        # ===! for waymo, we simplify timestamps as the time indices
        timestamps, timesteps = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(
            os.path.join(scene_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(
                os.path.join(scene_path, "ego_pose", f"{t:03d}.txt")
            )
            # compute ego_to_world transformation
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
            for cam_id in self.camera_list:
                cam_ids.append(cam_id)
                # transformation:
                #   (opencv_cam -> waymo_cam -> waymo_ego_vehicle) -> current_world
                cam2world = ego_to_world @ cam_to_egos[cam_id]
                cam_to_worlds.append(cam2world)
                intrinsics.append(_intrinsics[cam_id])
                # ===! we use time indices as the timestamp for waymo dataset for simplicity
                # ===! we can use the actual timestamps if needed
                # to be improved
                timestamps.append(t - self.start_timestep)
                timesteps.append(t - self.start_timestep)

        intrinsics = np.stack(intrinsics, axis=0)
        cam_to_worlds = np.stack(cam_to_worlds, axis=0)
        return intrinsics, cam_to_worlds


    def get_data_one_batch(self, idx, nearby_view_id=None):
        self.nearby_view_id = nearby_view_id
        return self.__getitem__(idx=idx)

    def num_poses(self):
        return len(self.render_rgb_files)

    def __len__(self):
        return len(self.render_rgb_files) * 100000 if self.mode == 'train' and self.args.render_video is not True else len(self.render_rgb_files)

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        render_name = rgb_file[-9:-4]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
                
        ##### For translation experiments during test #####
        # translation = np.array([2, 0, 0])
        # render_pose[:, 3] += np.dot(render_pose[:, :3], translation)
        # extrinsic_matrix = render_pose

        ##### For rotation experiments during test #####
        # theta = 30 * math.pi / 180
        # rotation_matrix = torch.tensor([[math.cos(theta), -math.sin(theta), 0, 0],
        #                             [math.sin(theta),  math.cos(theta), 0, 0],
        #                             [0,               0,               1, 0],
        #                             [0,               0,               0, 1]])
        # extrinsic_matrix = torch.from_numpy(extrinsic_matrix)
        # new_extrinsic_matrix = torch.matmul(rotation_matrix, extrinsic_matrix.float())
        # render_pose = new_extrinsic_matrix.numpy()

        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        camera = np.concatenate((list(self.image_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        if self.mode == 'train':
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views 
            # + np.random.randint(low=-2, high=2)
        else:
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = None
        # num_select = min(self.num_source_views*subsample_factor, 28)
        if self.args.selection_rule == 'pose' or self.mode != 'train':
            nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                    train_poses,
                                                    num_select=num_select,
                                                    tar_id=id_render,
                                                    angular_dist_method='dist')
        else:
            raise NotImplementedError
        
        if self.mode == 'eval_pose' and self.nearby_view_id is not None:
            nearest_pose_ids = np.array([self.nearby_view_id])

        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)
        # print(f'nearest pose ids: {nearest_pose_ids}'
        #         f'id_render: {id_render}')

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        # relative_poses = None if self.args.selection_rule == 'pose' else \
        #                  get_relative_poses(idx, view_graph['two_view_geometries'], idx_to_node_id, nearest_pose_ids)

        src_name = []
        src_rgbs = []
        src_cameras = []
        src_intrinsics, src_extrinsics = [], []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            src_name.append(id)
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            
            src_intrinsics.append(train_intrinsics_)
            src_extrinsics.append(train_pose)
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(self.image_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        src_intrinsics, src_extrinsics = np.stack(src_intrinsics, axis=0), np.stack(src_extrinsics, axis=0)
    
        pix_rgb, pix_camera, pix_src_rgbs, pix_src_cameras, pix_intrinsics, pix_src_intrinsics = loader_resize(rgb,camera.copy(),src_rgbs,src_cameras.copy(), size=self.image_size)

        pix_src_extrinsics = torch.from_numpy(src_extrinsics).float()
        pix_extrinsics = torch.from_numpy(render_pose).unsqueeze(0).float()
        
        pix_src_intrinsics = self.normalize_intrinsics(torch.from_numpy(src_intrinsics[:,:3,:3]).float(), self.image_size)
        pix_intrinsics = self.normalize_intrinsics(torch.from_numpy(intrinsics[:3,:3]).unsqueeze(0).float(), self.image_size)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6], dtype=torch.float32)

        # Resize the world to make the baseline 1.
        if pix_src_extrinsics.shape[0] == 2:
            a, b = pix_src_extrinsics[:, :3, 3]
            scale = (a - b).norm()
            if scale < 0.001:
                print(
                    f"Skipped {scene} because of insufficient baseline "
                    f"{scale:.6f}"
                )
            pix_src_extrinsics[:, :3, 3] /= scale
            pix_extrinsics[:, :3, 3] /= scale
        else:
            scale = 1
        
        return {'rgb': torch.from_numpy(pix_rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(pix_src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy( src_cameras),
                'depth_range': depth_range,
                'idx': idx,
                'scaled_shape': (0, 0), # (378, 504)
                "context": {
                        "extrinsics": pix_src_extrinsics,
                        "intrinsics": pix_src_intrinsics,
                        "image": torch.from_numpy(pix_src_rgbs[..., :3]).permute(0, 3, 1, 2),
                        "near":  depth_range[0].repeat(num_select) / scale,
                        "far": depth_range[1].repeat(num_select) / scale,
                        "index": torch.tensor([int(i) for i in src_name]),
                },
                "target": {
                        "extrinsics": pix_extrinsics,
                        "intrinsics": pix_intrinsics,
                        "image": torch.from_numpy(pix_rgb[..., :3]).unsqueeze(0).permute(0, 3, 1, 2),
                        "near": depth_range[0].unsqueeze(0) / scale,
                        "far": depth_range[1].unsqueeze(0) / scale,
                        "index": id_render,
                },
                }
    def normalize_intrinsics(self, intrinsics, img_size):
        h, w = img_size
        # 归一化内参矩阵
        intrinsics_normalized = intrinsics.clone()
        intrinsics_normalized[:, 0, 0] /= w
        intrinsics_normalized[:, 1, 1] /= h
        intrinsics_normalized[:, 0, 2] = 0.5
        intrinsics_normalized[:, 1, 2] = 0.5
        return intrinsics_normalized