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


import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
sys.path.append('../')
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids, get_nearby_view_ids
from ..pose_util import PoseInitializer


# only for training
class GoogleScannedDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.args = args
        self.mode = mode
        self.folder_path = os.path.join(args.rootdir, 'google_scanned_objects/')
        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.scene_path_list = glob.glob(os.path.join(self.folder_path, '*'))

        all_rgb_files = []
        all_pose_files = []
        all_intrinsics_files = []
        self.train_poses = []
        self.idx_to_node_id_list = []
        self.node_id_to_idx_list = []
        self.train_view_graphs = []

        num_files = 250
        for i, scene_path in enumerate(self.scene_path_list):
            rgb_files = [os.path.join(scene_path, 'images', f)
                         for f in sorted(os.listdir(os.path.join(scene_path, 'images')))]
            pose_files = [f.replace('images', 'pose').replace('png', 'txt') for f in rgb_files]
            intrinsics_files = [f.replace('images', 'intrinsics').replace('png', 'txt') for f in rgb_files]

            if np.min([len(rgb_files), len(pose_files), len(intrinsics_files)]) \
                    < num_files:
                print(scene_path)
                continue

            all_rgb_files.append(rgb_files)
            all_pose_files.append(pose_files)
            all_intrinsics_files.append(intrinsics_files)
            self.train_poses.append(np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0))

            pose_initializer = PoseInitializer(
                data_path=scene_path,
                image_ids={},
                load_external=True,
                args=args
            )
            self.train_view_graphs.append(pose_initializer.view_graph)

            idx_to_node_id, node_id_to_idx = {}, {}
            for j in range(len(pose_files)):
                idx_to_node_id[j] = j
                node_id_to_idx[j] = j
            self.idx_to_node_id_list.append(idx_to_node_id)
            self.node_id_to_idx_list.append(node_id_to_idx)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_pose_files = np.array(all_pose_files)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files)[index]

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        rgb_files = self.all_rgb_files[idx]
        pose_files = self.all_pose_files[idx]
        # train_poses = self.train_poses[idx]
        intrinsics_files = self.all_intrinsics_files[idx]
        view_graph = self.train_view_graphs[idx]
        idx_to_node_id = self.idx_to_node_id_list[idx]
        node_id_to_idx = self.node_id_to_idx_list[idx]

        id_render = np.random.choice(np.arange(len(rgb_files)))
        train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
        render_pose = train_poses[id_render]
        subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])
        num_select = self.num_source_views + np.random.randint(low=-2, high=3)

        # num_select = self.num_source_views*subsample_factor
        if self.args.selection_rule == 'pose' or self.mode != 'train':
            id_feat_pool = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                num_select=num_select,
                                                tar_id=id_render,
                                                angular_dist_method='vector')
        elif self.args.selection_rule == 'view_graph':
            id_feat_pool = get_nearby_view_ids(target_id=id_render,
                                               graph=view_graph['graph'],
                                               idx_to_node_id=idx_to_node_id,
                                               node_id_to_idx=node_id_to_idx,
                                               num_select=num_select)
        else:
            raise NotImplementedError
        
        id_feat = np.random.choice(id_feat_pool, min(self.num_source_views, len(id_feat_pool)), replace=False)

        assert id_render not in id_feat
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]):
            id_feat[np.random.choice(len(id_feat))] = id_render

        rgb = imageio.imread(rgb_files[id_render]).astype(np.float32) / 255.

        intrinsics = np.loadtxt(intrinsics_files[id_render])
        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics, render_pose.flatten())).astype(np.float32)

        # get depth range
        min_ratio = 0.1
        origin_depth = np.linalg.inv(render_pose)[2, 3]
        max_radius = 0.5 * np.sqrt(2) * 1.1
        near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        far_depth = origin_depth + max_radius
        depth_range = torch.tensor([near_depth, far_depth])

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.
            pose = np.loadtxt(pose_files[id])
            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            intrinsics = np.loadtxt(intrinsics_files[id])
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics, pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return {'rgb': torch.from_numpy(rgb),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_files[id_render],
                'src_rgbs': torch.from_numpy(src_rgbs),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range
                }

