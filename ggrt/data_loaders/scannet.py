import copy
import re
import os
import numpy as np
import cv2
import imageio
import torch
import torchvision.transforms as transforms

from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from .data_utils import random_crop, random_flip

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def resize_image(image, shape):
    """
    Resizes input image.

    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode

    Returns
    -------
    image : Image.PIL
        Resized image
    """
    # resized_image = Image.fromarray(image).resize(shape)
    # return np.asarray(resized_image).astype(np.float32) / 255.
    transform_image = transforms.Resize(shape)
    resized_image = transform_image(image)
    transform = transforms.ToTensor()
    resized_image = transform(resized_image).type(torch.FloatTensor)
    return resized_image

def dummy_calibration(image):
    w, h = [float(d) for d in image.size]
    return np.array([[1000. , 0.    , w / 2. - 0.5],
                     [0.    , 1000. , h / 2. - 0.5],
                     [0.    , 0.    , 1.          ]])

def get_idx(filename):
    return int(re.search(r'\d+', filename).group())

def read_files(directory, ext=('.png', '.jpg', '.jpeg', '.ppm'), skip_empty=True):
    files = defaultdict(list)
    for entry in os.scandir(directory):
        relpath = os.path.relpath(entry.path, directory)
        if entry.is_dir():
            color_path = os.path.join(entry.path, 'color')
            d_files = read_files(color_path, ext=ext, skip_empty=skip_empty)
            if skip_empty and not len(d_files):
                continue
            files[relpath + '/color'] = d_files[color_path]
        elif entry.is_file():
            if ext is None or entry.path.lower().endswith(tuple(ext)):
                pose_path = entry.path.replace('color', 'pose').replace('.jpg', '.txt')
                pose = np.genfromtxt(pose_path)
                if not np.isinf(pose).any():
                    files[directory].append(relpath)
    return files

def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(Image.open(file), dtype=int)

    depth = depth_png.astype(np.float) / 1000.

    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def read_image(filename):
    return Image.open(filename) # np.array(Image.open(filename))
    # return imageio.imread(filename) # .astype(np.float32) / 255.

########################################################################################################################
#### DATASET
########################################################################################################################

class ScannetDataset(Dataset):
    def __init__(self, args, mode='train', scenes=(),
                 forward_context=2, back_context=2, strides=(1,),
                 depth_type=None, **kwargs):
        super().__init__()
        self.mode = mode

        assert len(strides) == 1 and strides[0] == 1, \
            'ImageDataset currently only supports stride of 1.'

        self.min_depth = 0.2
        self.max_depth = 10.0

        self.depth_type = depth_type
        self.with_depth = depth_type != '' and depth_type is not None
        self.root_dir = os.path.join(args.rootdir)
        print(f'root_dir: {self.root_dir}')
        self.split = 'splits/test_all_list.txt' #'splits/train_all_list.txt' if mode == 'train' else 'splits/test_split.txt'

        self.backward_context = back_context
        self.forward_context = forward_context
        self.strides = strides[0]
        self.transform = transforms.Compose([transforms.PILToTensor()])

        self.files = []

        # =================== load from txt ====================
        scenes = list(scenes)
        load_specific_scenes = len(scenes) > 0
        print(f'load specific scenes: {scenes}')
        self.file_tree = defaultdict(list)
        with open(os.path.join(self.root_dir, self.split), "r") as f:
            split_data = f.readlines()
        
        self.subdir = 'data'
        # self.subdir = 'train' if self.mode == 'train' else 'test'
        for data in split_data:
            scene, filename = data.split()
            scene_path = os.path.join(self.root_dir, self.subdir, scene)

            if load_specific_scenes and scene[:-6] in scenes and os.path.exists(scene_path):
                self.file_tree[scene].append(filename)
            elif load_specific_scenes != True and os.path.exists(scene_path):
                self.file_tree[scene].append(filename)

        print(f'[INFO] Loaded {len(self.file_tree)} scenes for {self.mode}.')

        # downsample by 5
        keys = self.file_tree.keys()
        for k in keys: # self.file_tree:
            print(f'num images: {len(self.file_tree[k])}')
            i_train = [i for i in range(0, len(self.file_tree[k]), 5)]
            i_test = [i for i in range(len(self.file_tree[k])) if i not in i_train][::4][:10]
            print(f'i_train: {len(i_train)}')
            print(f'i_test: {len(i_test)}')
            candidate_files = []
            selected_idxs = i_train if self.mode == 'train' else i_test
            for idx in selected_idxs:
                candidate_files.append(self.file_tree[k][idx])
            
            self.file_tree[k] = candidate_files
            # self.file_tree[k] = self.file_tree[k][::5]
            print(f'[INFO] {len(self.file_tree[k])} images are selected for {self.mode} in scene {k}.')

        for k, v in self.file_tree.items():
            file_list = v
            files = [fname for fname in file_list if self._has_context(k, fname, file_list)]
            self.files.extend([[k, fname] for fname in files])

    def num_poses(self):
        return len(self.files)

    def __len__(self):
        return len(self.files)

    def _change_idx(self, idx, filename):
        _, ext = os.path.splitext(os.path.basename(filename))
        return str(idx) + ext

    def _has_context(self, session, filename, file_list):
        context_paths = self._get_context_file_paths(filename, file_list)
        return all([f in file_list for f in context_paths])

    def _get_context_file_paths(self, filename, filelist):
        # fidx = get_idx(filename)
        fidx = filelist.index(filename)
        idxs = list(np.arange(-self.backward_context * self.strides, 0, self.strides)) + \
               list(np.arange(0, self.forward_context * self.strides, self.strides) + self.strides)
        return [filelist[fidx+i] if 0 <= fidx+i < len(filelist) else 'none' for i in idxs]

    def _read_rgb_context_files(self, session, filename):
        context_paths = self._get_context_file_paths(filename, self.file_tree[session])
        
        return [read_image(os.path.join(self.root_dir, self.subdir, session, filename))
                for filename in context_paths]

    def _read_rgb_file(self, session, filename):
        # print(f'session: {session}')
        return read_image(os.path.join(self.root_dir, self.subdir, session, filename))

########################################################################################################################
#### DEPTH
########################################################################################################################

    def _read_depth(self, depth_file):
        """Get the depth map from a file."""
        if self.depth_type in ['velodyne']:
            return read_npz_depth(depth_file, self.depth_type)
        elif self.depth_type in ['groundtruth']:
            return read_png_depth(depth_file)
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))

    def _get_depth_file(self, image_file):
        """Get the corresponding depth file from an image file."""
        depth_file = image_file.replace('color', 'depth').replace('image', 'depth')
        depth_file = depth_file.replace('jpg', 'png')
        return depth_file

    def get_data_one_batch(self, idx, nearby_view_id=None):
        self.nearby_view_id = nearby_view_id
        return self.__getitem__(idx=idx)

    def __getitem__(self, idx):
        session, filename = self.files[idx]
        rgb = self._read_rgb_file(session, filename)
        raw_W, raw_H = rgb.size
        img_size = (320, 512) # h, w
        rgb = resize_image(rgb, img_size).permute(1, 2, 0)

        if self.with_depth:
            depth = self._read_depth(self._get_depth_file(os.path.join(self.root_dir, self.subdir, session, filename)))
            resized_depth = cv2.resize(depth, rgb.size, interpolation = cv2.INTER_NEAREST)

        intr_path = os.path.join(self.root_dir, self.subdir, session, filename).split('color')[0] + 'intrinsic/intrinsic_color.txt'
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = np.genfromtxt(intr_path)[:3, :3]
        intrinsics[0] *= img_size[0] / raw_W
        intrinsics[1] *= img_size[1] / raw_H

        context_paths = self._get_context_file_paths(filename, self.file_tree[session])
        src_rgbs = []
        for filename in context_paths:
            src_rgb = read_image(os.path.join(self.root_dir, self.subdir, session, filename))
            src_rgb = resize_image(src_rgb, img_size).permute(1, 2, 0) #/ 255.
            src_rgbs.append(src_rgb)

        pose_path = os.path.join(self.root_dir, self.subdir, session, filename).replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.genfromtxt(pose_path)

        context_pose_paths = [os.path.join(self.root_dir, self.subdir, session, x).replace('color', 'pose').
                                replace('.jpg', '.txt') for x in context_paths]
        context_poses = [np.genfromtxt(x) for x in context_pose_paths]

        src_cameras = []
        for src_pose in context_poses:
            # rel_pose = np.matmul(np.linalg.inv(src_pose), pose).astype(np.float32)
            src_pose = src_pose.astype(np.float32)
            src_camera = np.concatenate((list(img_size), intrinsics.flatten(), src_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = torch.stack(src_rgbs, dim=0)
        src_cameras = np.stack(src_cameras, axis=0)
        camera = np.concatenate((list(img_size), intrinsics.flatten(), pose.flatten())).astype(np.float32)

        depth_range = torch.tensor([self.min_depth, self.max_depth])
        # print(f'src_cameras shape: {src_cameras.shape}')

        sample = {
            'rgb': rgb[..., :3],
            'camera': torch.from_numpy(camera),
            'rgb_path': os.path.join(self.root_dir, self.subdir, session, filename),
            'src_rgbs': src_rgbs[..., :3],
            'src_cameras': torch.from_numpy(src_cameras),
            'depth_range': depth_range,
            'idx': idx,
            'scaled_shape': [0,0]
        }

        # print(filename, context_paths)

        # Add depth information if requested
        if self.with_depth:
            sample.update({'depth': resized_depth,})

        return sample

