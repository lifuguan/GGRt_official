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
import cv2
import math
import numpy as np
import queue
import re
import torchvision.transforms as transforms
import torch

from PIL import Image
from scipy.spatial.transform import Rotation as R
from networkx import Graph
from .base_utils import downsample_gaussian_blur

rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. eucledian norm, of ndarray along axis.
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.
    """
    quaternion = np.zeros((4, ), dtype=np.float64)
    quaternion[:3] = axis[:3]
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle/2.0) / qlen
    quaternion[3] = math.cos(angle/2.0)
    return quaternion


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


def rectify_inplane_rotation(src_pose, tar_pose, src_img, th=40):
    relative = np.linalg.inv(tar_pose).dot(src_pose)
    relative_rot = relative[:3, :3]
    r = R.from_matrix(relative_rot)
    euler = r.as_euler('zxy', degrees=True)
    euler_z = euler[0]
    if np.abs(euler_z) < th:
        return src_pose, src_img

    R_rectify = R.from_euler('z', -euler_z, degrees=True).as_matrix()
    src_R_rectified = src_pose[:3, :3].dot(R_rectify)
    out_pose = np.eye(4)
    out_pose[:3, :3] = src_R_rectified
    out_pose[:3, 3:4] = src_pose[:3, 3:4]
    h, w = src_img.shape[:2]
    center = ((w - 1.) / 2., (h - 1.) / 2.)
    M = cv2.getRotationMatrix2D(center, -euler_z, 1)
    src_img = np.clip((255*src_img).astype(np.uint8), a_max=255, a_min=0)
    rotated = cv2.warpAffine(src_img, M, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4)
    rotated = rotated.astype(np.float32) / 255.
    return out_pose, rotated


def random_crop(rgb, camera, src_rgbs, src_cameras, size=(400, 600), center=None):
    h, w = rgb.shape[:2]
    out_h, out_w = size[0], size[1]
    if out_w >= w or out_h >= h:
        return rgb, camera, src_rgbs, src_cameras

    if center is not None:
        center_h, center_w = center
    else:
        center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
        center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)
    center_h=80
    center_w=112
    rgb_out = rgb[center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2, :]
    src_rgbs = np.array(src_rgbs)
    src_rgbs = src_rgbs[:, center_h - out_h // 2:center_h + out_h // 2,
               center_w - out_w // 2:center_w + out_w // 2, :]
    camera[0] = out_h
    camera[1] = out_w
    camera[4] -= center_w - out_w // 2
    camera[8] -= center_h - out_h // 2
    src_cameras[:, 4] -= center_w - out_w // 2
    src_cameras[:, 8] -= center_h - out_h // 2
    src_cameras[:, 0] = out_h
    src_cameras[:, 1] = out_w
    return rgb_out, camera, src_rgbs, src_cameras, camera[2:18].reshape(4, 4)[..., :3, :3], src_cameras[:, 2:18].reshape(-1, 4, 4)[..., :3, :3],center_h,center_w

def loader_resize(rgb, camera, src_rgbs, src_cameras, size=(400, 600)):
    h, w = rgb.shape[:2]
    out_h, out_w = size[0], size[1]
    intrinsics = camera[2:18].reshape(4, 4)
    src_intrinsics = src_cameras[:, 2:18].reshape(-1, 4, 4)
    if out_w >= w or out_h >= h:
        return rgb, camera, src_rgbs, src_cameras, intrinsics[..., :3, :3], src_intrinsics[..., :3, :3]

    ratio_y = out_h / h
    ratio_x = out_w / w
    intrinsics[:1, :1] *= ratio_y
    src_intrinsics[:, :1, :1] *= ratio_y
    intrinsics[1:2, 1:2] *= ratio_x
    src_intrinsics[:, 1:2, 1:2] *= ratio_x
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



def random_crop_img_depth(rgb, camera, src_rgbs, src_cameras, depth, size=(400, 600), center=None):
    h, w = rgb.shape[:2]
    out_h, out_w = size[0], size[1]
    if out_w >= w or out_h >= h:
        return rgb, camera, src_rgbs, src_cameras

    if center is not None:
        center_h, center_w = center
    else:
        center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
        center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)

    rgb_out = rgb[center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2, :]
    src_rgbs = np.array(src_rgbs)
    src_rgbs = src_rgbs[:, center_h - out_h // 2:center_h + out_h // 2,
               center_w - out_w // 2:center_w + out_w // 2, :]
    depth = depth[center_h - out_h // 2:center_h + out_h // 2,
                  center_w - out_w // 2:center_w + out_w // 2]
    camera[0] = out_h
    camera[1] = out_w
    camera[4] -= center_w - out_w // 2
    camera[8] -= center_h - out_h // 2
    src_cameras[:, 4] -= center_w - out_w // 2
    src_cameras[:, 8] -= center_h - out_h // 2
    src_cameras[:, 0] = out_h
    src_cameras[:, 1] = out_w
    return rgb_out, camera, src_rgbs, src_cameras, depth


def random_flip(rgb, camera, src_rgbs, src_cameras):
    h, w = rgb.shape[:2]
    h_r, w_r = src_rgbs.shape[1:3]
    rgb_out = np.flip(rgb, axis=1).copy()
    src_rgbs = np.flip(src_rgbs, axis=-2).copy()
    camera[2] *= -1
    camera[4] = w - 1. - camera[4]
    src_cameras[:, 2] *= -1
    src_cameras[:, 4] = w_r - 1. - src_cameras[:, 4]
    return rgb_out, camera, src_rgbs, src_cameras


def get_color_jitter_params(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    transform = transforms.ColorJitter.get_params(color_jitter.brightness,
                                                  color_jitter.contrast,
                                                  color_jitter.saturation,
                                                  color_jitter.hue)
    return transform


def color_jitter(img, transform):
    '''
    Args:
        img: np.float32 [h, w, 3]
        transform:
    Returns: transformed np.float32
    '''
    img = Image.fromarray((255.*img).astype(np.uint8))
    img_trans = transform(img)
    img_trans = np.array(img_trans).astype(np.float32) / 255.
    return img_trans


def color_jitter_all_rgbs(rgb, ref_rgbs, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    transform = get_color_jitter_params(brightness, contrast, saturation, hue)
    rgb_trans = color_jitter(rgb, transform)
    ref_rgbs_trans = []
    for ref_rgb in ref_rgbs:
        ref_rgbs_trans.append(color_jitter(ref_rgb, transform))

    ref_rgbs_trans = np.array(ref_rgbs_trans)
    return rgb_trans, ref_rgbs_trans


def deepvoxels_parse_intrinsics(filepath, trgt_sidelength, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy = list(map(float, file.readline().split()))[:3]
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        near_plane = float(file.readline())
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    cx = cx / width * trgt_sidelength
    cy = cy / height * trgt_sidelength
    f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))


def get_nearest_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e20  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids


def get_nearby_view_ids(target_id: int, graph: Graph, idx_to_node_id: dict, node_id_to_idx: dict, num_select: int):
    # Map target idx to original node id in view graph.
    target_node_id = idx_to_node_id[target_id]
    
    neighbors = list(graph.neighbors(target_node_id))
    qu = queue.PriorityQueue()

    for j in neighbors:
        if j not in node_id_to_idx.keys():
            continue
        # Larger weight has higher priority level.
        priority_level = int((1.0 / graph[target_node_id][j]['weight']) * 1e5)
        qu.put([priority_level, j])

    k = 0
    select_ids = []
    while not qu.empty() and k < num_select:
        ele = qu.get()
        select_idx = node_id_to_idx[ele[1]] # Map node id in view graph to the index in train set.
        select_ids.append(select_idx)
        k += 1

    select_ids = np.array(select_ids)
    return select_ids


def get_relative_poses(target_id: int, two_view_geometries: dict,
                       idx_to_node_id: dict, selected_ids: np.ndarray) -> np.ndarray:
    """
    Return:
        rel_poses: relative poses from target view to nearby view (world -> camera). [num_views, 6]
    """
    target_node_id = idx_to_node_id[target_id]
    rel_poses = []
    for nearby_id in selected_ids:
        nearby_node_id = idx_to_node_id[nearby_id]
        
        if nearby_node_id == target_node_id:
            rel_pose = np.zeros(6)
        
        else:
            view_pair = (target_node_id, nearby_node_id) if target_node_id < nearby_node_id \
                                                         else (nearby_node_id, target_node_id)

            assert(view_pair in two_view_geometries.keys())
            rel_pose = two_view_geometries[view_pair].rel_pose
            rot_vec =  R.from_matrix(rel_pose[:3, :3]).as_rotvec()
            tvec = rel_pose[:3, 3]
            rel_pose = np.concatenate([rot_vec, tvec], axis=-1)
        rel_poses.append(rel_pose)
    
    rel_poses = np.stack(rel_poses, axis=0)

    return rel_poses


def read_pfm(filename):
    file = open(filename, "rb")
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def get_filename_no_ext(filename):
    return os.path.splitext(filename)[0]


def get_file_extension(filename):
    return os.path.splitext(filename)[-1]


def get_all_image_names(dir, formats=[
    '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']) -> list:
    image_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if not get_file_extension(file) in formats:
                continue
            # print(f"{os.path.join(dir, root, file)}")
            image_paths.append(os.path.join(dir, root, file))
    # print(f"num images: {len(image_paths)}")
    return sorted(image_paths)


def get_filename_from_abs_path(abs_path):
    return abs_path.split('/')[-1]
