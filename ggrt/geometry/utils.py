import os
from typing import List

import torch
import numpy as np

import imageio


def get_filename_no_ext(filename):
    return os.path.splitext(filename)[0]


def get_file_extension(filename):
    return os.path.splitext(filename)[-1]


def get_all_image_names(dir, formats=[
    '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG'
    ]) -> List:
    image_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if not get_file_extension(file) in formats:
                continue
            # print(f"{os.path.join(dir, root, file)}")
            image_paths.append(os.path.join(dir, root, file))
    # print(f"num images: {len(image_paths)}")
    return sorted(image_paths)


def imread(file):
    if file.endswith('png') or file.endwith('PNG'):
        return imageio.imread(file, ignoregamma=True)
    else:
        return imageio.imread(file)

trans_t = lambda t : torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1, 0,           0,            0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi),  0],
    [0, 0,           0,            1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0,          1, 0,           0],
    [np.sin(th), 0,  np.cos(th), 0],
    [0,          0, 0,           1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def invert_pose_np(pose: np.array) -> np.array:
    """
    Args:
        pose: camera pose of shape (4, 4) with the last row to be: [0, 0, 0, 1].
    
    Returns:
        a inverted pose in the opposite direction.
    """
    inverted_pose = np.zeros_like(pose)
    inverted_pose[3, 3] = 1.
    inverted_pose[:3, :3] = pose[:3, :3].T
    inverted_pose[:3, -1] = -np.dot(inverted_pose[:3, :3], pose[:3, -1])
    return inverted_pose


def coordinate_opengl_to_sfm_np(pose: np.array) -> np.array:
    rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    pose[:3, :] = np.matmul(rotation, pose[:3, :])
    return pose


def invert_pose_torch(pose: torch.tensor) -> torch.tensor:
    """
    Args:
        pose: camera pose of tensor shape (4, 4) with the last row to be: [0, 0, 0, 1].
    
    Returns:
        a inverted pose in the opposite direction.
    """
    inverted_pose = torch.zeros_like(pose)
    inverted_pose[3, 3] = 1.
    inverted_pose[:3, :3] = pose[0:3, 0:3].t()
    # print(inverted_pose[:3, :3].shape)
    # print(pose[:3, -1].shape)
    inverted_pose[:3, -1] = -torch.matmul(inverted_pose[:3, :3], pose[:3, -1])
    return inverted_pose


def coordinate_opengl_to_sfm_torch(pose: torch.tensor) -> torch.tensor:
    """
    Args:
        pose: camera pose of tensor shape (4, 4) or (3, 4),
            which is located at the opengl coordinate system.
    
    Returns:
        The transformed pose in the Pytorch3D rendering coordinate system.
    """
    rotation = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=pose.dtype)
    pose[:3, :] = torch.mm(rotation, pose[:3, :])
    return pose


def invert_poses_torch(poses: torch.tensor) -> torch.tensor:
    """
    Args:
        poses: camera poses of tensor shape (N, 4, 4).
    
    Returns:
        The inverted poses in the opposite direction.
    """
    inverted_poses = torch.zeros_like(poses)
    last_row = torch.tensor([[0, 0, 0, 1]])
    for i, pose in enumerate(poses):
        if pose.shape[0] == 3:
            pose = torch.cat((pose, last_row), dim=0)
        # print(f"pose shape: {pose.shape}")
        inverted_poses[i, :3, :] = invert_pose_torch(pose)[:3, :]
    return inverted_poses


def coordinates_opengl_to_sfm_torch(poses: torch.tensor) -> torch.tensor:
    """
    Args:
        pose: camera pose of tensor shape (N, 4, 4) or (N, 3, 4), which is located at 
            the opengl coordinate system.
    
    Returns:
        The transformed pose in the Pytorch3D rendering coordinate system.
    """
    poses_hat = torch.zeros_like(poses)
    for i, pose in enumerate(poses):
        poses_hat[i, :, :] = coordinate_opengl_to_sfm_torch(pose)
    return poses_hat


def get_filename_from_abs_path(abs_path):
    return abs_path.split('/')[-1]


def get_filename_no_ext(filename):
    return os.path.splitext(filename)[0]


def get_file_extension(filename):
    return os.path.splitext(filename)[-1]


def get_all_g2o_files(dir, formats=['.g2o', '.G2O']) -> List:
    g2o_filenames = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if not get_file_extension(file) in formats:
                continue
            g2o_filenames.append(os.path.join(dir, root, file))
    return sorted(g2o_filenames)


def read_g2o_file(filename: str, src='COLMAP'):
    # Obtain the nodes' number and edges' number according to filename.
    segments = os.path.splitext(get_filename_from_abs_path(filename))[0].split('_')
    num_nodes = int(segments[1][1:])
    num_edges = int(segments[2][1:])

    absolute_poses = np.zeros((num_nodes, 7), dtype=np.float64)
    relative_poses = np.zeros((num_edges, 7), dtype=np.float64)
    edge_indices = np.zeros((num_edges, 2))
    edge_pair_to_index = dict()

    f = open(filename)
    line = f.readline()
    edge_index = 0
    while line:
        data = line.split(' ')
        line = f.readline()
        # print(f'data[0]: {data[0].find("VERTEX_SE3:QUAT")}')
        if data[0].find('VERTEX_SE3:QUAT') >= 0:
            # Read vertices' data.
            node_index = int(data[1]) # - 1 if src == 'COLMAP' else int(data[1]) # node id
            absolute_poses[node_index, 0] = float(data[8]) # qw
            absolute_poses[node_index, 1] = float(data[5]) # qx
            absolute_poses[node_index, 2] = float(data[6]) # qy
            absolute_poses[node_index, 3] = float(data[7]) # qz
            absolute_poses[node_index, 4] = float(data[2]) # tx
            absolute_poses[node_index, 5] = float(data[3]) # ty
            absolute_poses[node_index, 6] = float(data[4]) # tz
            quat_norm = np.linalg.norm(absolute_poses[node_index, 0:4], ord=2)
            assert abs(quat_norm - 1) < 1e-5, f"Quaternion norm is not 1: {absolute_poses[node_index, 0:4]}"
        elif data[0].find('EDGE_SE3:QUAT') >= 0:
            # Read edges's data.
            src, dst = int(data[1]), int(data[2]) # edge index
            # if src == 'COLMAP':
            #     src -= 1
            #     dst -= 1
            
            edge_indices[edge_index, 0] = src
            edge_indices[edge_index, 1] = dst
            relative_poses[edge_index, 0] = float(data[9]) # qw
            relative_poses[edge_index, 1] = float(data[6]) # qx
            relative_poses[edge_index, 2] = float(data[7]) # qy
            relative_poses[edge_index, 3] = float(data[8]) # qz
            relative_poses[edge_index, 4] = float(data[3]) # tx
            relative_poses[edge_index, 5] = float(data[4]) # ty
            relative_poses[edge_index, 6] = float(data[5]) # tz
            edge_pair_to_index[(src, dst)] = edge_index
            edge_index += 1
        else:
            continue

    f.close()

    return absolute_poses, relative_poses, edge_indices, edge_pair_to_index


def read_g2o_file_valid(filename: str, valid_indices: list, src='COLMAP'):
    # Obtain the nodes' number and edges' number according to filename.
    segments = os.path.splitext(get_filename_from_abs_path(filename))[0].split('_')
    num_nodes = int(segments[1][1:])
    # num_edges = int(segments[2][1:])

    absolute_poses = np.zeros((num_nodes, 7), dtype=np.float64)
    relative_poses = []
    edge_indices = []

    f = open(filename)
    line = f.readline()

    while line:
        data = line.split(' ')
        line = f.readline()
        # print(f'data[0]: {data[0].find("VERTEX_SE3:QUAT")}')
        if data[0].find('VERTEX_SE3:QUAT') >= 0:
            # Read vertices' data.
            node_index = int(data[1]) # - 1 if src == 'COLMAP' else int(data[1])                      # node id
            absolute_poses[node_index, 0] = float(data[8]) # qw
            absolute_poses[node_index, 1] = float(data[5]) # qx
            absolute_poses[node_index, 2] = float(data[6]) # qy
            absolute_poses[node_index, 3] = float(data[7]) # qz
            absolute_poses[node_index, 4] = float(data[2]) # tx
            absolute_poses[node_index, 5] = float(data[3]) # ty
            absolute_poses[node_index, 6] = float(data[4]) # tz
            quat_norm = np.linalg.norm(absolute_poses[node_index, 0:4], ord=2)
            assert abs(quat_norm - 1) < 1e-5, f"Quaternion norm is not 1: {absolute_poses[node_index, 0:4]}"
        elif data[0].find('EDGE_SE3:QUAT') >= 0:
            # Read edges's data.
            src, dst = int(data[1]), int(data[2]) # edge index
            # if src == 'COLMAP':
            #     src -= 1
            #     dst -= 1
            
            if src not in valid_indices or dst not in valid_indices:
                continue

            edge_index = np.array([src, dst])
            relative_pose = np.array([
                float(data[9]), float(data[6]), float(data[7]), float(data[8]), # qw, qx, qy, qz
                float(data[3]), float(data[4]), float(data[5])])               # tx, ty, tz
            relative_poses.append(relative_pose)
            edge_indices.append(edge_index)
        else:
            continue

    f.close()

    relative_poses = np.stack(relative_poses, axis=0)
    edge_indices = np.stack(edge_indices, axis=0)

    return absolute_poses, relative_poses, edge_indices