
import sys
import configargparse
sys.path.append('../')

import json
import torch
import numpy as np

from utils_loc import *
from ggrt.geometry.align_poses import align_ate_c2b_use_a2b
from ggrt.pose_util import rotation_distance

from scipy.spatial.transform import Rotation


def read_g2o_file(filename: str):
    poses_dict = {}

    f = open(filename)
    line = f.readline()

    while line:
        data = line.split(' ')
        line = f.readline()

        if data[0].find('VERTEX_SE3:QUAT') >= 0:
            # Read vertices' data.
            node_index = int(data[1]) # - 1 if src == 'COLMAP' else int(data[1]) # node id
            # qw, qx, qy, qz, tx, ty, tz
            pose = np.array([float(data[8]), float(data[5]), float(data[6]), float(data[7]),
                                 float(data[2]), float(data[3]), float(data[4])])
            
            quat_norm = np.linalg.norm(pose[0:4], ord=2)
            assert abs(quat_norm - 1) < 1e-5, f"Quaternion norm is not 1: {absolute_poses[node_index, 0:4]}"

            poses_dict[node_index] = pose
        else:
            continue
    f.close()
    print(f'poses_dict len: {len(poses_dict)}')

    num_nodes = len(poses_dict)
    absolute_poses = np.zeros((num_nodes, 7), dtype=np.float64)
    for key in poses_dict.keys():
        absolute_poses[key] = poses_dict[key]

    return absolute_poses


def write_camera_poses_to_COLMAP_format(output_dir, gt_poses, pred_poses,
                                        camera_model='PINHOLE', width=504, height=378,
                                        fx=200, fy=200, cx=252, cy=189):
    num_poses = 2 * gt_poses.shape[0]
    camera_id = 0
    gt_cluster_id = 0
    pred_cluster_id = 1
    print(f'[INFO] Writing camera poses to {output_dir}...')

    # Write cameras txt.
    camera_file_path = os.path.join(output_dir, 'cameras.txt')
    camera_file = open(camera_file_path, 'w')
    camera_file.write('# Camera list with one line of data per camera:\n')
    camera_file.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
    camera_file.write(f'# Number of cameras: {num_poses}\n')
    camera_file.write(f'{camera_id} {camera_model} {width} {height} {fx} {fy} {cx} {cy}\n')
    camera_file.close()

    # Write images txt.
    image_file_path = os.path.join(output_dir, 'images.txt')
    image_file = open(image_file_path, 'w')
    image_file.write('# Image list with two lines of data per image:\n')
    image_file.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
    image_file.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
    image_file.write(f'# Number of images: {num_poses}, mean observations per image: 0\n')

    # Writing camera poses.
    for i in range(gt_poses.shape[0]):
        pose = gt_poses[i]
        qvec = Rotation.from_matrix(pose[:3, :3]).as_quat()
        tvec = pose[:3, -1]
        image_file.write(f'{i} {gt_cluster_id} {qvec[3]} {qvec[0]} {qvec[1]} {qvec[2]} {tvec[0]} {tvec[1]} {tvec[2]} ' +
                         f'{camera_id} {i}\n')
        image_file.write('-1\n') # No 3D points, just set a placeholder.
    for i in range(pred_poses.shape[0]):
        pose = pred_poses[i]
        qvec = Rotation.from_matrix(pose[:3, :3]).as_quat()
        tvec = pose[:3, -1]
        image_file.write(f'{i} {pred_cluster_id} {qvec[3]} {qvec[0]} {qvec[1]} {qvec[2]} {tvec[0]} {tvec[1]} {tvec[2]} ' +
                         f'{camera_id} {i}\n')
        # image_file.write(f'{i} {qvec[3]} {qvec[0]} {qvec[1]} {qvec[2]} {tvec[0]} {tvec[1]} {tvec[2]} ' +
        #                  f'{camera_id} {i}\n')
        image_file.write('-1\n') # No 3D points, just set a placeholder.
    image_file.close()

    # Write points txt (pseudo file).
    points_file_path = os.path.join(output_dir, 'points3D.txt')
    points_file = open(points_file_path, 'w')
    points_file.write('')
    points_file.close()


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--gt_view_graph_path', type=str, default='',
                       help='ground truth view graph path')
    parser.add_argument('--pred_view_graph_path', type=str, default='',
                       help='predicted view graph path')
    parser.add_argument('--output_dir', type=str, default='',
                       help='directory to store results.')

    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    
    gt_view_graph_filename = args.gt_view_graph_path
    pred_view_graph_filename = args.pred_view_graph_path

    gt_poses_quat = read_g2o_file(gt_view_graph_filename)
    pred_poses_quat = read_g2o_file(pred_view_graph_filename)

    num_poses = gt_poses_quat.shape[0]
    gt_poses, pred_poses = [], []
    for i in range(num_poses):
        gt_qw, gt_qx, gt_qy, gt_qz, gt_tx, gt_ty, gt_tz = \
            gt_poses_quat[i, 0], gt_poses_quat[i, 1], gt_poses_quat[i, 2], gt_poses_quat[i, 3],\
            gt_poses_quat[i, 4], gt_poses_quat[i, 5], gt_poses_quat[i, 6]
        gt_rot = Rotation.from_quat(np.array([gt_qx, gt_qy, gt_qz, gt_qw])).as_matrix()
        gt_tvec = np.array([[gt_tx], [gt_ty], [gt_tz]])

        gt_pose = np.concatenate([gt_rot, gt_tvec], axis=1)
        gt_poses.append(gt_pose)

        pred_qw, pred_qx, pred_qy, pred_qz, pred_tx, pred_ty, pred_tz = \
            pred_poses_quat[i, 0], pred_poses_quat[i, 1], pred_poses_quat[i, 2], pred_poses_quat[i, 3],\
            pred_poses_quat[i, 4], pred_poses_quat[i, 5], pred_poses_quat[i, 6]
        pred_rot = Rotation.from_quat(np.array([pred_qx, pred_qy, pred_qz, pred_qw])).as_matrix()
        pred_tvec = np.array([[pred_tx], [pred_ty], [pred_tz]])
        pred_pose = np.concatenate([pred_rot, pred_tvec], axis=1)
        pred_poses.append(pred_pose)
    
    gt_poses = torch.from_numpy(np.stack(gt_poses)).float()
    pred_poses = np.stack(pred_poses)
    # The predicted absolute poses rotate points from world to camera!
    pred_poses[..., :3, :3] = np.transpose(pred_poses[..., :3, :3], axes=(0, 2, 1))
    pred_poses = torch.from_numpy(pred_poses).float()

    aligned_pred_poses = align_ate_c2b_use_a2b(pred_poses, gt_poses)

    R_aligned, t_aligned = aligned_pred_poses.split([3, 1], dim=-1)
    R_aligned, t_aligned = R_aligned[..., :3, :3], t_aligned[..., :3, :]
    R_gt, t_gt = gt_poses.split([3, 1], dim=-1)

    R_error = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
    t_error = (t_aligned - t_gt)[..., 0].norm(dim=-1)
    
    mean_rotation_error = np.rad2deg(R_error.mean().cpu())
    mean_position_error = t_error.mean()
    med_rotation_error = np.rad2deg(R_error.median().cpu())
    med_position_error = t_error.median()

    metrics_dict = {'R_error_mean': str(mean_rotation_error),
                    "t_error_mean": str(mean_position_error),
                    'R_error_med': str(med_rotation_error),
                    't_error_med': str(med_position_error)
                    }
    
    # Writing metrics to json.
    json_obj = json.dumps(metrics_dict, indent=4)
    metrics_dir = os.path.join(args.output_dir, 'metrics')
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    json_file = os.path.join(metrics_dir, 'abs_pose_metrics.json')
    with open(json_file, 'w') as f:
        f.write(json_obj)
    
    # Convert camera poses to DAGSfM's format.
    # TODO(chenyu): set correct camera intrinsics.
    colmap_dir = os.path.join(args.output_dir, 'colmap')
    if not os.path.exists(colmap_dir):
        os.makedirs(colmap_dir)
    write_camera_poses_to_COLMAP_format(colmap_dir, gt_poses, pred_poses)

