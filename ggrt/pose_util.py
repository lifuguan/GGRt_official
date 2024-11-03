import os
import tqdm
import networkx
import queue

import torch
import torch.nn.functional as torch_F
import numpy as np

from typing import Tuple
from easydict import EasyDict as edict
from einops import rearrange
# from hloc.utils.database import COLMAPDatabase, blob_to_array, \
    # pair_id_to_image_ids, image_ids_to_pair_id

from ggrt.geometry.lie_group import SE3q, SE3
from ggrt.geometry.utils import get_all_g2o_files, read_g2o_file, read_g2o_file_valid


# @torch.cuda.amp.autocast(dtype=torch.float32)
def axis_angle_to_R(v):
    """
    Convert an axis-angle vector to rotation matrix
    from https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/lie_group_helper.py#L47

    Inputs:
        v: (B, 3)
    
    Outputs:
        R: (B, 3, 3)
    """

    if len(v.shape) == 1:
        v = v.unsqueeze(0)

    if len(v.shape) == 3:
        v = v.squeeze(0)
    
    zero = torch.zeros_like(v[:, :1]) # (B, 1)
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], 1) # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], 1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1) # (B, 3, 3)

    norm_v = rearrange(torch.norm(v, dim=1)+1e-7, 'b -> b 1 1')
    eye = torch.eye(3, device=v.device)
    R = eye + (torch.sin(norm_v)/norm_v)*skew_v + \
        ((1-torch.cos(norm_v))/norm_v**2)*(skew_v@skew_v)
    return R


def euler_angle_to_R(angle):
    """Convert euler angles to rotation matrix"""
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).view(B, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat


class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self, R=None, t=None):
        # construct a camera pose from the given R and/or t
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R,torch.Tensor):
                R = torch.tensor(R)
            if not isinstance(t,torch.Tensor):
                t = torch.tensor(t)
        
        assert(R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1) # [...,3,4]
        
        assert(pose.shape[-2:] == (3, 4))
        return pose

    def invert(self, pose, use_inverse=False):
        # Invert a camera pose
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        
        return pose_new

    @classmethod
    def from_vec(cls, vec):
        '''
        Initializes from a [B, 6] batch vector
        '''
        # axis_angle, tvec = vec[:, :3], vec[:, 3:]
        # rotation_mat = axis_angle_to_R(axis_angle)
        tvec, rot = vec[:, :3], vec[:, 3:]
        rotation_mat = euler_angle_to_R(rot)
        mat = torch.cat([rotation_mat, tvec.unsqueeze(-1)], dim=2) # [B, 3, 4]

        pose = torch.eye(4, device=vec.device, dtype=vec.dtype).repeat([len(vec), 1, 1])
        pose[:, :3, :3] = mat[:, :3, :3]
        pose[:, :3, -1] = mat[:, :3, -1]

        return pose


class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self, w): # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    def SO3_to_so3(self, R, eps=1e-7): # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = (
            (trace - 1) / 2).clamp(-1 + eps, 1 - eps
        ).acos_()[..., None, None] % np.pi # ln(R) will explode if theta==pi
        # FIXME: wei-chiu finds it weird
        lnR = 1 / (2 * self.taylor_A(theta) + 1e-8
            ) * (R - R.transpose(-2, -1))
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2],lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def se3_to_SE3(self, wu): # [...,3]
        w,u = wu.split([3, 3], dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return Rt

    def SE3_to_se3(self, Rt, eps=1e-8): # [...,3,4]
        R,t = Rt.split([3, 1], dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta ** 2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu    

    def skew_symmetric(self, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O, -w2, w1], dim=-1),
                          torch.stack([w2, O, -w0], dim=-1),
                          torch.stack([-w1, w0, O], dim=-1)], dim=-2)
        return wx

    def taylor_A(self, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            if i > 0:
                denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_B(self, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_C(self, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

lie = Lie()
pose = Pose()


def rotation_distance(R1, R2, eps=1e-7):
    """
    Args:
        R1: rotation matrix from camera 1 to world
        R2: rotation matrix from camera 2 to world
    Return:
        angle: the angular distance between camera 1 and camera 2.
    """
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    # R_diff = R1 @ R2.transpose(-2, -1)
    R_diff = R1.transpose(-2, -1) @ R2
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    
    # numerical stability near -1/+1
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()
    
    return angle


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    
    return X_hom


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def to_backend_tracks(tracks: dict):
    Tracks = torch.classes.tracks.Tracks
    TrackElements = torch.classes.track_elements.TrackElements

    tracks_backend = Tracks(len(tracks.keys()))
    for key in tracks.keys():
        track_elements_py = tracks[key]
        track_elements = TrackElements(len(track_elements_py))
        for element in track_elements_py:
            track_elements.add_track_element(element.image_id, element.point2D_idx)
        tracks_backend.add_track_element_vec(track_elements)

    return tracks_backend


class PoseInitializer(object):
    def __init__(self, data_path: str, image_ids: list, load_external: bool, args) -> None:
        super(PoseInitializer, self).__init__()
        self.device = torch.device(f'cuda:{args.local_rank}')
        
        if load_external == False:
            return

        # lower than this: inliers; higher than this: outliers. In-between: don't care.
        self.thresh_rot_low, self.thresh_rot_high = 5.0, 15.0

        self.pseudo_pose_gt, self.view_graph = self._load_view_graph(data_path, image_ids)
        
        self.position_estimator = None

    def init_poses_from_mst(self):
        num_poses = self.pseudo_pose_gt.shape[0]

        self.init_global_rotations_from_mst(self.two_view_geometries, num_poses)
        self.init_global_positions()

        # Rotations are from world to camera, while ibrnet uses camera to world.
        poses = torch.cat(
            (self.init_global_positions.transpose(1, 2), self.init_global_positions),
            dim=-1
        )

        # TODO(chenyu): recenter poses.

        return poses

    def init_poses_from_noisy_gt(self, pose_gt, noise_level=0.15, outlier_ratio=0.2):
        num_poses = pose_gt.shape[0]
        
        # Perturb ground truth poses with noise.
        so3_noise = torch.randn(num_poses, 3, device=self.device) * noise_level
        eu3_noise = torch.randn(num_poses, 3, device=self.device) * 0.2 * noise_level
        SO3_noise = axis_angle_to_R(so3_noise)

        init_poses = torch.from_numpy(pose_gt).to(self.device).float()
        init_poses[..., :3, :3] = SO3_noise @ init_poses[..., :3, :3]
        init_poses[..., :3, 3] += eu3_noise

        # se3_noise = torch.randn(num_poses, 6, device=self.device) * noise_level
        # pose_noise = torch.cat(
        #     (lie.se3_to_SE3(se3_noise),
        #     torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device).repeat(num_poses, 1, 1)
        # ), dim=1)

        # # init_poses = torch.from_numpy(pose_gt).to(self.device).float() @ pose_noise
        # init_poses = pose_noise @ torch.from_numpy(pose_gt).to(self.device).float()

        # Construct outliers in poses.
        num_outlier_poses = int(num_poses * outlier_ratio)
        # print(f'[INFO] Number of outlier poses: {num_outlier_poses}')
        if num_outlier_poses > 0:
            pose_indices = np.array([i for i in range(num_poses)])
            np.random.shuffle(pose_indices)
            outlier_pose_indices = pose_indices[:num_outlier_poses]

            pose_outlier_level = 0.5
            se3_outlier = torch.randn(num_outlier_poses, 6, device=self.device) * pose_outlier_level
            pose_outlier = torch.cat(
                (lie.se3_to_SE3(se3_outlier),
                torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device).repeat(num_outlier_poses, 1, 1)
            ), dim=1)
            # init_poses[outlier_pose_indices, :, :] = init_poses[outlier_pose_indices, :, :] @ pose_outlier
            init_poses[outlier_pose_indices, :, :] = pose_outlier @ init_poses[outlier_pose_indices, :, :]

        return init_poses
        

    @torch.no_grad()
    def _set_reference_frame(self, image_id, pose):        
        self.ref_image_id = image_id
        self.ref_image_rotation = pose[:3, :3]
        self.position_estimator.fix_camera_positions({image_id: pose[:3, 3].double()})

    @torch.no_grad()
    def init_global_rotations_from_mst(self,
                                       two_view_geometries: dict,
                                       num_rotations):
        init_global_rotations = torch.eye(3, device=self.device).repeat(num_rotations, 1, 1)
        init_global_rotations[self.ref_image_id, :, :] = self.ref_image_rotation

        # Build a weighted graph.
        graph = networkx.Graph()
        for edge in two_view_geometries.keys():
            weight = 1.0 / float(two_view_geometries[edge][1])
            graph.add_edge(edge[0], edge[1], weight=weight)
        
        # Find the minimum spanning tree.
        mst = networkx.minimum_spanning_tree(graph)

        visited = [False for i in range(num_rotations)]
        visited[self.ref_image_id] = True
        qu = queue.PriorityQueue()

        def compute_rotation(i, j):
            if i < j:
                R_ij = two_view_geometries[(i, j)][0]
            else:
                R_ij = two_view_geometries[(j, i)][0].transpose(0, 1)

            R_ij = R_ij.to(self.device).float()
            init_global_rotations[j] = R_ij @ init_global_rotations[i]

        def add_edges_to_heap(image_id):
            neighbors = list(mst.neighbors(image_id))
            for j in neighbors:
                if visited[j]: continue

                priority_level = int(mst[image_id][j]['weight'] * 1e5)
                qu.put([priority_level, (image_id, j)])

        add_edges_to_heap(self.ref_image_id)

        while not qu.empty():
            edge = qu.get()
            i, j = edge[1][0], edge[1][1]
            if visited[j] == True: continue

            compute_rotation(i, j)
            visited[j] = True

            add_edges_to_heap(j)

        self.init_global_rotations = init_global_rotations.to(self.device)

    @torch.no_grad()
    def init_global_positions(self):
        global_rotations_dict = dict()
        for i in range(self.init_global_rotations.shape[0]):
            global_rotations_dict[i] = self.init_global_rotations[i].double()

        # Obtain global positions by differentiable SVD.
        self.init_global_positions = self.position_estimator.estimate_positions(
            global_rotations_dict).to(self.device).float()

    def parse_cameras_and_bounds(self):
        filename = os.path.join(self.path_model, "poses_bounds.npy")
        data = torch.tensor(np.load(filename), dtype=torch.float32)
        
        # parse cameras (intrinsics and poses)
        cam_data = data[:, :-2].view([-1, 3, 5]) # [N,3,5]
        poses_raw = cam_data[..., :4] # [N,3,4]
        poses_raw[..., 0], poses_raw[..., 1] = poses_raw[..., 1], -poses_raw[..., 0]
        raw_H, raw_W, self.focal = cam_data[0, :, -1]
        assert(self.raw_H == raw_H and self.raw_W == raw_W)

        # parse depth bounds
        bounds = data[:, -2:] # [N,2]
        scale = 1. / (bounds.min() * 0.75) # not sure how this was determined
        poses_raw[..., 3] *= scale
        bounds *= scale
        
        # roughly center camera poses
        poses_raw = self.center_camera_poses(poses_raw)
        return poses_raw, bounds

    def center_camera_poses(self, poses):
        # compute average pose
        center = poses[..., 3].mean(dim=0)

        vz = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
        vy_hat = poses[..., 1].mean(dim=0)
        
        # in the original implementation, v0
        vx = torch_F.normalize(vy_hat.cross(vz), dim=0)
        vy = vz.cross(vx)
        pose_avg = torch.stack([vx,vy,vz,center],dim=-1)[None] # [1,3,4]
        
        # apply inverse of averaged pose
        poses = pose.compose([poses, pose.invert(pose_avg)])
        
        return poses
    
    def total_track_elements(self, tracks):
        num_track_elements = 0
        for key in tracks.keys():
            num_track_elements += len(tracks[key])
        
        return num_track_elements

    def _read_keypoints(self, path: str, training_image_ids: list) -> dict:
        print("[INFO] Loading keypoints...")
        
        database_path = os.path.join(path, "database.db")
        db = COLMAPDatabase.connect(database_path)

        keypoints_dict = dict()
        for image_id, rows, cols, data in db.execute(
                "SELECT image_id, rows, cols, data FROM keypoints"):
            if image_id - 1 not in training_image_ids:
                continue
        
            keypoints_dict[image_id - 1] = torch.from_numpy(
                blob_to_array(data, np.float32, (rows, cols))
            )[:, :2]
        
        db.close()

        print(f'[INFO] Loaded keypoints for {len(keypoints_dict)} images.')
        return keypoints_dict

    def _get_all_intrinsics(self, path):
        all_intrinsics = {}

        database_path = os.path.join(path, "database.db")
        db = COLMAPDatabase.connect(database_path)
        for image_id, name, camera_id in db.execute("SELECT image_id, name, camera_id FROM images"):
            rows = db.execute(
                f"SELECT model, params FROM cameras WHERE camera_id={camera_id}")
            model, params = next(rows)
            params = blob_to_array(params, np.float64)

            if model == 2: # SIMPLE_RADIAL
                fx, cx, cy = params[0], params[1], params[2]
                fy = fx
            else:
                print(f'[ERROR] Camera Model {model} not supported!')
                raise NotImplementedError

            intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
            # assert(image_id - 1 == self.image_name_to_idx[name])
            all_intrinsics[image_id - 1] = intrinsics
        
        return all_intrinsics

    def _read_matches_num(self, path: str):
        database_path = os.path.join(path, "database.db")
        # print(f'[DEBUG] database path: {database_path}')
        db = COLMAPDatabase.connect(database_path)

        image_pair_to_matches_num = dict()
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches"):
            if data is None:
                continue
            
            image_pair = pair_id_to_image_ids(pair_id)
            num_matches = blob_to_array(data, np.uint32, (-1, 2)).shape[0]
            image_pair_to_matches_num[(image_pair[0] - 1, image_pair[1] - 1)] = num_matches

        db.close()

        return image_pair_to_matches_num

    def _load_view_graph(self, path: str, training_image_ids: list):
        g2o_files = get_all_g2o_files(path)
        if len(g2o_files) == 0:
            print(f'[ERROR] No G2O files existed in {path}')
        view_graph_filename = g2o_files[0]

        two_view_geometries = dict()
        graph = networkx.Graph()

        # absolute rotations: N*7, relative rotations: M*7, edge_indices: M*2
        # absolute_poses, relative_poses, edge_indices = read_g2o_file_valid(
        #     filename=view_graph_filename,
        #     valid_indices=training_image_ids,
        #     src='COLMAP'
        # )
        absolute_poses, relative_poses, edge_indices, _ = read_g2o_file(
            filename=view_graph_filename,
            src='COLMAP'
        )
        
        gt_abs_poses = SE3q.from_rtvec(torch.from_numpy(absolute_poses)) # pseudo ground truth.
        relative_poses = SE3q.from_rtvec(torch.from_numpy(relative_poses))
        edge_index = edge_indices.transpose().astype(np.int64) # (2, M)

        # Remove one of the directions of the edges (this is handled internally)
        to_keep = edge_index[0] < edge_index[1] # (M, )
        edge_index = edge_index[:, to_keep]
        relative_poses = relative_poses[to_keep]

        # Convert to matrix form.
        relative_poses = SE3.from_rtvec(relative_poses.data) # [M, 3, 4]
        gt_abs_poses = SE3.from_rtvec(gt_abs_poses.data) #.data

        pose_i_gt, pose_j_gt = gt_abs_poses[edge_index[0, :]], gt_abs_poses[edge_index[1, :]]
        pose_ij_gt = pose_j_gt * pose_i_gt.inv()

        rel_pose_errors = pose_ij_gt.compare(relative_poses)

        # Initialize to 0.5 (don't care)
        edge_inliers = np.full(rel_pose_errors['rot_deg'].shape, 0.5, dtype=np.float32)
        inliers = rel_pose_errors['rot_deg'] < self.thresh_rot_low
        outliers = rel_pose_errors['rot_deg'] > self.thresh_rot_high
        edge_inliers[inliers] = 1.0
        edge_inliers[outliers] = 0.0

        image_pair_to_matches_num = self._read_matches_num(path)
        
        all_edge_indices = edge_index.T
        for i, edge in enumerate(all_edge_indices):
            image_pair = (edge[0], edge[1])
            if image_pair not in image_pair_to_matches_num.keys():
                continue
            
            two_view_geometries[image_pair] = edict(
                rel_pose=relative_poses.data[i, :4, :4],
                num_matches=image_pair_to_matches_num[image_pair],
                inlier_mask=edge_inliers[i]
            )
            graph.add_edge(image_pair[0], image_pair[1], weight=int(image_pair_to_matches_num[image_pair]))

        view_graph = {"two_view_geometries": two_view_geometries, "graph": graph}

        return gt_abs_poses, view_graph


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


# https://github.com/facebookresearch/pytorch3d/blob/8c2b0b01f87f62aa66019a88d8461d4e11f72cf6/pytorch3d/transforms/so3.py#L110
def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].

    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.

    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_rot: Batch of vectors of shape `(minibatch, 3)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `log_rot` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    return _so3_exp_map(log_rot, eps=eps)[0]


def _so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def euler_angle_to_R(angle):
    """Convert euler angles to rotation matrix"""
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).view(B, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat


def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[:, None, None]
        + (
            log_rotation_hat_square
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[
                :, None, None
            ]
        )
    )

    return V


# https://github.com/facebookresearch/pytorch3d/blob/8c2b0b01f87f62aa66019a88d8461d4e11f72cf6/pytorch3d/transforms/se3.py#L12
def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.
    A SE(3) matrix has the following form:
        ```
        [ R T ]
        [ 0 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.
    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.
    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ hat(log_rotation) 0 ]
                         [   log_translation 1 ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].
    Note that for any `log_transform` with `0 <= ||log_rotation|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_log_map(se3_exponential_map(log_transform)) == log_transform
    ```
    The conversion has a singularity around `||log(transform)|| = 0`
    which is handled by clamping controlled with the `eps` argument.
    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid unstable gradients in the singular case.
    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.
    Raises:
        ValueError if `log_transform` is of incorrect shape.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., :3]
    log_rotation = log_transform[..., 3:]

    # rotation is an exponential map of log_rotation
    (
        R,
        rotation_angles,
        log_rotation_hat,
        log_rotation_hat_square,
    ) = _so3_exp_map(log_rotation, eps=eps)

    # translation is V @ T
    V = _se3_V_matrix(
        log_rotation,
        log_rotation_hat,
        log_rotation_hat_square,
        rotation_angles,
        eps=eps,
    )
    T = torch.bmm(V, log_translation[:, :, None])[:, :, 0]

    transform = torch.zeros(
        N, 4, 4, dtype=log_transform.dtype, device=log_transform.device
    )

    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    transform[:, 3, 3] = 1.0

    return transform


def random_so3(num_poses: int = 1) -> torch.Tensor:
    so3 = torch.randn((num_poses, 3), dtype=torch.float32)
    return so3


def random_SO3(num_poses: int = 1) -> torch.Tensor:
    so3 = random_so3(num_poses=num_poses)
    SO3 = so3_exp_map(so3)
    
    return SO3


def random_se3(mean: float=0, std: float=1, num_poses: int = 1):
    se3 = torch.normal(mean, std, (num_poses, 6), dtype=torch.float32)
    return se3


def random_SE3(mean: float=0, std: float=1, num_poses: int=1):
    se3 = random_se3(mean, std, num_poses=num_poses)
    se3[:, :3] = torch.clamp(se3[:, :3], min=-0.2, max=0.2) # clamp translation
    SE3 = se3_exp_map(se3)
    
    return SE3
