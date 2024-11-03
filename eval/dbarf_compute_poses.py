
import sys
import json
import queue
import networkx
sys.path.append('../')

from scipy.spatial.transform import Rotation

from ggrt.base.checkpoint_manager import CheckPointManager
from ggrt.config import config_parser
from ggrt.model.dbarf import DBARFModel
from utils_loc import *
from ggrt.data_loaders import dataset_dict
from ggrt.pose_util import Pose
from ggrt.geometry.align_poses import align_ate_c2b_use_a2b
from ggrt.pose_util import rotation_distance
from eval.eval_dbarf import compose_state_dicts

# os.environ["CUDA_VISIBLE_DEVICES"]="0"


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


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


@torch.no_grad()
def compute_abs_poses_from_mst(model, dataset, mst, num_poses, view_graph_file):
    ref_image_id = 0
    visited = [False for i in range(num_poses)]
    visited[ref_image_id] = 0
    qu = queue.PriorityQueue()

    abs_poses = {ref_image_id: torch.eye(4)}

    def compute_pose(i, j):
        data = dataset.get_data_one_batch(i, j)

        _, pred_rel_poses, __, ___ = model.correct_poses(
            fmaps=None,
            target_image=data['rgb'].unsqueeze(0).cuda(),
            ref_imgs=data['src_rgbs'].unsqueeze(0).cuda(),
            target_camera=data['camera'].unsqueeze(0),
            ref_cameras=data['src_cameras'].unsqueeze(0),
            min_depth=data['depth_range'][0],
            max_depth=data['depth_range'][1],
            scaled_shape=data['scaled_shape'])
        # print(F'pred_rel_poses shape: {pred_rel_poses.shape}')
        rel_pose = Pose.from_vec(pred_rel_poses).to('cpu')
        # print(f'rel_pose shape: {rel_pose.shape}')
        cur_pose = abs_poses[i] @ rel_pose.inverse()
        abs_poses[j] = cur_pose

    def add_edges_to_heap(image_id):
        neighbors = list(mst.neighbors(image_id))
        for j in neighbors:
            if visited[j]:
                continue

            priority_level = int(1e5 / mst[image_id][j]['weight'])
            qu.put([priority_level, (image_id, j)])

    # Computing absolute camera poses along the maximum spanning tree.
    add_edges_to_heap(ref_image_id)

    while not qu.empty():
        edge = qu.get()
        i, j = edge[1][0], edge[1][1]
        if visited[j] == True: continue

        compute_pose(i, j)
        visited[j] = True

        add_edges_to_heap(j)

    # Output camera poses to view graph.
    for image_id in abs_poses.keys():
        pose = abs_poses[image_id].squeeze(0)
        # print(f'pose shape: {pose.shape}')
        rot_mat = pose[:3, :3]
        tvec = pose[:3, 3]
        # print(f'rot_mat shape: {rot_mat.shape}')
        qvec = Rotation.from_matrix(rot_mat).as_quat() # [qx, qy, qz, qw]
        view_graph_file.write(f'VERTEX_SE3:QUAT {image_id} {tvec[0]} {tvec[1]} {tvec[2]} ' +
                              f'{qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]}\n')


def compute_rel_poses(args):
    # Create IBRNet model
    model = DBARFModel(args, load_scheduler=False, load_opt=False, pretrained=False)
    state_dicts = compose_state_dicts(model=model)
    ckpt_manager = CheckPointManager()
    start_step = ckpt_manager.load(config=args, models=state_dicts)
    model.switch_to_eval()
    
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(args.rootdir, args.expname)
    os.makedirs(extra_out_dir, exist_ok=True)

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, start_step))

    test_dataset = dataset_dict[args.eval_dataset](args, 'eval_pose', scenes=args.eval_scenes)
    num_poses = test_dataset.num_poses()

    gt_view_graph_path = os.path.join(out_scene_dir, 'gt_view_graph.g2o')
    gt_view_graph_file = open(gt_view_graph_path, 'w')

    view_graph_path = os.path.join(out_scene_dir, 'pred_view_graph.g2o')
    view_graph_file = open(view_graph_path, 'w')
    R_errors = []
    t_errors = []

    # assert len(test_dataset.train_view_graphs) <= 1, "invalid view graphs number for evaluation"
    # view_graph = test_dataset.train_view_graphs[0]["graph"]
    # mst = networkx.maximum_spanning_tree(view_graph)

    # compute_abs_poses_from_mst(model, test_dataset, mst, num_poses, view_graph_file)
    view_graph_file.close()

    for i in range(num_poses):
        with torch.no_grad():
            # model.switch_to_eval()
            
            data = test_dataset.get_data_one_batch(i)
            src = data['idx']

            gt_camera_pose = data['camera'][..., -16:].reshape(4, 4)
            gt_rot_mat = gt_camera_pose[:3, :3]
            gt_tvec = gt_camera_pose[:3, 3].numpy()
            gt_qvec = Rotation.from_matrix(gt_rot_mat).as_quat() # [qx, qy, qz, qw]
            gt_view_graph_file.write(f'VERTEX_SE3:QUAT {src} {gt_tvec[0]} {gt_tvec[1]} {gt_tvec[2]} ' +
                                     f'{gt_qvec[0]} {gt_qvec[1]} {gt_qvec[2]} {gt_qvec[3]}\n')

            _, pred_rel_poses, __, ___ = model.correct_poses(
                fmaps=None,
                target_image=data['rgb'].unsqueeze(0).cuda(),
                ref_imgs=data['src_rgbs'].unsqueeze(0).cuda(),
                target_camera=data['camera'].unsqueeze(0),
                ref_cameras=data['src_cameras'].unsqueeze(0),
                min_depth=data['depth_range'][0],
                max_depth=data['depth_range'][1],
                scaled_shape=[torch.tensor(data['scaled_shape'][0]), torch.tensor(data['scaled_shape'][1])])

            pred_rel_poses = pred_rel_poses.detach().cpu()
            aligned_pred_poses, poses_gt = align_predicted_training_poses(pred_rel_poses, data)
            pose_error = evaluate_camera_alignment(aligned_pred_poses, poses_gt)
            
            R_errors.append(pose_error[0])
            t_errors.append(pose_error[1])

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
    
    # Writing metrics to json.
    json_obj = json.dumps(metrics_dict, indent=4)
    metrics_dir = os.path.join(out_scene_dir, 'metrics')
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    json_file = os.path.join(metrics_dir, 'rel_pose_metrics.json')
    with open(json_file, 'w') as f:
        f.write(json_obj)

    print(f'[INFO] Predicted results saved to {view_graph_path}')


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    compute_rel_poses(args)
