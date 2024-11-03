import os
from pathlib import Path
from pprint import pformat
import argparse

import torch
import numpy as np
from tqdm import tqdm

import pycolmap
from hloc.utils import viz_3d
from hloc.utils.database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids
from hloc.triangulation import import_features, geometric_verification
from disambiguation import calculate_geodesic_consistency_scores

from scripts import extract_features, match_features, \
    pairs_from_retrieval, reconstruction, filter_matches
from scripts.utils import read_all_keypoints, import_matches, \
    decompose_essential_matrix, read_camera_intrinsics_by_image_id
from ggrt.geometry.rotation import Rotation


def disambiguate_via_geodesic_consistency(database_path,
                                          track_degree,
                                          coverage_thres, alpha, minimal_views,
                                          ds):
    print('disambiguate via geodesic consistency: ')
    calculate_geodesic_consistency_scores.main(database_path, track_degree,
                                               coverage_thres, alpha,
                                               minimal_views, ds)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, default='datasets',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_matches', type=int, default=30,
                        help='Number of image pairs for loc, default: %(default)s')
    parser.add_argument('--disambiguate', action="store_true",
                        help='Enable/Disable disambiguating wrong matches.')
    parser.add_argument('--track_degree', type=int, default=3)
    parser.add_argument('--coverage_thres', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--minimal_views', type=int, default=5)
    parser.add_argument('--ds', type=str,
                        choices=['dict', 'smallarray', 'largearray'],
                        default='largearray')
    parser.add_argument('--filter_type', type=str, choices=[
                        'threshold', 'knn', 'mst_min', 'mst_mean', 'percentile'],
                        default='threshold')
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--percentile', type=float)
    parser.add_argument('--colmap_path', type=Path, default='colmap')
    parser.add_argument('--geometric_verification_type',
                        type=str,
                        choices=['default', 'strict'],
                        default='default')
    parser.add_argument('--recon', action="store_true",
                        help='Indicates whether to reconstruct the scene.')
    parser.add_argument('--visualize', action="store_true",
                        help='Whether to visualize the reconstruction.')
    parser.add_argument('--gpu_idx', type=str, default='0')
    args = parser.parse_args()
    return args


def output_view_graph(num_images, view_graph_path, relative_poses, model=None):
    with open(view_graph_path, 'w') as f:
        # Output nodes.
        if model is None:
            for i in range(num_images):
                # Actually we don't have the absolute camera poses.
                f.write(f'VERTEX_SE3:QUAT {i} 0 0 0 0 0 0 0\n')
        else:
            for i, image in model.images.items():
                f.write(f'VERTEX_SE3:QUAT {i - 1} {image.tvec[0]} {image.tvec[1]} {image.tvec[2]} ' +
                        f'{image.qvec[1]} {image.qvec[2]} {image.qvec[3]} {image.qvec[0]}\n')

        # Output edges.
        for relative_pose in relative_poses:
            relative_pose = relative_pose[0]
            src, dst = int(relative_pose[0]), int(relative_pose[1])
            extrinsics = relative_pose[2:].reshape(3, 4)
            rotation_matrix = extrinsics[:3, :3]
            tvec = extrinsics[:3, -1]
            qvec = Rotation.to_quaternion(rotation_matrix)
            f.write(f'EDGE_SE3:QUAT {src - 1} {dst - 1} {tvec[0]} {tvec[1]} {tvec[2]} ' +
                    f'{qvec[1]} {qvec[2]} {qvec[3]} {qvec[0]}\n')
    
    f.close()


def extract_relative_poses(
    database_path: Path, visualization=True
) -> (np.ndarray, np.ndarray):
    db = COLMAPDatabase.connect(database_path)
    keypoints = read_all_keypoints(db)
    # rows = db.execute("SELECT * FROM two_view_geometries")
    cursor = db.cursor()
    cursor.execute("SELECT * FROM two_view_geometries")
    num_match_pairs = 0
    for row in cursor:
        num_match_pairs += 1

    relative_motions = []
    pbar = tqdm(total=num_match_pairs)

    print(f'num two view geometries: {num_match_pairs}')
    cursor.execute("SELECT * FROM two_view_geometries")
    for rows in cursor:
        pbar.set_description('Extracting relative poses ')
        pbar.update(1)

        # pair_id, shape1, shape2, matches, config, F, E, H, qvec, tvec = next(rows)
        pair_id, shape1, shape2, matches, config, F, E, H, qvec, tvec = rows
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)

        if matches == None:
            continue

        matches = blob_to_array(matches, np.uint32).reshape(shape1, shape2)

        # FIXME(chenyu): in some cases, we need to decompose homography matrix rather than
        # the essential matrix. There should be some strategies to do this inside COLMAP.
        E = blob_to_array(E, np.float64).reshape(3, 3)

        intrinsics1 = read_camera_intrinsics_by_image_id(image_id1, db)
        intrinsics2 = read_camera_intrinsics_by_image_id(image_id2, db)

        relative_motion, points3d = decompose_essential_matrix(
            keypoints[image_id1], keypoints[image_id2], E, matches,
            intrinsics1, intrinsics2)
        
        if not (relative_motion is None or points3d is None):
            # image1, image2, [rotations, translations]
            relative_motion = np.concatenate(
                (np.array([[image_id1, image_id2]]), relative_motion), axis=1)
            relative_motions.append(relative_motion)

    if visualization == True:
        fig = viz_3d.init_figure()
        viz_3d.plot_points(fig, points3d)
        fig.show()

    return relative_motions


def main(args):
    # List the standard configurations available.
    print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

    device = 'cuda:' + args.gpu_idx if torch.cuda.is_available() else 'cpu'

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']
    # feature_conf = extract_features.confs['sift']
    # matcher_conf = match_features.confs['NN-ratio']

    if not os.path.exists(args.outputs):
        os.makedirs(args.outputs)

    # the SfM model we will build.
    sfm_dir = Path(os.path.join(args.outputs, 'sfm_superpoint+superglue'))
    # sfm_dir = Path(os.path.join(args.outputs, 'sfm_sift+nn-ratio'))
    # top-k retrieved by NetVLAD.
    match_pairs = Path(os.path.join(sfm_dir, f'pairs-netvlad.txt'))

    local_features_path = extract_features.main(
        feature_conf, args.dataset_dir, sfm_dir, device=device)

    global_descriptors_path = extract_features.main(
        retrieval_conf, args.dataset_dir, sfm_dir, device=device)
    pairs_from_retrieval.main(
        global_descriptors_path, match_pairs, args.num_matches, device=device)

    matches_path = match_features.main(
        matcher_conf, match_pairs, feature_conf['output'], sfm_dir, device=device)
    
    assert local_features_path.exists(), local_features_path
    assert match_pairs.exists(), match_pairs
    assert matches_path.exists(), matches_path

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database_path = sfm_dir / 'database.db'

    reconstruction.create_empty_db(database_path)
    reconstruction.import_images(
        args.dataset_dir, database_path, camera_mode=pycolmap.CameraMode.AUTO)
    image_ids = reconstruction.get_image_ids(database_path)
    import_features(image_ids, database_path, local_features_path)
    import_matches(
        image_ids, database_path, match_pairs, matches_path,
        min_match_score=None, skip_geometric_verification=False)
    geometric_verification(database_path, match_pairs, verbose=False)

    if args.disambiguate == True:
        print('Disambiguating Wrong Matches.')
        disambiguate_via_geodesic_consistency(database_path,
                                              args.track_degree,
                                              args.coverage_thres, args.alpha,
                                              args.minimal_views, args.ds)

        filtered_db_path = sfm_dir / 'disambig_database.db'
        scores_dir = database_path.parent
        scores_name = f'scores_yan_t{args.track_degree}_c' + \
                      f'{args.coverage_thres}_a{args.alpha}_m{args.minimal_views}.npy'
        filter_matches.main(
            args.colmap_path, sfm_dir, args.filter_type, args.threshold,
            scores_dir, scores_name, args.topk, args.percentile, database_path,
            filtered_db_path, args.geometric_verification_type)
        
        # Overwrite original database path.
        database_path = filtered_db_path

    # The relative rotations and translations are not stored in COLMAP's database.
    relative_poses = extract_relative_poses(database_path, args.visualize)
    view_graph_path = Path(args.outputs) / f'VG_N{len(image_ids)}_M{len(relative_poses)}.g2o'

    model = None
    if args.recon:
        model = reconstruction.main(
            database_path,
            sfm_dir, args.dataset_dir, match_pairs,
            local_features_path, matches_path)
        if args.visualize:
            fig = viz_3d.init_figure()
            viz_3d.plot_reconstruction(fig, model)
            fig.show()

    output_view_graph(len(image_ids), view_graph_path, relative_poses, model)

    print(f'Local features are extracted to: {local_features_path}')
    print(f'matches are saved to: {matches_path}')
    print(f'view graph is saved to {view_graph_path}')
    return view_graph_path, database_path, len(relative_poses)


if __name__ == '__main__':
    args = parse_args()
    main(args)
