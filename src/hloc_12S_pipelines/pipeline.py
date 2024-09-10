import logging
from pathlib import Path
import argparse

from .utils import create_reference_sfm
from .create_gt_sfm import correct_sfm_with_gt_depth
from ..Cambridge.utils import create_query_list_with_intrinsics, evaluate
from ... import extract_features, match_features, pairs_from_covisibility
from ... import triangulation, localize_sfm, pairs_from_retrieval
import pandas as pd
import numpy as np
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SCENES = ["apt1_kitchen", "apt1_living", "apt2_bed", "apt2_kitchen", "apt2_living", "apt2_luke", "office1_gates362", \
          "office1_gates381", "office1_lounge", "office1_manolis", "office2_5a", "office2_5b"] 
# SCENES = ["apt1_kitchen"]
import time 

# SCENES = ["apt1_kitchen"]
def run_scene(images, gt_dir, retrieval, outputs, results, num_covis,
              use_dense_depth, depth_dir=None):
    outputs.mkdir(exist_ok=True, parents=True)
    s_time = time.time()
    ref_sfm_sift = outputs / 'sfm_sift'
    ref_sfm = outputs / 'sfm_superpoint+superglue'
    query_list = outputs / 'query_list_with_intrinsics.txt'
    num_loc = 10
    loc_pairs = outputs / f'pairs-query-netvlad{num_loc}.txt'

    feature_conf = {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 2048,
            'keypoint_threshold':0.0,
        },
        'preprocessing': {
            'globs': ['*.color.jpg'],
            'grayscale': True,
            'resize_max': 648,
        },
    }
    matcher_conf = match_features.confs['superglue']
    matcher_conf['model']['sinkhorn_iterations'] = 5

    test_list = gt_dir / 'list_test.txt'
    # import pdb; pdb.set_trace()
    # create_reference_sfm(gt_dir, ref_sfm_sift, test_list)
    # create_query_list_with_intrinsics(gt_dir, query_list, test_list)


    ### New (for using netvlad to retrieve nearest images)
    retrieval_conf = extract_features.confs['netvlad']
    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    # for test_data.
    with open(test_list, 'r') as f:
        query_seqs = {q for q in f.read().rstrip().split('\n')}
    pairs_from_retrieval.main( global_descriptors, loc_pairs, num_loc, db_model=ref_sfm_sift, query_prefix=query_seqs)
    ## ---- 

    features = extract_features.main(
            feature_conf, images, outputs, as_half=True)

    sfm_pairs = outputs / f'pairs-db-covis{num_covis}.txt'
    pairs_from_covisibility.main(
            ref_sfm_sift, sfm_pairs, num_matched=num_covis)
    sfm_matches = match_features.main(
            matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    if not (use_dense_depth and ref_sfm.exists()):
        triangulation.main(
            ref_sfm, ref_sfm_sift,
            images,
            sfm_pairs,
            features,
            sfm_matches)
            # colmap_path='colmap')

    if use_dense_depth:
        assert depth_dir is not None
        ref_sfm_fix = outputs / 'sfm_superpoint+superglue+depth'
        correct_sfm_with_gt_depth(ref_sfm, depth_dir, ref_sfm_fix)
        ref_sfm = ref_sfm_fix

    # Thuan............
    # ref_sfm_sift_original = gt_dir
    # retrieval = outputs / "retrieval.txt"
    # pairs_from_covisibility.main(
    #         ref_sfm_sift_original, retrieval, num_matched=100)
    # create_retrieval(retrieval, test_list, 30)
    # Thuan............

    retrieval = loc_pairs
    loc_matches = match_features.main(
        matcher_conf, retrieval, feature_conf['output'], outputs)
    # exit()
    localize_sfm.main(
        ref_sfm,
        query_list,
        retrieval,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
        prepend_camera_name=True)
    

def create_retrieval(file_path, test_file, num_matched):
    file = pd.read_csv(file_path, header = None, sep = " ")
    test_list = pd.read_csv(test_file, header = None, sep = " ")
    test_list = list(test_list.iloc[:,0])
    col1 = []
    col2 = [] 
    for i in range(len(file)):
        if file.iloc[i,0] in test_list:
            if not (file.iloc[i,1] in test_list):
                col1.append(file.iloc[i,0])
                col2.append(file.iloc[i,1])
        else:
            break
    assert len(col1) == len(col2)
    tmp = np.zeros((len(col1), 2))
    tmp = pd.DataFrame(tmp)
    tmp.iloc[:,0] = col1
    tmp.iloc[:,1] = col2
    tmp.to_csv(file_path, header = False, sep = " ", index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', default=SCENES, choices=SCENES, nargs='+')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dataset', type=Path, default='datasets/12scenes',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs/12scenes',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--use_dense_depth', action='store_true')
    parser.add_argument('--num_covis', type=int, default=30,
                        help='Number of image pairs for SfM, default: %(default)s')
    args = parser.parse_args()

    gt_dirs = args.dataset / '12scenes_sfm_triangulated/{scene}/'
    retrieval_dirs = args.dataset / '12scenes_densevlad_retrieval_top_10'

    all_results = {}
    for scene in args.scenes:
        logging.info(f'Working on scene "{scene}".')
        results = args.outputs / scene / 'results_{}.txt'.format(
            "dense" if args.use_dense_depth else "sparse")
        if args.overwrite or not results.exists():
            run_scene(
                args.dataset / "12scenes_sfm_triangulated/{}/".format(scene),
                Path(str(gt_dirs).format(scene=scene)),
                retrieval_dirs / f'{scene}_top10.txt',
                args.outputs / scene,
                results,
                args.num_covis,
                args.use_dense_depth,
                depth_dir= f'/media/thuan/8tb/HLOC_DATA/dataset/12scenes/depth/12scenes_{scene}/train/depth')
        all_results[scene] = results
    # exit()
    for scene in args.scenes:
        logging.info(f'Evaluate scene "{scene}".')
        gt_dir = Path(str(gt_dirs).format(scene=scene))
        evaluate(gt_dir, all_results[scene], gt_dir / 'list_test.txt')
