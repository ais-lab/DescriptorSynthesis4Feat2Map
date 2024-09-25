import os
import shutil
import argparse
from hloc import (extract_features, 
                  match_features, 
                  reconstruction, 
                  visualization, 
                  pairs_from_retrieval)

from pathlib import Path

FIRE_TRAIN_SEQ = [1, 2]
FIRE_TEST_SEQ = [3, 4]
CHESS_TEST_SEQ = [3, 5]
HEADS_TEST_SEQ = [1]
OFFICE_TEST_SEQ = [2, 6, 7, 9]
PUMPKIN_TEST_SEQ = [1, 7]
REDKITCHEN_TEST_SEQ = [3, 4, 6, 12, 14]
STAIRS_TEST_SEQ = [1, 4]

def copy_images(src_dir, dst_dir, fnames):
    for fname in fnames:
        copy_image(src_dir, dst_dir, fname)


def copy_image(src_dir, dst_dir, fname):
    try:
        shutil.copy(os.path.join(src_dir, fname), 
                    os.path.join(dst_dir, fname.replace('/', '-')))
    except Exception as e:
        print(e)

def get_seq_num(name):
    seq_num = name.split('-')[1]
    # print(int(seq_num))
    return seq_num

def filter_netvlad(fname, scene):
    if scene == "chess":
        valid_seq = CHESS_TEST_SEQ
    elif scene == "fire":
        valid_seq = FIRE_TEST_SEQ
    elif scene == "heads":
        valid_seq = HEADS_TEST_SEQ
    elif scene == "office":
        valid_seq = OFFICE_TEST_SEQ
    elif scene == "pumpkin":
        valid_seq = PUMPKIN_TEST_SEQ
    elif scene == "redkitchen":
        valid_seq = REDKITCHEN_TEST_SEQ
    elif scene == "stairs":
        valid_seq = STAIRS_TEST_SEQ
    ret_names = []
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = [x.strip().split(' ') for x in lines]
    for line in lines:
        seq_num = get_seq_num(line[0])
        if seq_num in valid_seq:
            ret_names.append(line)
    

def netvlad_retrieval(images_path, 
                      outputs_path, 
                      query_list, 
                      train_list):
    images = Path(images_path)
    outputs = Path(outputs_path)
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    retrieval_conf = extract_features.confs['netvlad']
    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, 
                       sfm_pairs, 
                       num_matched=10, 
                       query_prefix=None, 
                       query_list=query_list,
                       db_prefix=None,
                       db_list = train_list)

def get_query_list(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [x.strip().replace('/', '-') for x in lines]
    return lines

def reformat_netvladresult(path, output):
    with open(path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split(' ')
        new_line = []
        for part in parts:
            sub_part = part.split('-')
            new_line.append('/'.join([
                        ('-'.join([sub_part[0], sub_part[1]])), 
                        ('-'.join([sub_part[2], sub_part[3]]))
                        ]))
        new_lines.append(' '.join(new_line))
            

    with open(output, 'w') as f:
        for line in new_lines:
            f.write(line+'\n')

def run_hloc_preprocessing(project_root:str, 
                           dataset_root:str,
                           dataset:str, 
                           scene:str, 
                           percentage:int): 

    train_root_path = os.path.join(project_root, 
                                   "data",
                                   f"sfm_hloc_superpoint_{percentage}_evenly")
    list_train_path = os.path.join(train_root_path, 
                                   dataset, scene, 
                                   "list_train.txt")
    dataset_root_path = os.path.join(dataset_root,
                                     dataset)
    list_test_path = os.path.join(dataset_root_path, 
                                  f"{dataset}_sfm_triangulated", 
                                  scene, 
                                  "triangulated", 
                                  "list_test.txt")

    with open(list_train_path, "r") as f:
        train_names = f.readlines()
    train_names = [x.strip() for x in train_names]
    with open(list_test_path, "r") as f:
        test_names = f.readlines()
    test_names = [x.strip() for x in test_names]

    print(train_names)
    
    dst_data_dir = os.path.join(project_root, 
                                "data",
                                "images",
                                "hloc_process",
                                f"{percentage}_evenly")

    dst_data_path = os.path.join(dst_data_dir, dataset, scene)
    scene_path = os.path.join(dataset_root_path, scene)

    copy_images(scene_path, dst_data_path, train_names)
    copy_images(scene_path, dst_data_path, test_names)
    
    output_netvlad = os.path.join(project_root, 
                                  "data",
                                  "netvlad",
                                  f"{percentage}_evenly")
    output_file_path = os.path.join(output_netvlad, dataset, scene)
    query_list = get_query_list(list_test_path)
    train_list = get_query_list(list_train_path)
    netvlad_retrieval(dst_data_path, output_file_path, query_list, train_list)


    reformat_netvladresult(os.path.join(output_file_path, 'pairs-netvlad.txt'), 
                           os.path.join(output_file_path, f'{scene}_top10.txt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hloc preprocessing")
    parser.add_argument('--project_root', type=str, 
                        default="../../")
    parser.add_argument('--dataset_root', type=str, 
                        default="/path/to/dataset")
    parser.add_argument('--dataset', type=str,
                        default='7scenes')
    args = parser.parse_args()
    SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    for scene in SCENES:
        run_hloc_preprocessing(args.project_root,
                             args.dataset_root,
                             args.dataset,
                             scene)

