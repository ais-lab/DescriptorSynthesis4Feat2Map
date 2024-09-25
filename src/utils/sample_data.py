from utils import read_model
import os
import pandas as pd
import argparse
import glob
import numpy as np
import re
import shutil
import pathlib

np.random.seed(2024)


DATASET = ["7scenes"]
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
SCENES_CAMBRIDGE = ["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
SCENES_12 = ['apt1_kitchen', 
            'apt1_living', 
            'apt2_bed', 
            'apt2_kitchen', 
            'apt2_living', 
            'apt2_luke', 
            'office1_gates362', 
            'office1_gates381', 
            'office1_lounge', 
            'office1_manolis', 
            'office2_5a', 
            'office2_5b']
# SCENES_12 = ['apt2_bed', 'office2_5a', 'office2_5b']
# TODO: Should change this into reading a csv files of training files and \
# creating the list_test.txt from that. Currently using the read_features function
# not the good way to do it.

def read_features(data_dir):
    readme = pd.read_csv(os.path.join(data_dir, "readme.txt"), header=None, sep=" ")
    file_names = readme.iloc[:, 0].values
    h5_train_files = readme.iloc[:, 1].values
    h5_label_files = readme.iloc[:, 2].values
    poses = readme.iloc[:, 3:10].values
    cameras = readme.iloc[:, 10:].values

    return {"file_names": file_names,
            "h5_train_files": h5_train_files,
            "h5_label_files": h5_label_files,
            "poses": poses,
            "cameras": cameras}


def create_test_list(list_train, colmap_images, output_path=""):
    list_test = []
    for id_, image in colmap_images.items():
        if image.name not in list_train:
            list_test.append(image.name)

    assert len(list_test) != 0
    print(os.path.join(output_path, "list_test.txt"))
    with open(os.path.join(output_path, "list_test.txt"), 'w') as f:
        for line in list_test:
            f.write(f"{line}\n")


def check_stats(cameras, images, points3D):
    img_names = []
    for id_, image in images.items():
        img_names.append(image.name)
    print("Number of images", len(img_names))
    return img_names


def evenly_array(dataset_length, sampling_percentage):
    num_samples = int(dataset_length*sampling_percentage/100)
    space_between = int(dataset_length/num_samples)
    index = 0
    ret_indices = []
    for i in range(num_samples):
        ret_indices.append(f'{index:06d}')
        index += space_between
    return ret_indices 


def evenly_sample_7scenes(dataset_path, output_path, scene_sampling_percentage=1):
    train_split_file = os.path.join(dataset_path, 'TrainSplit.txt')
    test_split_file = os.path.join(dataset_path, 'TestSplit.txt')
    with open(train_split_file, 'r') as f:
        trains = f.readlines()
    with open(test_split_file, 'r') as f:
        tests = f.readlines()
    train_seq_num = []
    total_seq_num = []

    for line in trains:
        train_seq_num.append(int(re.findall(r'\d+', line)[0]))
        total_seq_num.append(int(re.findall(r'\d+', line)[0]))
    for line in tests:
        total_seq_num.append(int(re.findall(r'\d+', line)[0]))
    train_fnames, total_fnames = [], []
    train_samples, test_samples = [], []
    for seq in total_seq_num:
        seq_name = f'seq-{seq:02d}'
        fnames = os.listdir(os.path.join(dataset_path, seq_name))
        for fname in fnames:
            if fname.endswith('.color.png'):
                total_fnames.append(f'{seq_name}/{fname}')
        if seq in train_seq_num:
            for fname in fnames:
                if fname.endswith('.color.png'):
                    train_fnames.append(f'{seq_name}/{fname}')
            sample_indices = evenly_array(len(fnames)/3, scene_sampling_percentage)
            for i in sample_indices:
                train_samples.append(f'{seq_name}/frame-{i}.color.png')
        assert len(set(train_samples)) == len(train_samples)
    for name in total_fnames:
        if name not in train_samples:
            test_samples.append(name)
    
    # save list of train and test files
    with open(os.path.join(output_path, 'list_test.txt'), 'w') as f:
        for line in test_samples:
            f.write(f'{line}\n')
    with open(os.path.join(output_path, 'list_train.txt'), 'w') as f:
        for line in train_samples:
            f.write(f'{line}\n')

        
def filter_images(dataset_dir, output_dir, list_train):
    """Copy images, depth and pose files to output_dir"""
    with open(list_train, 'r') as f:
        train_files = f.readlines()
    train_files = [line.strip('\n') for line in train_files]
    for fname in train_files:
        seq_name = fname.split('/')[0]
        file = fname.split('/')[1]
        src = os.path.join(dataset_dir, fname)
        src_depth = os.path.join(dataset_dir, fname.replace('.color.png', '.depth.png'))
        src_pose = os.path.join(dataset_dir, fname.replace('.color.png', '.pose.txt'))
        seq_path = pathlib.Path(os.path.join(output_dir, seq_name))
        seq_path.mkdir(parents=True, exist_ok=True)
        dst = os.path.join(output_dir, seq_name, file)
        dst_depth = os.path.join(output_dir, seq_name, file.replace('.color.png', '.depth.png'))
        dst_pose = os.path.join(output_dir, seq_name, file.replace('.color.png', '.pose.txt'))
        shutil.copy(src, dst)
        shutil.copy(src_depth, dst_depth)
        shutil.copy(src_pose, dst_pose)

def evenly_sample_12scenes(dataset_path, output_path, sampling_percentage=5, sampling_number=None):
    # import pdb; pdb.set_trace()
    # img_folder = os.path.join(dataset_path, 'data')
    # img_fnames = os.listdir(img_folder)
    img_fnames = []
    cameras, images, _ = read_model(dataset_path)
    for id, img in images.items():
        img_name = img.name.split('/')[1]
        img_fnames.append(img_name)
    list_test_path = os.path.join(dataset_path, 'list_test_bak.txt')
    with open(list_test_path, 'r') as f:
        list_test = f.readlines()
    list_test = [l.strip().split('/')[1] for l in list_test]
    list_train = []
    for name in img_fnames:
        if name not in list_test:
            list_train.append(name)
    
    def evenly_array_12scenes(dataset_length, sampling_percentage, index, sampling_number=None):
        print("dataset length", dataset_length)
        if sampling_number == None:
            num_samples = int(dataset_length*sampling_percentage/100)
        else:
            assert isinstance(sampling_number, int)
            num_samples = sampling_number
        space_between = int(dataset_length/num_samples)
        # index = 1
        ret_indices = []
        for i in range(num_samples):
            ret_indices.append(f'{index:06d}')
            index += space_between
        return ret_indices

    def get_train_indices(fnames):
        indices = []
        for fname in fnames:
            frame_num = fname.split('.')[0]
            indices.append(int(frame_num.split('-')[1]))
        return indices
    train_indices = sorted(get_train_indices(list_train))
    sample_indices = evenly_array_12scenes(len(list_train), sampling_percentage, index=train_indices[0], sampling_number=sampling_number)
    sample_list_train = []
    for idx in sample_indices:
        sample_list_train.append(f'frame-{idx}.color.jpg')
    for name in list_train:
        if name not in sample_list_train:
            list_test.append(name)

    # print(sample_list_train[:5])
    # print(list_test[:5])
    # exit()
    
    with open(os.path.join(output_path, 'list_test.txt'), 'w') as f:
        for line in list_test:
            f.write(f'data/{line}\n')
    
    with open(os.path.join(output_path, 'list_train.txt'), 'w') as f:
        for line in sample_list_train:
            f.write(f'data/{line}\n')
    


    

def main():
    #parser = argparse.ArgumentParser(description="Create list_test for experiments")
    #parser.add_argument("--sfm", type=str, help="Path to sfm model")
    #parser.add_argument("--dataset", type=str, help="Dataset name")
    #parser.add_argument("--scene", type=str, help="Scene name")
    #parser.add_argument("--output", type=str, help="Output path")
    #parser.add_argument("--filtered", type=str, help="Path to filtered images")
    #args = parser.parse_args()

    #args.sfm = "/home/hoang/Hoang_workspace/dataset/datasets/7scenes/7scenes_sfm_triangulated/"
    #args.dataset = "7scenes"
    #args.scene = "heads"
    #args.filtered = "/home/hoang/Hoang_workspace/dataset/feat2map_dataset_10percentbak/7scenes/"
    #args.output = "/media/hoang/Data/D2S--/data/sfm_hloc_superpoint_10/"

    #assert args.dataset in DATASET
    #assert args.scene in SCENES
    #cameras, images, points3D = read_model(os.path.join(args.sfm, args.scene, "triangulated"))


    #training_data = read_features(os.path.join(os.path.join(args.filtered, args.scene), "train"))
    #list_train = training_data["file_names"]
    #list_test = create_test_list(list_train, images, os.path.join(args.output, args.dataset, args.scene))

    parser = argparse.ArgumentParser(description="Sample data in uniform manner")
    parser.add_argument("--data", type=str, help="data path")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--scene", type=str, help="scene name")
    parser.add_argument("--output_path", type=str, help="output_dir")
    parser.add_argument("--p", type=int, help="sampling percentage")
    args = parser.parse_args()

    args.data = "/home/hoang/Hoang_workspace/dataset/datasets/"
    # args.data = "/media/hoang/Data/data/datasets"
    # args.dataset = "7scenes"
    # args.dataset = "cambridge"
    args.dataset = "12scenes"
    # args.scene = "redkitchen"
    # args.scene = "chess"
    
    args.output_path = "/media/hoang/Data/D2S--/data/sfm_hloc_superpoint_2_evenly/"
    args.p = 2
    # scene_path = os.path.join(args.data, args.dataset, args.scene)
    # cambridge_colmap = os.path.join(args.data, args.dataset, 'CambridgeLandmarks_Colmap_Retriangulated_1024px', args.scene)

    #uniform_sample_7scenes(scene_path, os.path.join(args.output_path, args.dataset, args.scene), args.p)
    # evenly_sample_7scenes(scene_path, os.path.join(args.output_path, args.dataset, args.scene), args.p)

    # for scene in SCENES:
    #     scene_path = os.path.join(args.data, args.dataset, scene)
    #     evenly_sample_7scenes(scene_path, os.path.join(args.output_path, args.dataset, scene), args.p)

    # evenly_sample_Cambridge(scene_path, cambridge_colmap, os.path.join(args.output_path, args.dataset, args.scene), args.p)
    dataset_path = "/media/hoang/Data/data/12scenes/"
    for scene in SCENES_12:
        scene_path = os.path.join(dataset_path, args.dataset, "12scenes_sfm_triangulated", scene)
        output = "/media/hoang/Data/D2S--/data/sfm_hloc_superpoint_200_evenly"
        evenly_sample_12scenes(scene_path, os.path.join(output, args.dataset, scene), args.p, 200)

if __name__ == "__main__":
    main()
    




    
