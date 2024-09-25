"""
Given a csv file with columns
name, qx, qy, qz, qw, tx, ty, tz
where name is the image name, qx, qy, qz, qw are the quaternion of the rotation, tx, ty, tz are the translation
This script will convert the poses to the format of instant-ngp transform.json for creating the camera_path
"""

import pandas as pd
import math
import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import json
from utils.misc import read_model, convert_bin2txt
from utils.colmap2ngp import qvec2rotmat


def extract_pose_from_colmap_model(model_path, out_path):
    """
    
    """
    _, images, _ = read_model(model_path, ext=".bin")
    names= []
    qxs, qys, qzs, qws= [], [], [], []
    txs, tys, tzs = [], [], []
    for id_, image in images.items():
        q_vecs = image.qvec
        qxs.append(q_vecs[1])
        qys.append(q_vecs[2])
        qzs.append(q_vecs[3])
        qws.append(q_vecs[0])
        t_vecs = image.tvec
        txs.append(t_vecs[0])
        tys.append(t_vecs[1])
        tzs.append(t_vecs[2])
        names.append(image.name.replace("/", "-"))

    df_dict = {"name": names, 
               "qx": qxs, "qy": qys, "qz": qzs, "qw": qws, 
               "tx": txs, "ty": tys, "tz": tzs}
    df = pd.DataFrame(df_dict)
    df.to_csv(out_path, index=False)
    


def convert_poses_to_transform(camera_folder, pose_file, aabb_scale, keep_colmap_coords, out_path):
    df = pd.read_csv(pose_file)
    names = df["name"].values.tolist()
    qx = df["qx"].values
    qy = df["qy"].values
    qz = df["qz"].values
    qw = df["qw"].values
    tx = df["tx"].values
    ty = df["ty"].values
    tz = df["tz"].values
    qvecs = np.concatenate([qw.reshape([-1, 1]), qx.reshape([-1, 1]), qy.reshape([-1, 1]), qz.reshape([-1, 1])], 1)
    tvecs = np.concatenate([tx.reshape([-1, 1]), ty.reshape([-1, 1]), tz.reshape([-1, 1])], 1)
    cameras = {}
    sfm_files = os.listdir(camera_folder)
    if 'cameras.txt' not in sfm_files:
        print("No cameras.txt found! Trying to convert .bin format to .txt")
        convert_bin2txt(camera_folder, camera_folder)
    with open(os.path.join(camera_folder, "cameras.txt"), "r") as f:
    # Your code here
        camera_angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            camera = {}
            camera_id = int(els[0])
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fl_x"] = float(els[4])
            camera["fl_y"] = float(els[4])
            camera["k1"] = 0
            camera["k2"] = 0
            camera["k3"] = 0
            camera["k4"] = 0
            camera["p1"] = 0
            camera["p2"] = 0
            camera["cx"] = camera["w"] / 2
            camera["cy"] = camera["h"] / 2
            camera["is_fisheye"] = False
            if els[1] == "SIMPLE_PINHOLE":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
            elif els[1] == "PINHOLE":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["p1"] = float(els[10])
                camera["p2"] = float(els[11])
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                camera["is_fisheye"] = True
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["k3"] = float(els[10])
                camera["k4"] = float(els[11])
            else:
                print("Unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
            camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
            camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
            camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi

            print(f"camera {camera_id}:\n\tres={camera['w'],camera['h']}\n\tcenter={camera['cx'],camera['cy']}\n\tfocal={camera['fl_x'],camera['fl_y']}\n\tfov={camera['fovx'],camera['fovy']}\n\tk={camera['k1'],camera['k2']} p={camera['p1'],camera['p2']} ")
            cameras[camera_id] = camera

    if len(cameras) == 0:
        print("No cameras found!")
        sys.exit(1)


    i = 0
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    if len(cameras) == 1:
        camera = cameras[camera_id]
        out = {
            "camera_angle_x": camera["camera_angle_x"],
            "camera_angle_y": camera["camera_angle_y"],
            "fl_x": camera["fl_x"],
            "fl_y": camera["fl_y"],
            "k1": camera["k1"],
            "k2": camera["k2"],
            "k3": camera["k3"],
            "k4": camera["k4"],
            "p1": camera["p1"],
            "p2": camera["p2"],
            "is_fisheye": camera["is_fisheye"],
            "cx": camera["cx"],
            "cy": camera["cy"],
            "w": camera["w"],
            "h": camera["h"],
            "aabb_scale": aabb_scale,
            "frames": [],
        }
    else:
        out = {
            "frames": [],
            "aabb_scale": aabb_scale
        }
    up = np.zeros(3)

    for idx, name in enumerate(names):
        qvec = qvecs[idx]
        tvec = tvecs[idx]
        R = qvec2rotmat(qvec)
        t = tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(m)
        
        frame = {"file_path": name, "transform_matrix": c2w}
        out["frames"].append(frame)

    nframes = len(out["frames"])
    if keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat)

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes, "frames")
    print(f"writing {out_path}")
    with open(out_path, "w+") as outfile:
        json.dump(out, outfile, indent=2)


def main():
    parser = argparse.ArgumentParser(description="convert format from colmap to nerf tranform.json")
    parser.add_argument("--project_root", type=str,
                        default="../../")
    parser.add_argument("--dataset", type=str,
                        default="7scenes")
    parser.add_argument("--scene", type=str,
                        default="chess")
    parser.add_argument("--percentage", type=int,
                        default=10)
    parser.add_argument("--topk", type=int,
                        default=2)
    parser.add_argument("--n_samples", type=int,
                        default=10)
    parser.add_argument()
    args = parser.parse_args()
    sfm_sample = os.path.join(args.project_root,
                              "data",
                              f"sfm_hloc_superpoint_{args.percentage}",
                              "args.dataset",
                              "args.scene",
                              "sfm_superpoint+superglue")

    #sfm_full_fire = "/media/hoang/Data/data/hloc/full_2048/fire/sfm_superpoint+superglue/"
    output_path = os.path.join(args.project_root,
                               "data",
                               "pose",
                               f"{args.percentage}", 
                               args.dataset,
                               args.scene)
    ### Test with the synthesis poses
    pose_file = os.path.join(output_path,
            f"synthetic_poses_topk-{args.topk}_samples_{args.n_samples}")
    transform_file = os.path.join(output_path, 
            f"synthetic_transform_topk-{args.topk}_samples-{args.n_samples}.json")
    convert_poses_to_transform(sfm_sample, 
                               pose_file, 
                               1.0, 
                               True, 
                               transform_file)

if __name__ == "__main__":
    main()

