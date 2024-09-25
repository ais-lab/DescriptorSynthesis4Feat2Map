import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import quaternion
from superpoint.lightglue import LightGlue
from utils.camera_pose_vis import CameraPoseVisualizer
from utils.misc import read_model, qvec2rotmat
import math
import random
import h5py
import torch
from scipy.spatial.distance import cdist
# TODO: read the sfm model of the data and get the pose information
# Check for the highest and lowest values of the translation component (x, y, z)
# Visualize a volume that contains the whole scene using those values

GLOBAL_MATCHER = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().cuda()

def get_descriptors(img_name, features):
    data = {}
    with h5py.File(features, 'r') as fd:
        grp = fd[img_name]
        for k, v in grp.items():
            data[k+'0'] = torch.from_numpy(v.__array__()).float()
        # data['image0'] = torch.empty(1, ) + tuple(grp['image_size'])[::-1]
    return data

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotmat2euler(rotmat):
    assert(isRotationMatrix(rotmat))
    sy = math.sqrt(rotmat[0,0] * rotmat[0,0] + rotmat[1,0] * rotmat[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rotmat[2,1], rotmat[2,2])
        y = math.atan2(-rotmat[2,0], sy)
        z = math.atan2(rotmat[1,0], rotmat[0,0])
    else:
        x = math.atan2(-rotmat[1,2], rotmat[1,1])
        y = math.atan2(-rotmat[2,0], sy)
        z = 0
    return np.array([x, y, z])

def find_angle_extremes_diff(poses):
    names = list(poses.keys())
    angle_diffs = {}
    for i, name in enumerate(names):
        if i == 0 or i == len(names) - 1:
            continue
        else:
            previous_name = names[i-1]
            next_name = names[i+1]
            angle_diffs[name] = {"before": np.absolute(rotmat2euler(poses[name]['rotation']) - rotmat2euler(poses[previous_name]['rotation'])),
                                 "after": np.absolute(rotmat2euler(poses[name]['rotation']) - rotmat2euler(poses[next_name]['rotation']))}
            
    x_rot_min = 999999
    y_rot_min = 999999
    z_rot_min = 999999
    x_name, y_name, z_name = "", "", ""
    for k, v in angle_diffs.items():
        if v['before'][0] < x_rot_min:
            x_rot_min = v['before'][0]
            x_name = k
        if v['after'][0] < x_rot_min:
            x_rot_min = v['after'][0]
            x_name = k
        if v['before'][1] < y_rot_min:
            y_rot_min = v['before'][1]
            y_name = k
        if v['after'][1] < y_rot_min:
            y_rot_min = v['after'][1]
            y_name = k
        if v['before'][2] < z_rot_min:
            z_rot_min = v['before'][2]
            z_name = k
        if v['after'][2] < z_rot_min:
            z_rot_min = v['after'][2]
            z_name = k
    return [x_rot_min, y_rot_min, z_rot_min], [x_name, y_name, z_name]


def pose_vis(qvec:np.ndarray, tvec:np.ndarray):
    mat_rotation = quaternion.as_rotation_matrix(np.quaternion(qvec[3], qvec[0], qvec[1], qvec[2]))
    mat_translation = np.array([[tvec[0]], [tvec[1]], [tvec[2]]])
    mat_extrinsic = np.concatenate([np.concatenate([mat_rotation, mat_translation], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    return mat_extrinsic

def get_pose(images):
    img_pose_dict = {}
    for id_, image in images.items():
        img_pose_dict[image.name] = {"qvec": image.qvec,
                                    "rotation":qvec2rotmat(image.qvec), 
                                     "translation":image.tvec,
                                     "pose_vis": pose_vis(image.qvec, image.tvec)}
    return img_pose_dict


def get_list_test(fname):
    with open(fname, 'r') as f:
        list_test = f.read().rstrip().split('\n')
    return list_test



def get_test_poses(images, test_list):
    test_pose_dict = {}
    for id_, image in images.items():
        if image.name in test_list:
            test_pose_dict[image.name] = {"rotation":qvec2rotmat(image.qvec), 
                                          "translation":image.tvec,
                                          "pose_vis": pose_vis(image.qvec, image.tvec)}
    return test_pose_dict


def find_corners(pose_t):
    xs = pose_t[:,0]
    ys = pose_t[:,1]
    zs = pose_t[:,2]
    min_x = np.min(xs)
    max_x = np.max(xs)
    min_y = np.min(ys)
    max_y = np.max(ys)
    min_z = np.min(zs)
    max_z = np.max(zs)
    corners = {"min_x":min_x, "max_x":max_x, 
               "min_y":min_y, "max_y":max_y,
               "min_z":min_z, "max_z":max_z}
    vol_point_1 = [min_x, min_y, min_z]
    vol_point_2 = [max_x, min_y, min_z]
    vol_point_3 = [max_x, max_y, min_z]
    vol_point_4 = [min_x, max_y, min_z]

    vol_point_5 = [min_x, min_y, max_z]
    vol_point_6 = [max_x, min_y, max_z]
    vol_point_7 = [max_x, max_y, max_z]
    vol_point_8 = [min_x, max_y, max_z]

    return corners, np.array([vol_point_1, vol_point_2, vol_point_3, vol_point_4, 
            vol_point_5, vol_point_6, vol_point_7, vol_point_8])


def calculate_grid(points):
    assert points.shape == (3, 3)
    p0, p1, p2 = points 
    x0, y0, z0 = p0 
    x1, y1, z1 = p1 
    x2, y2, z2 = p2 

    ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]
    
    u_cross_v = [uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx]
    point = np.array(p0)
    normal = np.array(u_cross_v)

    d = -point.dot(normal)

    xx, yy = np.meshgrid(range(10), range(10))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

    return xx, yy, z


def sample_volume(corners, vol_points=None, t_sampling_num=10):
    distance_x = math.sqrt(corners['max_x'] - corners['min_x'])
    distance_y = math.sqrt(corners['max_y'] - corners['min_y'])
    distance_z = math.sqrt(corners['max_z'] - corners['min_z'])
    assert t_sampling_num != 0
    sampling_dist_x = distance_x / t_sampling_num
    sampling_dist_y = distance_y / t_sampling_num
    sampling_dist_z = distance_z / t_sampling_num

    print("sampling_dist_x: ", sampling_dist_x)
    print("sampling_dist_y: ", sampling_dist_y) 
    print("sampling_dist_z: ", sampling_dist_z)

    xs = np.arange(corners['min_x'], corners['max_x'], sampling_dist_x).tolist()
    ys = np.arange(corners['min_y'], corners['max_y'], sampling_dist_y).tolist()
    zs = np.arange(corners['min_z'], corners['max_z'], sampling_dist_z).tolist()

    poses = {}
    poses_for_vis = []
    counter = 0
    for x in xs:
        for y in ys:
            for z in zs:
                poses['synthetic_t_{}'.format(counter)] = [x, y, z]
                poses_for_vis.append([x, y, z])
    return poses, poses_for_vis


def relative_distance(pose_t, poses_t, names, top_k):
    l2_norms = np.linalg.norm(poses_t - pose_t, axis=1)
    top_k_indices = np.argpartition(l2_norms, top_k)[:top_k]
    close_pose_names = [names[i] for i in top_k_indices]
    #print("Closest poses to {} are: {}".format(name, close_pose_names))
    #print(top_k_indices)
    return close_pose_names

def angle_X(degree):
    tmp = np.pi/180.0
    return np.array([[1, 0, 0],
                     [0, np.cos(degree*tmp), -np.sin(degree*tmp)],
                     [0, np.sin(degree*tmp), np.cos(degree*tmp)]])

def angle_Y(degree):
    tmp = np.pi/180.0
    return np.array([[np.cos(degree*tmp), 0, np.sin(degree*tmp)],
                     [0, 1, 0],
                     [-np.sin(degree*tmp), 0, np.cos(degree*tmp)]])

def angle_Z(degree):
    tmp = np.pi/180.0
    return np.array([[np.cos(degree*tmp), -np.sin(degree*tmp), 0],
                     [np.sin(degree*tmp), np.cos(degree*tmp), 0],
                     [0, 0, 1]])

def rotate(pose, a_x, a_y, a_z):
    new_pose = angle_X(a_x) @ angle_Y(a_y) @ angle_Z(a_z) @ pose
    return new_pose

def sample_angle(pose:np.ndarray, range_x:list, range_y:list, range_z:list, sampling_num:int, distance_threshold=None):
    # if the synthetic translation is close to the real pose translation
    # then we increase the amount of available rotations and sampling more
    sampling_diff_x = (range_x[1] - range_x[0]) / sampling_num
    sampling_diff_y = (range_y[1] - range_y[0]) / sampling_num
    sampling_diff_z = (range_z[1] - range_z[0]) / sampling_num
    x_angles = np.arange(range_x[0], range_x[1], sampling_diff_x)
    y_angles = np.arange(range_y[0], range_y[1], sampling_diff_y)
    z_angles = np.arange(range_z[0], range_z[1], sampling_diff_z)
    avaialble_rots = []
    for x in x_angles:
        for y in y_angles:
            for z in z_angles:
                avaialble_rots.append([x, y, z])
    new_poses = []
    for rot in avaialble_rots:
        new_poses.append(rotate(pose, rot[0], rot[1], rot[2]))
    return new_poses


def save_synthetic_poses(synthetic_poses, output_dir):
    names = list(synthetic_poses.keys())
    ref_name = []
    ref_name2 = []
    ref_matches = []
    qw, qx, qy, qz, tx, ty, tz = [], [], [], [], [], [], []
    for name in names:
        qw.append(synthetic_poses[name]['qvec'][0])
        qx.append(synthetic_poses[name]['qvec'][1])
        qy.append(synthetic_poses[name]['qvec'][2])
        qz.append(synthetic_poses[name]['qvec'][3])
        tx.append(synthetic_poses[name]['tvec'][0])
        ty.append(synthetic_poses[name]['tvec'][1])
        tz.append(synthetic_poses[name]['tvec'][2])
        ref_name.append(synthetic_poses[name]['ref_name'])
        ref_name2.append(synthetic_poses[name]['ref_name2'])
        ref_matches.append(synthetic_poses[name]['ref_matches'])
    df = pd.DataFrame({"name": names,
                       "qw": qw, "qx": qx, "qy": qy, "qz": qz, 
                       "tx": tx, "ty": ty, "tz": tz,
                       "ref_name": ref_name, "ref_name2": ref_name2,
                       "ref_matches": ref_matches})
    df.to_csv(os.path.join(output_dir), index=False)


def two_point_sampling(t_1, t_2, num_points):
    x1, y1, z1 = t_1
    x2, y2, z2 = t_2
    try:
        x_coords = np.arange(min(x1, x2), max(x1, x2), (max(x1, x2) - min(x1, x2)) / num_points)
        y_coords = np.arange(min(y1, y2), max(y1, y2), (max(y1, y2) - min(y1, y2)) / num_points)
        z_coords = np.arange(min(z1, z2), max(z1, z2), (max(z1, z2) - min(z1, z2)) / num_points)
        x_coords = x_coords[:num_points]
        y_coords = y_coords[:num_points]
        z_coords = z_coords[:num_points]
        sampled_points = np.column_stack([x_coords, y_coords, z_coords])
        return sampled_points
    except Exception as e:
        print(t_1, t_2, num_points)
        return None


def create_volume_around_point(X, d, num_points):
    # Generate evenly spaced coordinates within the volume
    x_vals = np.linspace(X[0] - d/2, X[0] + d/2, num_points)
    y_vals = np.linspace(X[1] - d/2, X[1] + d/2, num_points)
    z_vals = np.linspace(X[2] - d/2, X[2] + d/2, num_points)

    # Create a meshgrid from the coordinates
    x, y, z = np.meshgrid(x_vals, y_vals, z_vals)

    # Stack the coordinates to form the sampled points
    sampled_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Remove the point X from the sampled points
    sampled_points = np.array([point for point in sampled_points if not np.array_equal(point, X)])

    return sampled_points

def uniform_angle(pose, angles, num_points):
    assert len(angles) == 2
    ret = []
    for i in range(num_points):
        ax = random.uniform(angles[0], angles[1])
        ay = random.uniform(angles[0], angles[1])
        az = random.uniform(angles[0], angles[1])
        ret.append(rotate(pose, ax, ay, az))
    return ret

def slerp(q1, q2, t):
    # t determine the weight, t closer to 0 result in an output closer to q1 and vice versa
    # Compute the cosine of the angle between the two vectors.
    dot = np.dot(q1, q2)

    # If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
    # the shorter path. So we'll reverse one quaternion's direction to get the correct slerp.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Clamp the dot product to stay within the domain of acos()
    # This also ensures that the result is a unit quaternion (necessary for the quaternion to represent a rotation)
    dot = np.clip(dot, -1.0, 1.0)

    # Calculate the angle between the quaternions
    theta_0 = np.arccos(dot)  # theta_0 = angle between input vectors
    theta = theta_0 * t       # theta = angle between v0 and result

    # Calculate the weights for q1 and q2
    q1_weight = np.cos(theta) - dot * np.sin(theta)
    q2_weight = np.sin(theta)

    # The quaternion to return
    return (q1 * q1_weight) + (q2 * q2_weight)


def create_pairs(distances, k):
    pairs = []

    for i, distances_row in enumerate(distances):
        # Exclude the point itself and get the indices of the k closest points
        closest_indices = np.argsort(distances_row)[1:k+1]

        # Create pairs of the current point with its k closest neighbors
        current_point = i
        for close_index in closest_indices:
            pairs.append((current_point, close_index))

    return pairs

def ref_pose_matching(ref_1, ref_2, matcher):
    matcher_data = {
        "image0": {"keypoints": ref_1['keypoints0'].unsqueeze(0).cuda(),
                   "descriptors": ref_1['descriptors0'].unsqueeze(0).transpose(1, 2).cuda()},
        "image1": {"keypoints": ref_2['keypoints0'].unsqueeze(0).cuda(),
                   "descriptors": ref_2['descriptors0'].unsqueeze(0).transpose(1, 2).cuda()}
    }
    matcher_out = matcher(matcher_data)
    matches = matcher_out['matches'][0].cpu().detach().numpy()
    len_matches = matches[:, 0]
    len_matches = len([i for i in len_matches if i != -1])
    return len_matches

def generate_camera_poses_2(training_images, 
                            output_dir, 
                            feature_path, 
                            top_k_ref=5, 
                            num_t_samples=10,
                            save_pose=False):
    synthetic_poses = {}
    training_poses = get_pose(training_images)
    training_names = list(training_poses.keys())
    training_translation = [v['translation'] for k, v in training_poses.items()]
    #extremes, corners = find_corners(np.array(training_translation))

    for img_name, pose in training_poses.items():
        first_ref = get_descriptors(img_name, os.path.join(feature_path,"feats-superpoint-n4096-r1024.h5" ))
        top_closest_names = relative_distance(np.array(pose['translation']), 
                                              training_translation, 
                                              training_names, 
                                              top_k=top_k_ref)
        #top_closest_names = top_closest_names[1:]
        for name in top_closest_names:
            if name == img_name:
                continue
            second_ref = get_descriptors(name, os.path.join(feature_path,"feats-superpoint-n4096-r1024.h5" ))
            num_matches = ref_pose_matching(first_ref, second_ref, GLOBAL_MATCHER)
            # if num_matches < 400:
            #     continue
            t_1 = pose['translation']
            q_1 = pose['qvec']
            q_1 = np.array([q_1[1], q_1[2], q_1[3], q_1[0]])
            t_2 = training_poses[name]['translation']
            q_2 = training_poses[name]['qvec']
            q_2 = np.array([q_2[1], q_2[2], q_2[3], q_2[0]])
            if set(t_1) == set(t_2):
                continue
            sampled_points = two_point_sampling(t_1, t_2, num_t_samples)
            if sampled_points is None:
                continue
            # angle sample using slerp
            for i, point in enumerate(sampled_points):
                t = i / num_t_samples
                q = slerp(q_1, q_2, t)
                q = np.array([q[3], q[0], q[1], q[2]]).tolist()
                synthetic_poses['synthetic_{}from_{}_{}'.format(i, img_name, name)] = {'qvec': q, 
                                                                              'tvec': point, 
                                                                              'ref_name': img_name, 
                                                                              'ref_name2':name, 
                                                                              'ref_matches': num_matches}


    if save_pose:
        save_synthetic_poses(synthetic_poses, output_dir)
    return synthetic_poses




def main():
    parser = argparse.ArgumentParser(description="creating novel pose")
    parser.add_argument('--dataset_root', type=str,
                        default="/path/to/dataset")
    parser.add_argument('--project_root', type=str,
                        default="../../")
    parser.add_argument('--dataset', type=str,
                        default='7scenes')
    parser.add_argument('--scene', type=str,
                        default='chess')
    parser.add_argument('--sfm_hloc_dir', type=str,
                        default="/path/to/hloc/sfm/output")
    parser.add_argument('--percentage', type=int,
                        default=10)
    parser.add_argument('--topk_ref', type=int,
                        default=2)
    parser.add_argument('--num_samples', type=int,
                        default=10)

    args = parser.parse_args()

    scene = "stairs"
    hloc_sfm_root = os.path.join(args.project_root, 
                                 "data",
                                 f"sfm_hloc_superpoint_{args.percentage}_evenly",
                                 args.dataset)
    superpoint_folder_name = "sfm_superpoint+superglue"
    feature_path = os.path.join(hloc_sfm_root, 
                                args.scene)

    _, train_images, _ = read_model(
            os.path.join(hloc_sfm_root, 
                         scene, 
                         superpoint_folder_name), 
            ext='.txt')
    output_synthetic = "/media/hoang/Data/D2S--/data/poses/5_evenly/7scenes/stairs/synthetic_poses_topk-2_samples-10.csv"
    output_synthetic = os.path.join(args.project_root,
                                    "data",
                                    "poses",
                                    f"{args.percentage}_evenly",
                                    args.dataset,
                                    args.scene,
                                    f"synthetic_poses_topk-{args.topk_ref}_samples-{args.num_samples}")

    synthetic_pose = generate_camera_poses_2(train_images, 
                                             output_synthetic,
                                             feature_path, 
                                             top_k_ref=args.topk_ref,
                                             num_t_samples=args.num_samples,
                                             save_pose= True)


if __name__ == "__main__":
    main()
