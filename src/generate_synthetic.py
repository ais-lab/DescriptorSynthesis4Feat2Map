from tqdm import tqdm
import numpy as np 
import os
from proj_frame_select import generate_proj_frame
from check_synth_proj import *
from superpoint.detector import single_img_processing, SuperPointDetector
from utils.utils import read_model
import pandas as pd
from superpoint.lightglue import LightGlue
import torch


GLOBAL_DETECTOR = SuperPointDetector()
GLOBAL_MATCHER = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()

def numpy2str(array):
    out = ""
    for i in array:
        out += str(i) 
        out += " "
    return out.rstrip(out[-1])


def text_pose(tvec, qvec):
    """
    Generate a string list of 7DoF camera pose
    Parameters
    ----------
    tvec : numpy
        Translation vector.
    qvec : numpy
        Rotational vector.
    """
    out = numpy2str(tvec) + " " + numpy2str(qvec)
    return out

def get_refimg_fnames(colmap_ref_images):
    """
    Return a list of reference image names
    """
    img_list = []
    for id_, image in colmap_ref_images.items():
        img_list.append(image.name)
    return img_list

def get_synthetic_fnames(synthetic_img_folder):
    """
    Return a list of synthetic image names given the synthetic image folder 
    """
    img_list = []
    for name in os.listdir(synthetic_img_folder):
        if 'jpg' in name or 'png' in name:
            img_list.append(name)
    return img_list


def make_out_dir(out_dir, mode):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir = os.path.join(out_dir, mode)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    h5 = os.path.join(out_dir, "h5")
    if not os.path.exists(h5):
        os.makedirs(h5)
    return out_dir


def filter_out_of_bound(data_3D:dict, image_size):

    assert "xys" in data_3D.keys()
    assert isinstance(data_3D["xys"], np.ndarray)
    out_of_bound_indices = []
    for idx, coord in enumerate(data_3D["xys"]):
        if coord[0] < 0 or coord[0] > image_size[1] or coord[1] < 0 or coord[1] > image_size[0]:
            # print("out of bound", coord)
            # print("image size", image_size)
            out_of_bound_indices.append(idx)
    #print("number of out of bound points", len(out_of_bound_indices))
    
    for idx, coord_3d in enumerate(data_3D["p3Ds"]):
        if idx in out_of_bound_indices:
            # TODO: also set the ids = -1
            data_3D["p3Ds"][idx] = np.array([0, 0, 0])
            data_3D["errors"][idx] = 0
            data_3D["p3D_ids"][idx] = -1
    return data_3D
    
        


def generate_synthetic_data(sfm_ref_path, sfm_full_path, synthetic_img_folder, out_dir):
    mode = "synthetic"
    counter = 0
    # Read the colmap reference dataset
    ref_cameras, ref_images, ref_points3D = read_model(sfm_ref_path)
    ref_img_fnames = get_refimg_fnames(ref_images)
    # Read the colmap full dataset 
    _, full_images, _ = read_model(sfm_full_path)

    synthetic_img_fnames = get_synthetic_fnames(synthetic_img_folder)
    reprojection_dict = generate_proj_frame(ref_images, full_images)
    output = make_out_dir(out_dir, mode)

    for id_, image in tqdm(full_images.items()):
        image_name = image.name
        if image_name in ref_img_fnames:
            continue
        synthetic_name = image_name.replace("/", "-")
        assert synthetic_name in synthetic_img_fnames

        s_name = "train_" + str(counter) + ".h5"
        s_name3d = "label_" + str(counter) + ".h5"
        pose = text_pose(image.tvec, image.qvec)
        pose_for_reproj = get_pose(image)

        camera = camera2txt(ref_cameras[1])

        assert reprojection_dict[image_name] in ref_img_fnames
        ref_proj_frame = get_image(ref_images, reprojection_dict[image_name])
        _, ref_proj_points3D, errors = get_keypoints(ref_proj_frame, ref_points3D)
        _, _, reproj_2Ds = reproject(camera, ref_proj_points3D, pose_for_reproj)

        p3D_ids = ref_proj_frame.point3D_ids
        xys = reproj_2Ds 


        extraction_result, resize_img = single_img_processing(GLOBAL_DETECTOR, 
                                                  os.path.join(synthetic_img_folder, synthetic_name),
                                                  keypoints = reproj_2Ds)
        


        # print("reprojection coordinates", xys)
        # print("superpoint return coordinates", extraction_result["keypoints"])
        
        assert len(p3D_ids) == len(xys) == len(ref_proj_points3D)
        print("p3ds no stack", ref_proj_points3D.shape)
        tmp = np.stack(ref_proj_points3D, 0)
        print("p3ds shape", tmp.shape)
        print("p3dids type", type(p3D_ids))
        print("p3dids shape", p3D_ids.shape)
        exit()
        data_3D = {"p3D_ids": p3D_ids, 
                   "xys": xys, 
                   "p3Ds": np.stack(ref_proj_points3D, 0), 
                   "errors": errors}
        
        data_3D = filter_out_of_bound(data_3D, extraction_result["image_size"])
        # print(data_3D["xys"])
        # print(data_3D["p3Ds"])
        # print(data_3D["p3D_ids"])
        # break
        # Label files
        with h5py.File(os.path.join(output, "h5", s_name3d), 'w') as fd:
            grp = fd.create_group(s_name3d.replace(".h5", ""))
            for k, v in data_3D.items():
                grp.create_dataset(k, data=v)
        counter += 1
        with open(os.path.join(out_dir, "readme.txt"), 'a') as wt:
            wt.write("{0} {1} {2} {3} {4}\n".format(*[image_name, s_name, s_name3d, pose, camera]))

        
        data = {"image_size": np.array([640, 480])}
        for k, v in extraction_result.items():
            if k == "image_size":
                continue
            data[k] = v
        # Data files
        with h5py.File(os.path.join(output, "h5", s_name), 'w') as fd:
            grp = fd.create_group(s_name.replace(".h5", ""))
            for k, v in data.items():
                grp.create_dataset(k, data=v)
    print("Done")
            
def read_dataframe(df_path):
    df = pd.read_csv(df_path)
    names = df["name"].values.tolist()
    ref_frames  = df["ref_name"].values.tolist()
    qx = df["qx"].values
    qy = df["qy"].values
    qz = df["qz"].values
    qw = df["qw"].values
    tx = df["tx"].values
    ty = df["ty"].values
    tz = df["tz"].values

    qvecs = np.concatenate([qw.reshape([-1, 1]), qx.reshape([-1, 1]), qy.reshape([-1, 1]), qz.reshape([-1, 1])], 1)
    tvecs = np.concatenate([tx.reshape([-1, 1]), ty.reshape([-1, 1]), tz.reshape([-1, 1])], 1)
    print(qvecs.shape)
    print(tvecs.shape)

    ret_dict = {}
    for i, name in enumerate(names):
        ret_dict[name.replace("/", "-")] = {
            "ref_frame": ref_frames[i],
            "qvec": qvecs[i],
            "tvec": tvecs[i]
        }
    return ret_dict

def get_descriptors(img_name, features):
    data = {}
    with h5py.File(features, 'r') as fd:
        grp = fd[img_name]
        for k, v in grp.items():
            data[k+'0'] = torch.from_numpy(v.__array__()).float()
        # data['image0'] = torch.empty(1, ) + tuple(grp['image_size'])[::-1]
    return data


def keypoint_3D_map(original_p3Ds, original_p3D_ids, original_errors, matches, synthetic_kpts):
    original_matches_indices = matches[:, 1]
    synthetic_matches_indices = matches[:, 0]
    s_o_map = {s: o for s, o in zip(synthetic_matches_indices, original_matches_indices)}

    synthetic_p3Ds = []
    synthetic_p3D_ids = []
    synthetic_errors = []
    valid_counter = 0
    for i, kpt in enumerate(synthetic_kpts):
        if i in synthetic_matches_indices:
            synthetic_p3Ds.append(original_p3Ds[s_o_map[i]])
            synthetic_p3D_ids.append(original_p3D_ids[s_o_map[i]])
            valid_counter += 1
            synthetic_errors.append(original_errors[s_o_map[i]])
        else:
            synthetic_p3Ds.append(np.array([0, 0, 0]))
            synthetic_p3D_ids.append(-1)
            synthetic_errors.append(0)
    # print("Valid counter: ", valid_counter)

    return np.stack(np.array(synthetic_p3Ds), 0), \
            np.array(synthetic_p3D_ids), \
            np.array(synthetic_errors)



def generate_synthetic_data_lightglue_match(sfm_ref_path, dataframe, synthetic_img_folder, feature_path, out_dir):
    mode = "synthetic"
    counter = 0
    # Read the colmap reference dataset
    ref_cameras, ref_images, ref_points3D = read_model(sfm_ref_path, ext='.txt')
    #ref_img_fnames = get_refimg_fnames(ref_images)
    # Read the colmap full dataset 
    #_, full_images, _ = read_model(sfm_full_path)
    synthetic_dict = read_dataframe(dataframe)

    synthetic_img_fnames = get_synthetic_fnames(synthetic_img_folder)
    #reprojection_dict = generate_proj_frame(ref_images, full_images)
    output = make_out_dir(out_dir, mode)
    matches_dict = {}
    # print(synthetic_img_folder)
    # exit()

    for name, val in tqdm(synthetic_dict.items()):
        image_name = name
        print(image_name)
        assert image_name in synthetic_img_fnames

        s_name = "train_" + str(counter) + ".h5"
        s_name3d = "label_" + str(counter) + ".h5"
        pose = text_pose(val['tvec'], val['qvec'])
        camera = camera2txt(ref_cameras[1])

        ref_proj_frame = get_image(ref_images, val['ref_frame'])
        ref_proj_points2D, ref_proj_points3D, errors = get_keypoints(ref_proj_frame, ref_points3D)

        p3D_ids = ref_proj_frame.point3D_ids

        extraction_result, resize_img = single_img_processing(GLOBAL_DETECTOR, 
                                                  os.path.join(synthetic_img_folder, name),
                                                  keypoints = None)
        


        ref_features = get_descriptors(val['ref_frame'], os.path.join(feature_path, "feats-superpoint-n4096-r1024.h5"))
        matcher_data = {"image0": {"keypoints": torch.from_numpy(extraction_result['keypoints']).unsqueeze(0).cuda(), 
                                   "descriptors": torch.from_numpy(extraction_result['descriptors']).unsqueeze(0).transpose(1, 2).cuda()
                                   },
                        "image1": {"keypoints": torch.from_numpy(ref_proj_points2D).float().unsqueeze(0).cuda(),
                                   "descriptors": ref_features['descriptors0'].unsqueeze(0).transpose(1, 2).cuda()
                                   }
                        }


        matching_result = GLOBAL_MATCHER(matcher_data) 
        matches = matching_result['matches'][0].cpu().detach().numpy()


        len_matches = matches[:, 0] 
        len_matches = len([i for i in len_matches if i != -1])
        if len_matches < 500:
            continue

        synthetic_p3Ds, synthetic_p3D_ids, synthetic_errors = keypoint_3D_map(ref_proj_points3D, 
                                                                              p3D_ids, 
                                                                              errors, 
                                                                              matches, 
                                                                              extraction_result['keypoints'])
        matches_dict[image_name]  = {
            "num_matches": len_matches
        }

        #continue
        
        
        data_3D = {"p3D_ids": synthetic_p3D_ids, 
                   "xys": extraction_result['keypoints'], 
                   "p3Ds": synthetic_p3Ds, 
                   "errors": synthetic_errors}
        
        with h5py.File(os.path.join(output, "h5", s_name3d), 'w') as fd:
            grp = fd.create_group(s_name3d.replace(".h5", ""))
            for k, v in data_3D.items():
                grp.create_dataset(k, data=v)
        counter += 1
        with open(os.path.join(out_dir, "readme.txt"), 'a') as wt:
            wt.write("{0} {1} {2} {3} {4}\n".format(*[image_name, s_name, s_name3d, pose, camera]))

        
        data = {"image_size": np.array([640, 480])}
        for k, v in extraction_result.items():
            if k == "image_size":
                continue
            data[k] = v
        # Data files
        with h5py.File(os.path.join(output, "h5", s_name), 'w') as fd:
            grp = fd.create_group(s_name.replace(".h5", ""))
            for k, v in data.items():
                grp.create_dataset(k, data=v)
    print("Done")
    return matches_dict

def main():
    #synthetic_img_folder = "/media/hoang/Data/D2S--/data/custom/NGPGenerated2"
    # synthetic_img_folder = "/media/hoang/Data/D2S--/output/nerfstudio/generated_data/fire7scene"
    # out_dir = "/media/hoang/Data/D2S--/output/nerfstudio/generated_data/fire7scene_features"
    # sfm_ref_path = "/media/hoang/Data/data/hloc/output_10percent_2048/7scenes/fire/sfm_superpoint+superglue/"
    # sfm_full_path = "/media/hoang/Data/data/hloc/full_2048/fire/sfm_superpoint+superglue/"


    ### Test with the synthetic images that are a bit shifted from the original images
    synthetic_img_folder = "/media/hoang/Data/D2S--/output/nerfstudio/7scenes/fire/images"
    out_dir = "/media/hoang/Data/D2S--/output/nerfstudio/7scenes/fire/features"
    sfm_ref_path = "/media/hoang/Data/D2S--/data/sfm_hloc_superpoint_10_uniform/7scenes/fire" 
    sfm_full_path = "/media/hoang/Data/D2S--/data/sfm_hloc_superpoint_100/7scenes/fire"
    #generate_synthetic_data(sfm_ref_path, sfm_full_path, synthetic_img_folder, out_dir)


    ### Test with synthetic from new poses:
    dataframe_path = "/media/hoang/Data/D2S--/test/synthetic_transform2/synthetic_poses2.csv"
    synthetic_img_folder = "/media/hoang/Data/D2S--/output/synthetic/nerfstudio/7scenes/fire/images"
    out_dir = "/media/hoang/Data/D2S--/output/synthetic/nerfstudio/7scenes/fire/features"


    ### Test with 1% dataset
    sfm_ref_path = "/media/hoang/Data/D2S--/data/sfm_hloc_superpoint_5_evenly/7scenes/stairs/sfm_superpoint+superglue"
    feature_path = "/media/hoang/Data/D2S--/data/sfm_hloc_superpoint_5_evenly/7scenes/stairs/"
    dataframe_path = "/media/hoang/Data/D2S--/data/poses/5_evenly/7scenes/stairs/synthetic_poses_topk-2_samples-10.csv"
    synthetic_img_folder = "/media/hoang/Data/D2S--/output/synthetic/5_evenly/nerfstudio/7scenes/stairs/images"
    out_dir = "/media/hoang/Data/D2S--/output/synthetic/5_evenly/nerfstudio/7scenes/stairs/features"


    
    ### Test with 1% dataset
    #sfm_ref_path = "/media/hoang/Data/D2S--/data/sfm_hloc_superpoint_1_evenly/12scenes/office2_5b/sfm_superpoint+superglue"
    #feature_path = "/media/hoang/Data/D2S--/data/sfm_hloc_superpoint_1_evenly/12scenes/office2_5b/"
    #dataframe_path = "/media/hoang/Data/D2S--/data/poses/1_evenly/12scenes/office2_5b/synthetic_poses_topk-2_samples-30.csv"
    #synthetic_img_folder = "/media/hoang/Data/D2S--/output/synthetic/1_evenly/nerfstudio/12scenes/office2_5b/images"
    #out_dir = "/media/hoang/Data/D2S--/output/synthetic/1_evenly/nerfstudio/12scenes/office2_5b/features"
    match_dict = generate_synthetic_data_lightglue_match(sfm_ref_path, 
                                                         dataframe_path,
                                                         synthetic_img_folder,
                                                         feature_path,
                                                         out_dir)
    num_matches = []
    for k, v in match_dict.items():
        num_matches.append(v["num_matches"])

    print("Min num matches", num_matches)
    with open(os.path.join(out_dir, 'match_count.txt'), 'w') as f:
        for line in num_matches:
            f.write(f"{line}\n")



if __name__ == '__main__':
    main()
