# Descriptor Synthesis by NeRF for D2S
### [Project Page](https://austrianoakvn.github.io/nerf-vloc) | [Paper](https://arxiv.org/pdf/2403.10297)

## Demo 
 ![system](assets/web_demo.mp4)
 ![demo](assets/office_crop_pcd.mp4)

## Installation 

The program is tested with Python 3.8+ and torch 1.13.1 and dependencies in `requirements.txt`.

You will also need to install the following from sources:
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [lightglue](https://github.com/cvg/LightGlue)
- [Hloc](https://github.com/cvg/Hierarchical-Localization)

## Supported datasets 
- [Microsoft 7scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [12scenes]()


## How to use 
The code is still under refactoring so it may be confusing while reading.

- Hloc need to run first in order to obtain the sfm model (in this case we use triangulation from superpoint+superglue).
- Extract a subset of the dataset to train nerf on by `preprocessing.py`.
- Generate novel poses using `create_novel_pose.py`
- Generate synthetic images by using `view_synthesis.py`
- Generate descriptors by matching using `generate_synthetic.py`
- 

## Citation
Consider citing if you find this usefull
```
@article{bui2024leveraging,
  title={Leveraging Neural Radiance Field in Descriptor Synthesis for Keypoints Scene Coordinate Regression},
  author={Bui, Huy-Hoang and Bui, Bach-Thuan and Tran, Dinh-Tuan and Lee, Joo-Ho},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robotics and Systems},
  year={2024}
}
```

## Acknowledgement
We thank the author of [Hloc](https://github.com/cvg/Hierarchical-Localization), and [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) for providing their opensource contribution.

