import logging
import numpy as np
import cv2

from hloc.utils.read_write_model import read_model, write_model


def create_reference_sfm(full_model, ref_model, blacklist=None, ext='.bin'):
    '''Create a new COLMAP model with only training images.'''
    logging.info('Creating the reference model.')
    ref_model.mkdir(exist_ok=True)
    cameras, images, points3D = read_model(full_model, ext)

    if blacklist is not None:
        with open(blacklist, 'r') as f:
            blacklist = f.read().rstrip().split('\n')

    images_ref = dict()
    for id_, image in images.items():
        if blacklist and image.name in blacklist:
            continue
        images_ref[id_] = image

    points3D_ref = dict()
    for id_, point3D in points3D.items():
        ref_ids = [i for i in point3D.image_ids if i in images_ref]
        if len(ref_ids) == 0:
            continue
        points3D_ref[id_] = point3D._replace(image_ids=np.array(ref_ids))

    write_model(cameras, images_ref, points3D_ref, ref_model, '.bin')
    logging.info(f'Kept {len(images_ref)} images out of {len(images)}.')


def scale_sfm_images(full_model, scaled_model, image_dir):
    '''Duplicate the provided model and scale the camera intrinsics so that
       they match the original image resolution - makes everything easier.
    '''
    logging.info('Scaling the COLMAP model to the original image size.')
    scaled_model.mkdir(exist_ok=True)
    cameras, images, points3D = read_model(full_model)

    scaled_cameras = {}
    for id_, image in images.items():
        name = image.name
        img = cv2.imread(str(image_dir / name))
        assert img is not None, image_dir / name
        h, w = img.shape[:2]
        h, w = [484,648]

        cam_id = image.camera_id
        if cam_id in scaled_cameras:
            assert scaled_cameras[cam_id].width == w
            assert scaled_cameras[cam_id].height == h
            continue

        camera = cameras[cam_id]
        assert camera.model == 'SIMPLE_RADIAL'
        sx = w / camera.width
        sy = h / camera.height
        assert sx == sy, (sx, sy)
        scaled_cameras[cam_id] = camera._replace(
            width=w, height=h, params=camera.params*np.array([sx, sx, sy, 1.]))

    write_model(scaled_cameras, images, points3D, scaled_model)
