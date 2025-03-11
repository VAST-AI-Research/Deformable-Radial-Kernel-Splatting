#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

import os
import torch
import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R

from PIL import Image
import tqdm


WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image.convert('RGB'), resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if np.asarray(cam_info.image).shape[-1] == 4:
        loaded_mask = torch.from_numpy(np.asarray(cam_info.image)[..., -1:]).float().permute(2, 0, 1) / 255.

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def interpolate_se3_with_noise(matrices, N, quat_noise_scale=0.01, trans_noise_scale=0.01, weight_noise_scale=0.01):
    """
    Interpolates SE(3) matrices and adds noise to quaternions and translations.

    :param matrices: Tensor of shape [B, 4, 4] representing the input matrices.
    :param N: The desired number of output matrices.
    :param quat_noise_scale: The standard deviation of the Gaussian noise for quaternions.
    :param trans_noise_scale: The standard deviation of the Gaussian noise for translations.
    :return: Tensor of shape [N, 4, 4] with interpolated and noisy matrices.
    """
    B = matrices.shape[0]

    # Extract rotations and translations
    rotations = matrices[:, :3, :3]
    translations = matrices[:, :3, 3]

    # Convert rotations to scipy Rotation objects
    rotation_objects = R.from_matrix(rotations)

    # Interpolate indices
    indices = np.linspace(0, B - 1, N)
    indices = indices + np.random.randn(*indices.shape) * weight_noise_scale
    indices = indices.clip(1/2/N, B-1-1/2/N)
    lower_indices = np.floor(indices).astype('int')
    upper_indices = np.ceil(indices).astype('int')
    weights = indices - lower_indices.astype('float')

    # SLERP for rotations
    slerp = Slerp(range(B), rotation_objects)
    interpolated_rotations = slerp(indices).as_quat()

    # Linear interpolation for translations
    interpolated_translations = (1 - weights)[:, None] * translations[lower_indices] + \
                                weights[:, None] * translations[upper_indices]

    # Add noise to quaternions
    noise_quat = np.random.normal(scale=quat_noise_scale, size=interpolated_rotations.shape)
    noisy_quats = R.from_quat(interpolated_rotations + noise_quat).as_matrix()

    # Add noise to translations
    noise_trans = np.random.randn(*interpolated_translations.shape) * trans_noise_scale
    noisy_translations = interpolated_translations + noise_trans

    # Combine rotations and translations into SE(3) matrices
    interpolated_matrices = np.stack([np.eye(4)]*N, axis=0)
    interpolated_matrices[:, :3, :3] = noisy_quats
    interpolated_matrices[:, :3, 3]  = noisy_translations

    return interpolated_matrices


def visualize_cameras(poses, z_val=0.1, check_path='./check_poses/', check_name='poses.obj', color_value=None):
    points = np.array([[0., 0., 0.],
                       [-1., -1., 1.],
                       [-1., 1., 1.],
                       [1., -1., 1.],
                       [1., 1., 1.]])
    faces = np.array([[0, 1, 2],
                      [0, 3, 1],
                      [0, 4, 3],
                      [0, 2, 4],
                      [1, 3, 2],
                      [2, 3, 4]])
    points *= z_val
    new_points = np.einsum('na,mba->mnb', np.concatenate([points, np.ones_like(points[..., :1])], axis=-1), poses)
    new_points = new_points[..., :-1].reshape([-1, 3])
    new_faces = np.stack([faces + points.shape[0] * i for i in range(poses.shape[0])], axis=0).reshape([-1, 3])
    colors = np.linspace(0, 255, poses.shape[0], dtype=np.int32) if color_value is None else color_value
    colors = np.broadcast_to(np.stack([colors] * 3, axis=-1)[:, np.newaxis], [colors.shape[0], points.shape[0], 3]).reshape([-1, 3])

    if not os.path.exists(check_path):
        os.makedirs(check_path)
    
    str_v = [f"v {new_points[i][0]} {new_points[i][1]} {new_points[i][2]} {colors[i][0]} {colors[i][0]} {colors[i][0]}\n" for i in range(new_points.shape[0])]
    str_f = [f"f {new_faces[i][0]+1} {new_faces[i][1]+1} {new_faces[i][2]+1}\n" for i in range(new_faces.shape[0])]
    with open(os.path.join(check_path, check_name), 'w') as file:
        file.write(f'{"".join(str_v)}{"".join(str_f)}')


@torch.no_grad()
def augment_cameras(viewpoint_stack, teacher_model, bg, vis_path=None):
    print('Augmenting train views')
    Rs, Ts = [], []
    for camera in viewpoint_stack:
        Rs.append(camera.R.transpose())
        Ts.append(camera.T)
    W2C = np.stack([np.eye(4)] * len(Rs), axis=0)
    W2C[:, :3, :3] = np.stack(Rs, axis=0)
    W2C[:, :3, 3]  = np.stack(Ts, axis=0)
    C2W = np.linalg.inv(W2C)
    aug_C2W = interpolate_se3_with_noise(matrices=C2W, N=len(Rs), quat_noise_scale=0.1, trans_noise_scale=0.05*(C2W[:, :3, 3].max()-C2W[:, :3, 3].min()))
    if vis_path is not None:
        visualize_cameras(C2W,     check_path=vis_path, check_name='original_cam.obj', color_value=0)
        visualize_cameras(aug_C2W, check_path=vis_path, check_name='augment_cam.obj',  color_value=255)
    aug_W2C = np.linalg.inv(aug_C2W)
    cam0 = viewpoint_stack[0]
    aug_viewpoint_stack = []
    for i in tqdm.tqdm(range(aug_W2C.shape[0])):
        R, T = aug_W2C[i, :3, :3].transpose(), aug_W2C[i, :3, 3]
        cam = Camera(colmap_id=None, R=R, T=T, 
                     FoVx=cam0.FoVx, FoVy=cam0.FoVy, 
                     image=cam0.original_image, gt_alpha_mask=None,
                     image_name=None, uid=id, data_device=cam0.data_device)
        render_pkg = teacher_model.render_func(cam, teacher_model, bg_color=bg)
        gt_image = render_pkg["render"]
        cam.original_image = gt_image
        gt_alpha_mask = render_pkg['alpha'] if 'alpha' in render_pkg else None
        cam.gt_alpha_mask = gt_alpha_mask
        aug_viewpoint_stack.append(cam)
        if vis_path is not None:
            os.makedirs(vis_path, exist_ok=True)
            Image.fromarray((gt_image.detach().cpu().clip(0, 1).permute(1,2,0).numpy() * 255).astype('uint8')).save(os.path.join(vis_path, '%05d.png' % i))
    return aug_viewpoint_stack


@torch.no_grad()
def obtain_depths_for_distilling(viewpoint_stack, teacher_model, bg, vis_path=None):
    print('Generating depth images of train views for distilling')
    for i, camera in enumerate(viewpoint_stack):
        gt_image = teacher_model.render_func(camera, teacher_model, bg_color=bg)["depth"]
        camera.depth_image = gt_image
        if vis_path is not None:
            os.makedirs(vis_path, exist_ok=True)
            vis_depth = (gt_image - gt_image.min()) / (gt_image.max() - gt_image.min())
            vis_depth = vis_depth.repeat(3, 1, 1)
            Image.fromarray((vis_depth.detach().cpu().clip(0, 1).permute(1,2,0).numpy() * 255).astype('uint8')).save(os.path.join(vis_path, '%05d.png' % i))
    return viewpoint_stack
