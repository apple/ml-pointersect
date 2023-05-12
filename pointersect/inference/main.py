#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# The file implements the main interface of inference.
import copy
import json
import os
import traceback
import typing as T
from pprint import pprint
from timeit import default_timer as timer

import fire
import numpy as np
import open3d as o3d
import torch
import yaml

from cdslib.core.utils import print_and_save
from pointersect.data import dataset_helper
from pointersect.data import hypersim_dataset
from pointersect.inference import infer, inference_utils
from pointersect.inference.structures import Camera
from pointersect.inference.structures import CameraTrajectory
from pointersect.inference.structures import Mesh
from pointersect.inference.structures import PointCloud
from pointersect.inference.structures import RGBDImage


def get_settings(
        mesh_scale: float = 1.,
        input_camera_trajectory_params: T.Dict[str, T.Any] = None,
        output_camera_trajectory_params: T.Dict[str, T.Any] = None,
        input_camera_setting: T.Dict[str, T.Any] = None,
        output_camera_setting: T.Dict[str, T.Any] = None,
        save_settings: T.Dict[str, T.Any] = None,
        pr_setting: T.Dict[str, T.Any] = None,
        model_loading_settings: T.Dict[str, T.Any] = None,
):
    input_camera_trajectory_params_defaults = {
        'min_r': 3,
        'max_r': 4,
        'max_angle': 30.,
        'num_circle': 4,
        'r_freq': 1,
    }
    output_camera_trajectory_params_defaults = {
        'min_r': 3,
        'max_r': 4,
        'max_angle': 30.,
        'num_circle': 4,
        'r_freq': 1,
    }
    input_camera_setting_defaults = {
        'fov': 60,
        'width_px': 100,  # 256,
        'height_px': 100,  # 256,
    }
    output_camera_setting_defaults = {
        'fov': 60,
        'width_px': 400,  # 256,
        'height_px': 400,  # 256,
        'ray_offsets': 'center',
    }
    pr_setting_defaults = dict(
        ray_radius=None,  # 0.1,  # if <0, set to grid_width/grid_size *2
        grid_size=None,  # 100,
        grid_center=None,  # 0,
        grid_width=None,
        # 2.2 * mesh_scale,  # full width, it is for mesh range: [-scale, scale] + some margin for numerical error
    )
    save_settings_defaults = dict(
        gif_fps=10,
        save_pt=False,  # True,
        save_npy=True,
        save_gif=True,
        save_png=True,
        save_ply=True,
        overwrite=True,
        background_color=1.,
        font_size=12,
        idx_ref=-1,
    )
    model_loading_settings_defaults = dict(
        loss_name='test_epoch_loss_hit',  # 'valid_epoch_loss_hit',  # training test set is not the test set
        loss_smooth_window=100,  # 1,
        loss_smooth_std=30.,  # 1.,
    )

    if input_camera_trajectory_params is not None:
        input_camera_trajectory_params_defaults.update(input_camera_trajectory_params)
    input_camera_trajectory_params = input_camera_trajectory_params_defaults

    if output_camera_trajectory_params is not None:
        output_camera_trajectory_params_defaults.update(output_camera_trajectory_params)
    output_camera_trajectory_params = output_camera_trajectory_params_defaults

    if input_camera_setting is not None:
        input_camera_setting_defaults.update(input_camera_setting)
    input_camera_setting = input_camera_setting_defaults

    if output_camera_setting is not None:
        output_camera_setting_defaults.update(output_camera_setting)
    output_camera_setting = output_camera_setting_defaults

    if pr_setting is not None:
        pr_setting_defaults.update(pr_setting)
    pr_setting = pr_setting_defaults

    if save_settings is not None:
        save_settings_defaults.update(save_settings)
    save_settings = save_settings_defaults

    if model_loading_settings is not None:
        model_loading_settings_defaults.update(model_loading_settings)
    model_loading_settings = model_loading_settings_defaults

    return dict(
        input_camera_trajectory_params=input_camera_trajectory_params,
        output_camera_trajectory_params=output_camera_trajectory_params,
        input_camera_setting=input_camera_setting,
        output_camera_setting=output_camera_setting,
        pr_setting=pr_setting,
        save_settings=save_settings,
        model_loading_settings=model_loading_settings,
    )


def render_mesh(
        mesh_filename: str,
        output_dir: str,
        model_filename: T.Union[str, T.List[str]],
        # for input/output camera
        input_point_sample_method: str,  # 'rgbd', 'poisson_disk'
        n_input_imgs: int,
        n_output_imgs: int,
        input_camera_trajectory_mode: str,
        output_camera_trajectory_mode: str,
        # for pointersect
        k: int,
        # for comparison with baselines
        render_pointersect: bool = True,
        render_npbgpp: bool = False,
        render_surfel: bool = True,
        render_nglod: bool = False,
        render_ngp: bool = False,
        render_dsnerf: bool = False,
        render_ibrnet: bool = False,
        render_poisson: bool = True,
        densify_neural_points: bool = False,
        # other settings
        mesh_scale: float = 1.0,
        rnd_seed: int = 0,
        input_camera_trajectory_params: T.Dict[str, T.Any] = None,
        output_camera_trajectory_params: T.Dict[str, T.Any] = None,
        input_camera_setting: T.Dict[str, T.Any] = None,
        output_camera_setting: T.Dict[str, T.Any] = None,
        save_settings: T.Dict[str, T.Any] = None,
        pr_setting: T.Dict[str, T.Any] = None,
        model_loading_settings: T.Dict[str, T.Any] = None,
        max_ray_chunk_size: int = int(1e4),  # k=40: int(4e4),
        max_pr_chunk_size: int = -1,
        max_model_chunk_size: int = -1,
        th_hit_prob: float = 0.5,
        n_input_points: int = 9728,
        force_same_intrinsic: bool = False,
        voxel_downsample_cell_width: float = -1,
        voxel_downsample_sigma: float = 0.5,
        neural_point_upsample_ratio_x48: int = 1,
        total_nglod_epoch: int = 50,
        nglod_test_every_iter: int = 50,
        total_ngp_epoch: int = 50,
        ngp_test_every_iter: int = 50,
        total_dsnerf_iter: int = 50000,
        dsnerf_test_every_iter: int = 5000,
        test_plane_normal: bool = False,
        drop_point_cloud_features: bool = False,
        drop_point_cloud_normal: bool = False,
        ibrnet_chunk_size: int = 4096,
        surfel_point_size: float = 1.,
        **kwargs,
):
    """
    Given a mesh file (as ground truth),
    1) sample point cloud using RGBD cameras, or directly sample from mesh
    2) render the point cloud from different viewpoints (using different methods)
    3) compute errors (surface normal, point to mesh, image error, silhouette error, etc)

    Args:
        mesh_filename:
            filename of the mesh
        output_dir:
            dir name of save the results
        model_filename:
            checkpoint filename the pointersect model.
            If a list is given, it will render each model.
        input_point_sample_method:
            the method to generate the input point cloud
        n_input_imgs:
            number of input views to create point cloud
        n_output_imgs:
            number of output views to create point cloud
        input_camera_trajectory_mode:
            name of the camera trajectory used for input
        output_camera_trajectory_mode:
            name of the camera trajectory used during output
        k:
            number of neighbor points to used during pointersect
        mesh_scale:
            scale the mesh to [-scale, scale]
        rnd_seed:
            random seed to use
        input_camera_trajectory_params

    Returns:

    """

    setting_dict = get_settings(
        mesh_scale=mesh_scale,
        input_camera_trajectory_params=input_camera_trajectory_params,
        output_camera_trajectory_params=output_camera_trajectory_params,
        input_camera_setting=input_camera_setting,
        output_camera_setting=output_camera_setting,
        save_settings=save_settings,
        pr_setting=pr_setting,
        model_loading_settings=model_loading_settings,
    )
    input_camera_trajectory_params = setting_dict['input_camera_trajectory_params']
    output_camera_trajectory_params = setting_dict['output_camera_trajectory_params']
    input_camera_setting = setting_dict['input_camera_setting']
    output_camera_setting = setting_dict['output_camera_setting']
    pr_setting = setting_dict['pr_setting']
    save_settings = setting_dict['save_settings']
    model_loading_settings = setting_dict['model_loading_settings']

    if os.path.exists(output_dir) and not save_settings['overwrite']:
        raise RuntimeError
    os.makedirs(output_dir, exist_ok=True)

    # strategy to render large images:
    # we are going to store everything on cpu
    # pr: cuda (since it handles large number of points and rays)
    # pointersect: cuda
    data_device = torch.device('cpu')
    model_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load mesh
    assert os.path.exists(mesh_filename), f'{mesh_filename} not exist'
    print(f'loading mesh: {mesh_filename}')
    stime = timer()
    mesh = Mesh(
        mesh=mesh_filename,
        scale=mesh_scale,
    )
    input_mesh_dir = os.path.join(output_dir, 'input_mesh')
    os.makedirs(input_mesh_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(
        filename=os.path.join(input_mesh_dir, 'mesh.obj'),
        mesh=mesh.mesh,
    )
    print(f'done loading mesh. used {timer() - stime: .2f} secs ')

    # create camera trajectory
    print(f'loading output camera trajectory..')
    stime = timer()
    if n_input_imgs > 0:
        input_cam_trajectory = CameraTrajectory(
            mode=input_camera_trajectory_mode,
            n_imgs=n_input_imgs,
            total=1,  # use all
            rng_seed=rnd_seed,
            params=input_camera_trajectory_params,
        )
        input_cameras = input_cam_trajectory.get_camera(
            fov=input_camera_setting['fov'],
            width_px=input_camera_setting['width_px'],
            height_px=input_camera_setting['height_px'],
            device=data_device,
        )
    else:
        input_cam_trajectory = None
        input_cameras = None
    output_camera_trajectory = CameraTrajectory(
        mode=output_camera_trajectory_mode,
        n_imgs=n_output_imgs,
        total=1,  # use all
        rng_seed=rnd_seed,
        params=output_camera_trajectory_params,
    )
    if not force_same_intrinsic:
        output_cameras = output_camera_trajectory.get_camera(
            fov=output_camera_setting['fov'],
            width_px=output_camera_setting['width_px'],
            height_px=output_camera_setting['height_px'],
            device=data_device,
        )
    else:
        output_cameras = output_camera_trajectory.get_camera(
            fov=input_camera_setting['fov'],
            width_px=input_camera_setting['width_px'],
            height_px=input_camera_setting['height_px'],
            device=data_device,
        )
    print(f'done loading output camera trajectory. used {timer() - stime: .2f} secs ')

    # render input rgbd images
    print(f'loading input rgbd images..')
    stime = timer()
    input_rgbd_images = mesh.get_rgbd_image(
        camera=input_cameras,
        render_normal_w=True,
        device=data_device,
    )
    print(f'done rendering input rgbd images. used {timer() - stime: .2f} secs ')

    # render ground-truth rgbd images
    print(f'loading output ground-truth rgbd images..')
    stime = timer()
    output_rgbd_images = mesh.get_rgbd_image(
        camera=output_cameras,
        render_normal_w=True,
        device=data_device,
    )
    gt_rgbd_dir = os.path.join(output_dir, 'gt_rgbd')
    output_rgbd_images.save(
        output_dir=gt_rgbd_dir,
        overwrite=save_settings['overwrite'],
        save_png=save_settings['save_png'],
        save_pt=save_settings['save_pt'],
        save_gif=save_settings['save_gif'],
        gif_fps=save_settings['gif_fps'],
        background_color=save_settings['background_color'],
    )
    print(f'done rendering output ground-truth rgbd images. used {timer() - stime: .2f} secs ')

    # render input point cloud
    # (b=1, q, h, w) -> (b=1, n)
    print(f'generating input point cloud with {input_point_sample_method}')
    stime = timer()
    if input_point_sample_method == 'rgbd':
        input_point_cloud = input_rgbd_images.get_pcd(
            subsample=1,
            remove_background=True,
        )
    else:
        input_point_cloud = mesh.sample_point_cloud(
            num_points=n_input_points,
            method=input_point_sample_method,
        )['point_cloud']

    # voxel downsampling
    input_point_cloud = input_point_cloud.voxel_downsampling(
        cell_width=voxel_downsample_cell_width,
        sigma=voxel_downsample_sigma,
    )

    if drop_point_cloud_features:
        # we provide ground-truth normal for poisson
        input_point_cloud.drop_features(drop_normal=drop_point_cloud_normal)
    print(f'done generating point cloud. used {timer() - stime: .2f} secs ')

    # save inputs (rgbd images, pcd)
    print(f'saving information to disk for debugging..')
    stime = timer()
    rgbd_dir = os.path.join(output_dir, 'input_rgbd')
    input_rgbd_images.save(
        output_dir=rgbd_dir,
        overwrite=save_settings['overwrite'],
        save_png=save_settings['save_png'],
        save_pt=save_settings['save_pt'],
        save_gif=save_settings['save_gif'],
        gif_fps=save_settings['gif_fps'],
        background_color=save_settings['background_color'],
    )
    pcd_dir = os.path.join(output_dir, 'input_pcd')
    input_point_cloud.save(
        output_dir=pcd_dir,
        overwrite=save_settings['overwrite'],
        save_ply=save_settings['save_ply'],
        save_pt=save_settings['save_pt'],
    )

    # save input/output camera trajectory
    input_camera_dir = os.path.join(output_dir, 'input_camera')
    input_cameras.save(
        output_dir=input_camera_dir,
        overwrite=save_settings['overwrite'],
        save_ply=save_settings['save_ply'],
        save_individual_ply=save_settings['save_ply'],
        save_pt=True,  # save_settings['save_pt'],
        world_frame_size=1.,
        camera_frame_size=0.5,
        scene_meshes=[mesh.mesh],
    )
    output_camera_dir = os.path.join(output_dir, 'output_camera')
    output_cameras.save(
        output_dir=output_camera_dir,
        overwrite=save_settings['overwrite'],
        save_ply=save_settings['save_ply'],
        save_individual_ply=save_settings['save_ply'],
        save_pt=True,  # save_settings['save_pt'],
        world_frame_size=1.,
        camera_frame_size=0.5,
        scene_meshes=[mesh.mesh],
    )
    print(f'done saving information. used {timer() - stime: .2f} secs ')

    metric_dict = main_render(
        render_pointersect=render_pointersect,
        render_npbgpp=render_npbgpp,
        render_surfel=render_surfel,
        render_nglod=render_nglod,
        render_ngp=render_ngp,
        render_dsnerf=render_dsnerf,
        render_ibrnet=render_ibrnet,
        densify_neural_points=densify_neural_points,
        render_poisson=render_poisson,
        input_rgbd_images=input_rgbd_images,
        input_point_cloud=input_point_cloud,
        output_cameras=output_cameras,
        model_filename=model_filename,
        k=k,
        th_hit_prob=th_hit_prob,
        max_ray_chunk_size=max_ray_chunk_size,
        max_pr_chunk_size=max_pr_chunk_size,
        max_model_chunk_size=max_model_chunk_size,
        output_camera_setting=output_camera_setting,
        pr_setting=pr_setting,
        model_loading_settings=model_loading_settings,
        save_settings=save_settings,
        data_device=data_device,
        model_device=model_device,
        output_dir=output_dir,
        gt_rgbd_images=output_rgbd_images,
        gt_mesh=mesh,
        neural_point_upsample_ratio_x48=neural_point_upsample_ratio_x48,
        total_nglod_epoch=total_nglod_epoch,
        nglod_test_every_iter=nglod_test_every_iter,
        total_ngp_epoch=total_ngp_epoch,
        ngp_test_every_iter=ngp_test_every_iter,
        total_dsnerf_iter=total_dsnerf_iter,
        dsnerf_test_every_iter=dsnerf_test_every_iter,
        test_plane_normal=test_plane_normal,
        ibrnet_chunk_size=ibrnet_chunk_size,
        surfel_point_size=surfel_point_size,
    )
    return metric_dict


def render_rgbd(
        input_rgbd_images: RGBDImage,  # (1, q)
        output_dir: str,
        model_filename: T.Union[str, T.List[str]],
        n_output_imgs: int,
        output_camera_trajectory_mode: str,
        # for pointersect
        k: int,
        # for comparison with baselines
        render_pointersect: bool = True,
        render_npbgpp: bool = False,
        render_surfel: bool = True,
        render_nglod: bool = False,
        render_ngp: bool = False,
        render_dsnerf: bool = False,
        render_ibrnet: bool = False,
        render_poisson: bool = True,
        densify_neural_points: bool = False,
        # other settings
        rnd_seed: int = 0,
        output_camera_trajectory_params: T.Dict[str, T.Any] = None,
        output_camera_setting: T.Dict[str, T.Any] = None,
        save_settings: T.Dict[str, T.Any] = None,
        pr_setting: T.Dict[str, T.Any] = None,
        model_loading_settings: T.Dict[str, T.Any] = None,
        max_ray_chunk_size: int = int(1e4),  # k=40: int(4e4),
        max_pr_chunk_size: int = -1,
        max_model_chunk_size: int = -1,
        th_hit_prob: float = 0.5,
        gt_rgbd_images: T.Optional[RGBDImage] = None,
        input_point_cloud_subsample: int = 1,
        surfel_point_size: float = 1.,
        force_same_intrinsic: bool = False,
        voxel_downsample_cell_width: float = -1,
        voxel_downsample_sigma: float = 0.5,
        neural_point_upsample_ratio_x48: int = 1,
        total_nglod_epoch: int = 50,
        nglod_test_every_iter: int = 50,
        total_ngp_epoch: int = 50,
        ngp_test_every_iter: int = 50,
        total_dsnerf_iter: int = 50000,
        dsnerf_test_every_iter: int = 5000,
        test_plane_normal: bool = False,
        drop_point_cloud_features: bool = False,
        drop_point_cloud_normal: bool = False,
        pcd_outlier_removal_radius: float = -1,
        pcd_outlier_removal_num_points: int = -1,
        ibrnet_chunk_size: int = 4096,
        **kwargs,
):
    """Render given rgbd images."""

    setting_dict = get_settings(
        output_camera_trajectory_params=output_camera_trajectory_params,
        output_camera_setting=output_camera_setting,
        save_settings=save_settings,
        pr_setting=pr_setting,
        model_loading_settings=model_loading_settings,
    )
    output_camera_trajectory_params = setting_dict['output_camera_trajectory_params']
    output_camera_setting = setting_dict['output_camera_setting']
    pr_setting = setting_dict['pr_setting']
    save_settings = setting_dict['save_settings']
    model_loading_settings = setting_dict['model_loading_settings']

    if os.path.exists(output_dir) and not save_settings['overwrite']:
        raise RuntimeError
    os.makedirs(output_dir, exist_ok=True)

    # strategy to render large images:
    # we are going to store everything on cpu
    # pr: cuda (since it handles large number of points and rays)
    # pointersect: cuda
    data_device = torch.device('cpu')
    model_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create output camera
    if output_camera_trajectory_mode != 'gt':
        output_camera_trajectory = CameraTrajectory(
            mode=output_camera_trajectory_mode,
            n_imgs=n_output_imgs,
            total=1,  # use all
            rng_seed=rnd_seed,
            params=output_camera_trajectory_params,
        )
        if not force_same_intrinsic:
            output_cameras = output_camera_trajectory.get_camera(
                fov=output_camera_setting['fov'],
                width_px=output_camera_setting['width_px'],
                height_px=output_camera_setting['height_px'],
                device=data_device,
            )
        else:
            # create the placeholder
            output_cameras = output_camera_trajectory.get_camera(
                fov=output_camera_setting['fov'],  # dummy
                width_px=output_camera_setting['width_px'],  # dummy
                height_px=output_camera_setting['height_px'],  # dummy
                device=data_device,
            )

            avg_intrinsic = input_rgbd_images.camera.intrinsic.mean(dim=1, keepdim=True)  # (b, 1, 3, 3)
            assert torch.allclose(
                input_rgbd_images.camera.intrinsic,
                avg_intrinsic,
            ), f'force_same_intrinsic=True averages input image intrinsics.'

            output_cameras.intrinsic = avg_intrinsic.expand(-1, output_cameras.H_c2w.size(1), -1, -1)
            output_cameras.height_px = input_rgbd_images.camera.height_px
            output_cameras.width_px = input_rgbd_images.camera.width_px

            # flip_output_camera_H_yz if intrinsic[2, 2] is negative
            output_cameras.H_c2w[..., 1:2] = output_cameras.H_c2w[..., 1:2] * output_cameras.intrinsic[..., 1:2,
                                                                              1:2].sign()
            output_cameras.H_c2w[..., 2:3] = output_cameras.H_c2w[..., 2:3] * output_cameras.intrinsic[..., 2:3,
                                                                              2:3].sign()
    else:
        assert gt_rgbd_images is not None
        output_cameras = gt_rgbd_images.camera.clone()

    gt_rgbd_dir = os.path.join(output_dir, 'gt_rgbd')
    if gt_rgbd_images is not None:
        gt_rgbd_images.save(
            output_dir=gt_rgbd_dir,
            overwrite=save_settings['overwrite'],
            save_png=save_settings['save_png'],
            save_pt=save_settings['save_pt'],
            save_gif=save_settings['save_gif'],
            gif_fps=save_settings['gif_fps'],
            background_color=save_settings['background_color'],
        )

    # render input point cloud
    # (b=1, q, h, w) -> (b=1, n)
    print(f'generating input point cloud with rgbd')
    input_point_cloud = input_rgbd_images.get_pcd(
        subsample=input_point_cloud_subsample,
        remove_background=True,
    )

    # voxel downsampling
    input_point_cloud = input_point_cloud.voxel_downsampling(
        cell_width=voxel_downsample_cell_width,
        sigma=voxel_downsample_sigma,
        bidx=0,
    )

    if drop_point_cloud_features:
        # we provide ground-truth normal for poisson
        input_point_cloud.drop_features(drop_normal=drop_point_cloud_normal)

    # point cloud outlier removal
    if pcd_outlier_removal_radius is not None and pcd_outlier_removal_radius > 1e-6:
        input_point_cloud.remove_outlier(
            radius=pcd_outlier_removal_radius,
            min_num_points_in_radius=pcd_outlier_removal_num_points,
            printout=True,
        )

    # save inputs (rgbd images, pcd)
    print(f'saving rgbd images', flush=True)
    rgbd_dir = os.path.join(output_dir, 'input_rgbd')
    input_rgbd_images.save(
        output_dir=rgbd_dir,
        overwrite=save_settings['overwrite'],
        save_png=save_settings['save_png'],
        save_pt=save_settings['save_pt'],
        save_gif=save_settings['save_gif'],
        gif_fps=save_settings['gif_fps'],
        background_color=save_settings['background_color'],
    )
    print(f'saving input point cloud', flush=True)
    pcd_dir = os.path.join(output_dir, 'input_pcd')
    input_point_cloud.save(
        output_dir=pcd_dir,
        overwrite=save_settings['overwrite'],
        save_ply=save_settings['save_ply'],
        save_pt=save_settings['save_pt'],
    )

    # save input/output camera trajectory
    print(f'saving input cameras', flush=True)
    input_camera_dir = os.path.join(output_dir, 'input_camera')
    input_cameras = input_rgbd_images.camera
    # o3d_pcd = input_rgbd_images.get_pcd().get_o3d_pcds()[0]
    input_cameras.save(
        output_dir=input_camera_dir,
        overwrite=save_settings['overwrite'],
        save_ply=save_settings['save_ply'],
        save_individual_ply=save_settings['save_ply'],
        save_pt=True,
        world_frame_size=1.,
        camera_frame_size=0.5,
        scene_meshes=None,
    )
    print(f'saving output cameras', flush=True)
    output_camera_dir = os.path.join(output_dir, 'output_camera')
    output_cameras.save(
        output_dir=output_camera_dir,
        overwrite=save_settings['overwrite'],
        save_ply=save_settings['save_ply'],
        save_individual_ply=save_settings['save_ply'],
        save_pt=True,
        world_frame_size=1.,
        camera_frame_size=0.5,
        scene_meshes=None,
    )

    # sys.exit()

    print(f'main rendering', flush=True)
    main_render(
        render_pointersect=render_pointersect,
        render_npbgpp=render_npbgpp,
        render_surfel=render_surfel,
        render_nglod=render_nglod,
        render_ngp=render_ngp,
        render_dsnerf=render_dsnerf,
        render_ibrnet=render_ibrnet,
        render_poisson=render_poisson,
        densify_neural_points=densify_neural_points,
        input_rgbd_images=input_rgbd_images,
        input_point_cloud=input_point_cloud,
        output_cameras=output_cameras,
        model_filename=model_filename,
        k=k,
        th_hit_prob=th_hit_prob,
        max_ray_chunk_size=max_ray_chunk_size,
        max_pr_chunk_size=max_pr_chunk_size,
        max_model_chunk_size=max_model_chunk_size,
        output_camera_setting=output_camera_setting,
        pr_setting=pr_setting,
        model_loading_settings=model_loading_settings,
        save_settings=save_settings,
        data_device=data_device,
        model_device=model_device,
        output_dir=output_dir,
        gt_rgbd_images=gt_rgbd_images,
        surfel_point_size=surfel_point_size,
        neural_point_upsample_ratio_x48=neural_point_upsample_ratio_x48,
        total_nglod_epoch=total_nglod_epoch,
        nglod_test_every_iter=nglod_test_every_iter,
        total_ngp_epoch=total_ngp_epoch,
        ngp_test_every_iter=ngp_test_every_iter,
        total_dsnerf_iter=total_dsnerf_iter,
        dsnerf_test_every_iter=dsnerf_test_every_iter,
        test_plane_normal=test_plane_normal,
        ibrnet_chunk_size=ibrnet_chunk_size,
    )


def main_render(
        render_pointersect: bool,
        render_npbgpp: bool,
        render_surfel: bool,
        render_nglod: bool,
        render_ngp: bool,
        render_dsnerf: bool,
        render_ibrnet: bool,
        densify_neural_points: bool,
        render_poisson: bool,
        input_rgbd_images: T.Optional[RGBDImage],
        input_point_cloud: T.Optional[PointCloud],
        output_cameras: Camera,
        model_filename: T.Union[T.List[str], str],
        k: int,
        th_hit_prob: float,
        max_ray_chunk_size: int,
        max_pr_chunk_size: int,
        max_model_chunk_size: int,
        output_camera_setting: T.Dict[str, T.Any],
        pr_setting: T.Dict[str, T.Any],
        model_loading_settings: T.Dict[str, T.Any],
        save_settings: T.Dict[str, T.Any],
        data_device: torch.device,
        model_device: torch.device,
        output_dir: str = None,
        gt_rgbd_images: RGBDImage = None,
        gt_mesh: Mesh = None,
        surfel_point_size: T.Union[float, T.List[float]] = 1.,
        num_samples_per_pixel: int = 1,
        neural_point_upsample_ratio_x48: int = 1,
        total_nglod_epoch: int = 50,
        nglod_test_every_iter: int = 50,
        total_ngp_epoch: int = 50,
        ngp_test_every_iter: int = 50,
        total_dsnerf_iter: int = 50000,
        dsnerf_test_every_iter: int = 5000,
        test_plane_normal: bool = False,
        ibrnet_chunk_size: int = 4096,
):
    """
    Main rendering function that calls different methods.
    """
    if isinstance(model_filename, str):
        model_filename = [model_filename]

    result_rgbd_dict = dict()  # "name" -> rgbd_image
    total_time_dict = dict()
    pointersect_result_names = []
    # render with pointersect
    if render_pointersect:
        for idx in range(len(model_filename)):
            print(f'---- Rendering with pointersect ({model_filename[idx]})----')
            try:
                pointersect_result = infer.render_point_cloud_camera_using_pointersect(
                    model_filename=model_filename[idx],
                    k=k,
                    point_cloud=input_point_cloud,
                    output_cameras=output_cameras,
                    output_camera_setting=output_camera_setting,
                    model_loading_settings=model_loading_settings,
                    pr_setting=pr_setting,
                    model_device=model_device,
                    data_device=data_device,
                    th_hit_prob=th_hit_prob,
                    max_ray_chunk_size=max_ray_chunk_size,
                    max_pr_chunk_size=max_pr_chunk_size,
                    max_model_chunk_size=max_model_chunk_size,
                    print_out=True,
                    num_samples_per_pixel=num_samples_per_pixel,
                )  # (b, q, h, w)

                # save pointersect results
                if len(model_filename) == 1:
                    pointersect_dir = os.path.join(output_dir, f'pointersect')
                else:
                    pointersect_dir = os.path.join(output_dir, f'pointersect-{model_filename[idx]}')
                rgbd_images_pointersect = pointersect_result.get_rgbd_image(camera=output_cameras)
                rgbd_images_pointersect.save(
                    output_dir=pointersect_dir,
                    overwrite=save_settings['overwrite'],
                    save_png=save_settings['save_png'],
                    save_pt=save_settings['save_pt'],
                    save_gif=save_settings['save_gif'],
                    gif_fps=save_settings['gif_fps'],
                    background_color=save_settings['background_color'],
                )
                result_rgbd_dict[f'ours-{model_filename[idx]}'] = rgbd_images_pointersect
                total_time_dict[f'ours-{model_filename[idx]}'] = pointersect_result.total_time
                pointersect_result_names.append(f'ours-{model_filename[idx]}')

                # save the model info
                if pointersect_result.model_info is not None:
                    filename = os.path.join(pointersect_dir, 'model_info.json')
                    with open(filename, 'w') as f:
                        json.dump(pointersect_result.model_info, f, indent=2)

                if pointersect_result.total_time is not None:
                    filename = os.path.join(pointersect_dir, 'total_time.json')
                    with open(filename, 'w') as f:
                        json.dump(dict(total_time=pointersect_result.total_time), f, indent=2)

                if test_plane_normal and pointersect_result.intersection_plane_normals_w is not None:
                    if len(model_filename) == 1:
                        pointersect_dir = os.path.join(output_dir, f'pointersect-plane')
                    else:
                        pointersect_dir = os.path.join(output_dir, f'pointersect-plane-{model_filename[idx]}')
                    rgbd_images_pointersect = pointersect_result.get_rgbd_image(
                        camera=output_cameras, use_plane_normal=True)
                    rgbd_images_pointersect.save(
                        output_dir=pointersect_dir,
                        overwrite=save_settings['overwrite'],
                        save_png=save_settings['save_png'],
                        save_pt=save_settings['save_pt'],
                        save_gif=save_settings['save_gif'],
                        gif_fps=save_settings['gif_fps'],
                        background_color=save_settings['background_color'],
                    )
                    result_rgbd_dict[f'ours-plane-{model_filename[idx]}'] = rgbd_images_pointersect
                    total_time_dict[f'ours-plane-{model_filename[idx]}'] = pointersect_result.total_time
                    pointersect_result_names.append(f'ours-plane-{model_filename[idx]}')

                    # save the model info
                    if pointersect_result.model_info is not None:
                        filename = os.path.join(pointersect_dir, 'model_info.json')
                        with open(filename, 'w') as f:
                            json.dump(pointersect_result.model_info, f, indent=2)

                    if pointersect_result.total_time is not None:
                        filename = os.path.join(pointersect_dir, 'total_time.json')
                        with open(filename, 'w') as f:
                            json.dump(dict(total_time=pointersect_result.total_time), f, indent=2)

            except:
                print(traceback.format_exc())

    if render_npbgpp:
        print(f'---- Rendering with npbgpp ----')
        try:
            with torch.no_grad():
                out_dict = infer.render_rgbd_using_npbgpp(
                    input_rgbd_images=input_rgbd_images,
                    output_cameras=output_cameras.clone(),
                    # output_rgbd_images=gt_rgbd_images,
                )
                rgbd_images_npbgpp = out_dict['rgbd_image']
                total_time = out_dict['total_time']
        except:
            print(traceback.format_exc())
            rgbd_images_npbgpp = None
            total_time = None

        if rgbd_images_npbgpp is not None:
            # save npbgpp results
            npbgpp_dir = os.path.join(output_dir, 'npbgpp')
            rgbd_images_npbgpp.save(
                output_dir=npbgpp_dir,
                overwrite=save_settings['overwrite'],
                save_png=save_settings['save_png'],
                save_pt=save_settings['save_pt'],
                save_gif=save_settings['save_gif'],
                gif_fps=save_settings['gif_fps'],
                background_color=save_settings['background_color'],
            )

            if total_time is not None:
                filename = os.path.join(npbgpp_dir, 'total_time.json')
                with open(filename, 'w') as f:
                    json.dump(dict(total_time=total_time), f, indent=2)

        result_rgbd_dict['npbg++'] = rgbd_images_npbgpp
        total_time_dict[f'npbg++'] = total_time

    surfel_result_names = []
    if render_surfel:
        print(f'---- Rasterizing point cloud with surfel ----')
        stime_surfel = timer()
        tmp_output_cameras = copy.deepcopy(output_cameras)

        # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
        tmp_output_cameras.H_c2w[..., 0:1] = \
            tmp_output_cameras.H_c2w[..., 0:1] * tmp_output_cameras.intrinsic[..., 2:3, 2:3].sign()
        # tmp_output_cameras.H_c2w[..., 1:2] = \
        #     tmp_output_cameras.H_c2w[..., 1:2] * tmp_output_cameras.intrinsic[..., 1:2, 1:2].sign()
        tmp_output_cameras.H_c2w[..., 2:3] = \
            tmp_output_cameras.H_c2w[..., 2:3] * tmp_output_cameras.intrinsic[..., 2:3, 2:3].sign()

        rgb_shading_mode = 'raw' if input_point_cloud.rgb is not None else 'half'
        print(f'rgb_shading_mode = {rgb_shading_mode}')

        if isinstance(surfel_point_size, (float, int)):
            point_sizes = [surfel_point_size]
        else:
            point_sizes = surfel_point_size

        for point_size in point_sizes:
            print(f'point size: {point_size}')
            stime = timer()
            rgbd_images_surfel = input_point_cloud.rasterize_surfel(
                camera=tmp_output_cameras,
                point_size=point_size,
                render_normal_map=True,
                rgb_shading_mode=rgb_shading_mode,
            )
            total_time = timer() - stime
            print(f'finished, used {total_time:.2f} secs')

            # save surfel results
            surfel_dir = os.path.join(output_dir, f'surfel-{point_size:g}')
            rgbd_images_surfel.save(
                output_dir=surfel_dir,
                overwrite=save_settings['overwrite'],
                save_png=save_settings['save_png'],
                save_pt=save_settings['save_pt'],
                save_gif=save_settings['save_gif'],
                gif_fps=save_settings['gif_fps'],
                background_color=save_settings['background_color'],
            )

            if total_time is not None:
                filename = os.path.join(surfel_dir, 'total_time.json')
                with open(filename, 'w') as f:
                    json.dump(dict(total_time=total_time), f, indent=2)

            result_rgbd_dict[f'vis. splatting ({point_size:g})'] = rgbd_images_surfel
            total_time_dict[f'vis. splatting ({point_size:g})'] = total_time
            surfel_result_names.append(f'vis. splatting ({point_size:g})')

        total_time_surfel = timer() - stime_surfel
        print(f'rasterization finished, used {total_time_surfel:.2f} secs')

    if render_poisson:
        print(f'---- Poisson reconstruction  ----')
        print('running possion reconstruction..', flush=True)

        tmp_output_cameras = copy.deepcopy(output_cameras)

        # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
        tmp_output_cameras.H_c2w[..., 0:1] = \
            tmp_output_cameras.H_c2w[..., 0:1] * tmp_output_cameras.intrinsic[..., 2:3, 2:3].sign()
        # tmp_output_cameras.H_c2w[..., 1:2] = \
        #     tmp_output_cameras.H_c2w[..., 1:2] * tmp_output_cameras.intrinsic[..., 1:2, 1:2].sign()
        tmp_output_cameras.H_c2w[..., 2:3] = \
            tmp_output_cameras.H_c2w[..., 2:3] * tmp_output_cameras.intrinsic[..., 2:3, 2:3].sign()

        try:
            stime_poisson = timer()
            mesh: Mesh = input_point_cloud.get_mesh(
                bidx=0,
                method='poisson',
                recompute_normal=True,
                poisson_depth=8,
            )
            print('rendering rgbd images on the reconstructed mesh', flush=True)
            rgbd_images_poisson = mesh.get_rgbd_image(
                camera=tmp_output_cameras,
                render_normal_w=True,
                render_method='rasterization',
                camera_for_normal=output_cameras,
            )
            total_time = timer() - stime_poisson
            print(f'finished, used {total_time:.2f} secs')

            # save results
            poisson_dir = os.path.join(output_dir, 'poisson')
            os.makedirs(poisson_dir, exist_ok=True)
            o3d.io.write_triangle_mesh(
                filename=os.path.join(poisson_dir, 'reconstructed_mesh.ply'),
                mesh=mesh.mesh,
            )
            rgbd_images_poisson.save(
                output_dir=poisson_dir,
                overwrite=save_settings['overwrite'],
                save_png=save_settings['save_png'],
                save_pt=save_settings['save_pt'],
                save_gif=save_settings['save_gif'],
                gif_fps=save_settings['gif_fps'],
                background_color=save_settings['background_color'],
            )

            if total_time is not None:
                filename = os.path.join(poisson_dir, 'total_time.json')
                with open(filename, 'w') as f:
                    json.dump(dict(total_time=total_time), f, indent=2)

            result_rgbd_dict['poisson'] = rgbd_images_poisson
            total_time_dict[f'poisson'] = total_time
        except:
            print(traceback.format_exc())
            result_rgbd_dict['poisson'] = None
            total_time_dict[f'poisson'] = None

    if densify_neural_points:
        print(f'---- Densify with neural points ----')
        if input_point_cloud is None:
            input_point_cloud = input_rgbd_images.get_pcd(
                subsample=1,
                remove_background=True,
            )

        # print(f'gpu memory allocated: {torch.cuda.memory_allocated()/1e9:.3f} GB')
        try:
            with torch.no_grad():
                out_dict = infer.densify_with_neural_points(
                    point_cloud=input_point_cloud,
                    upsample_ratio_x48=neural_point_upsample_ratio_x48,
                )
                point_cloud_np = out_dict['point_cloud']
                total_time = out_dict['total_time']
        except:
            print(traceback.format_exc())
            point_cloud_np = None
            total_time = None

        # save npbgpp results
        if point_cloud_np is not None:
            np_dir = os.path.join(output_dir, 'neural_points')
            point_cloud_np.save(
                output_dir=os.path.join(np_dir, 'point_cloud'),
                overwrite=save_settings['overwrite'],
                save_pt=save_settings['save_pt'],
                save_ply=save_settings['save_ply'],
            )
            rgbd_images_np = point_cloud_np.rasterize_surfel(
                camera=output_cameras,
                point_size=0.1,
                default_rgb=[124. / 255, 200 / 255., 248. / 255],
                render_normal_map=True,
                rgb_shading_mode='half',
            )
            rgbd_images_np.save(
                output_dir=np_dir,
                overwrite=save_settings['overwrite'],
                save_png=save_settings['save_png'],
                save_pt=save_settings['save_pt'],
                save_gif=save_settings['save_gif'],
                gif_fps=save_settings['gif_fps'],
                background_color=save_settings['background_color'],
            )
            if total_time is not None:
                filename = os.path.join(np_dir, 'total_time.json')
                with open(filename, 'w') as f:
                    json.dump(dict(total_time=total_time), f, indent=2)
        else:
            rgbd_images_np = None
            total_time = None

        result_rgbd_dict['neural points'] = rgbd_images_np
        total_time_dict[f'neural points'] = total_time

    nglod_result_names = []
    if render_nglod:

        print(f'---- Training a NGLOD network ----')

        # assume if E[2,2] = -1, E = E_+ H_c2i, where H_c2i = eye([1, -1, -1, 1])
        # since kaolin-wisp does not support negative E[2,2], we incorporate it into
        # the extrinsic matrix => H_c2w' = H_c2w * eye([1, -1, -1, 1]).
        tmp_input_rgbd_images = copy.deepcopy(input_rgbd_images)
        tmp_input_cameras = tmp_input_rgbd_images.camera
        # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
        tmp_input_cameras.H_c2w[..., 1:2] = \
            tmp_input_cameras.H_c2w[..., 1:2] * tmp_input_cameras.intrinsic[..., 1:2, 1:2].sign()
        tmp_input_cameras.H_c2w[..., 2:3] = \
            tmp_input_cameras.H_c2w[..., 2:3] * tmp_input_cameras.intrinsic[..., 2:3, 2:3].sign()
        tmp_input_rgbd_images.camera = tmp_input_cameras

        # make sure intrinsic is all positive
        tmp_input_cameras.intrinsic = tmp_input_cameras.intrinsic.abs()

        tmp_output_cameras = copy.deepcopy(output_cameras)

        # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
        tmp_output_cameras.H_c2w[..., 1:2] = \
            tmp_output_cameras.H_c2w[..., 1:2] * tmp_output_cameras.intrinsic[..., 1:2, 1:2].sign()
        tmp_output_cameras.H_c2w[..., 2:3] = \
            tmp_output_cameras.H_c2w[..., 2:3] * tmp_output_cameras.intrinsic[..., 2:3, 2:3].sign()
        # make sure intrinsic is all positive
        tmp_output_cameras.intrinsic = tmp_output_cameras.intrinsic.abs()

        if gt_rgbd_images is not None:
            tmp_gt_rgbd_images = copy.deepcopy(gt_rgbd_images)
            tmp_gt_cameras = tmp_gt_rgbd_images.camera
            # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
            tmp_gt_cameras.H_c2w[..., 1:2] = \
                tmp_gt_cameras.H_c2w[..., 1:2] * tmp_gt_cameras.intrinsic[..., 1:2, 1:2].sign()
            tmp_gt_cameras.H_c2w[..., 2:3] = \
                tmp_gt_cameras.H_c2w[..., 2:3] * tmp_gt_cameras.intrinsic[..., 2:3, 2:3].sign()
            tmp_gt_rgbd_images.camera = tmp_gt_cameras

            # make sure intrinsic is all positive
            tmp_gt_cameras.intrinsic = tmp_gt_cameras.intrinsic.abs()
        else:
            tmp_gt_rgbd_images = gt_rgbd_images

        try:
            with torch.no_grad():
                out_dict = infer.render_rgbd_using_nglod(
                    input_rgbd_images=tmp_input_rgbd_images,
                    output_rgbd_images=tmp_gt_rgbd_images,
                    output_cameras=tmp_output_cameras,
                    total_epoch=total_nglod_epoch,
                    valid_every=nglod_test_every_iter,
                )
                total_time = out_dict['total_time']

            for i in range(len(out_dict['rgbd_images'])):
                rgbd_images_nglod = out_dict['rgbd_images'][i]
                iter = out_dict['iters'][i]

                if rgbd_images_nglod is not None:
                    # save npbgpp results
                    nglod_dir = os.path.join(output_dir, f'nglod-{iter}')
                    rgbd_images_nglod.save(
                        output_dir=nglod_dir,
                        overwrite=save_settings['overwrite'],
                        save_png=save_settings['save_png'],
                        save_pt=save_settings['save_pt'],
                        save_gif=save_settings['save_gif'],
                        gif_fps=save_settings['gif_fps'],
                        background_color=save_settings['background_color'],
                    )
                    nglod_result_names.append(f'nglod-{iter}')
                    result_rgbd_dict[f'nglod-{iter}'] = rgbd_images_nglod
                    total_time_dict[f'nglod-{iter}'] = total_time / float(out_dict['iters'][-1] + 1) * (iter + 1)

                    if total_time is not None:
                        filename = os.path.join(nglod_dir, 'total_time.json')
                        with open(filename, 'w') as f:
                            json.dump(
                                dict(
                                    total_time=total_time / float(out_dict['iters'][-1] + 1) * (iter + 1)),
                                f, indent=2)
        except:
            print(traceback.format_exc())
            result_rgbd_dict['nglod'] = None
            total_time_dict[f'nglod'] = None
            nglod_result_names.append('nglod')

    ngp_result_names = []
    if render_ngp:

        print(f'---- Training a NGP network ----')

        # assume if E[2,2] = -1, E = E_+ H_c2i, where H_c2i = eye([1, -1, -1, 1])
        # since kaolin-wisp does not support negative E[2,2], we incorporate it into
        # the extrinsic matrix => H_c2w' = H_c2w * eye([1, -1, -1, 1]).
        tmp_input_rgbd_images = copy.deepcopy(input_rgbd_images)
        tmp_input_cameras = tmp_input_rgbd_images.camera
        # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
        tmp_input_cameras.H_c2w[..., 1:2] = \
            tmp_input_cameras.H_c2w[..., 1:2] * tmp_input_cameras.intrinsic[..., 1:2, 1:2].sign()
        tmp_input_cameras.H_c2w[..., 2:3] = \
            tmp_input_cameras.H_c2w[..., 2:3] * tmp_input_cameras.intrinsic[..., 2:3, 2:3].sign()
        tmp_input_rgbd_images.camera = tmp_input_cameras

        # make sure intrinsic is all positive
        tmp_input_cameras.intrinsic = tmp_input_cameras.intrinsic.abs()

        tmp_output_cameras = copy.deepcopy(output_cameras)

        # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
        tmp_output_cameras.H_c2w[..., 1:2] = \
            tmp_output_cameras.H_c2w[..., 1:2] * tmp_output_cameras.intrinsic[..., 1:2, 1:2].sign()
        tmp_output_cameras.H_c2w[..., 2:3] = \
            tmp_output_cameras.H_c2w[..., 2:3] * tmp_output_cameras.intrinsic[..., 2:3, 2:3].sign()
        # make sure intrinsic is all positive
        tmp_output_cameras.intrinsic = tmp_output_cameras.intrinsic.abs()

        if gt_rgbd_images is not None:
            tmp_gt_rgbd_images = copy.deepcopy(gt_rgbd_images)
            tmp_gt_cameras = tmp_gt_rgbd_images.camera
            # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
            tmp_gt_cameras.H_c2w[..., 1:2] = \
                tmp_gt_cameras.H_c2w[..., 1:2] * tmp_gt_cameras.intrinsic[..., 1:2, 1:2].sign()
            tmp_gt_cameras.H_c2w[..., 2:3] = \
                tmp_gt_cameras.H_c2w[..., 2:3] * tmp_gt_cameras.intrinsic[..., 2:3, 2:3].sign()
            tmp_gt_rgbd_images.camera = tmp_gt_cameras

            # make sure intrinsic is all positive
            tmp_gt_cameras.intrinsic = tmp_gt_cameras.intrinsic.abs()
        else:
            tmp_gt_rgbd_images = gt_rgbd_images

        try:
            with torch.no_grad():
                out_dict = infer.render_rgbd_using_ngp(
                    input_rgbd_images=tmp_input_rgbd_images,
                    output_rgbd_images=tmp_gt_rgbd_images,
                    output_cameras=tmp_output_cameras,
                    total_epoch=total_ngp_epoch,
                    valid_every=ngp_test_every_iter,
                )
                total_time = out_dict['total_time']

            for i in range(len(out_dict['rgbd_images'])):
                rgbd_images_ngp = out_dict['rgbd_images'][i]
                iter = out_dict['iters'][i]

                if rgbd_images_ngp is not None:
                    # save npbgpp results
                    ngp_dir = os.path.join(output_dir, f'ngp-{iter}')
                    rgbd_images_ngp.save(
                        output_dir=ngp_dir,
                        overwrite=save_settings['overwrite'],
                        save_png=save_settings['save_png'],
                        save_pt=save_settings['save_pt'],
                        save_gif=save_settings['save_gif'],
                        gif_fps=save_settings['gif_fps'],
                        background_color=save_settings['background_color'],
                    )
                    ngp_result_names.append(f'ngp-{iter}')
                    result_rgbd_dict[f'ngp-{iter}'] = rgbd_images_ngp
                    total_time_dict[f'ngp-{iter}'] = total_time / float(out_dict['iters'][-1] + 1) * (iter + 1)

                    if total_time is not None:
                        filename = os.path.join(ngp_dir, 'total_time.json')
                        with open(filename, 'w') as f:
                            json.dump(
                                dict(
                                    total_time=total_time / float(out_dict['iters'][-1] + 1) * (iter + 1)
                                ), f, indent=2)

        except:
            print(traceback.format_exc())
            result_rgbd_dict['ngp'] = None
            total_time_dict[f'ngp'] = None
            ngp_result_names.append('ngp')

    dsnerf_result_names = []
    if render_dsnerf:

        print(f'---- Training a Depth-Supervised NeRF network ----')

        try:
            with torch.no_grad():
                out_dict = infer.render_rgbd_using_dsnerf(
                    input_rgbd_images=input_rgbd_images,
                    output_rgbd_images=gt_rgbd_images,
                    output_cameras=output_cameras,
                    total_iter=total_dsnerf_iter,
                    test_every_iter=dsnerf_test_every_iter,
                )
            total_time = out_dict['total_time']

            for i in range(len(out_dict['rgbd_images'])):
                rgbd_images_dsnerf = out_dict['rgbd_images'][i]
                iter = out_dict['iters'][i]

                if rgbd_images_dsnerf is not None:
                    # save npbgpp results
                    dsnerf_dir = os.path.join(output_dir, f'dsnerf-{iter}')
                    rgbd_images_dsnerf.save(
                        output_dir=dsnerf_dir,
                        overwrite=save_settings['overwrite'],
                        save_png=save_settings['save_png'],
                        save_pt=save_settings['save_pt'],
                        save_gif=save_settings['save_gif'],
                        gif_fps=save_settings['gif_fps'],
                        background_color=save_settings['background_color'],
                    )
                    dsnerf_result_names.append(f'dsnerf-{iter}')
                    result_rgbd_dict[f'dsnerf-{iter}'] = rgbd_images_dsnerf
                    total_time_dict[f'dsnerf-{iter}'] = total_time / float(out_dict['iters'][-1]) * iter

                    if total_time is not None:
                        filename = os.path.join(dsnerf_dir, 'total_time.json')
                        with open(filename, 'w') as f:
                            json.dump(dict(total_time=total_time / float(out_dict['iters'][-1]) * iter), f, indent=2)

        except:
            print(traceback.format_exc())
            rgbd_images_dsnerf = None
            total_time = None
            result_rgbd_dict['dsnerf'] = rgbd_images_dsnerf
            total_time_dict[f'dsnerf'] = total_time
            dsnerf_result_names.append('dsnerf')

    if render_ibrnet:

        print(f'---- Render with pretrained IBRNet ----')
        # assume if E[2,2] = -1, E = E_+ H_c2i, where H_c2i = eye([1, -1, -1, 1])
        # since kaolin-wisp does not support negative E[2,2], we incorporate it into
        # the extrinsic matrix => H_c2w' = H_c2w * eye([1, -1, -1, 1]).
        tmp_input_rgbd_images = copy.deepcopy(input_rgbd_images)
        tmp_input_cameras = tmp_input_rgbd_images.camera
        # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
        tmp_input_cameras.H_c2w[..., 1:2] = \
            tmp_input_cameras.H_c2w[..., 1:2] * tmp_input_cameras.intrinsic[..., 1:2, 1:2].sign()
        tmp_input_cameras.H_c2w[..., 2:3] = \
            tmp_input_cameras.H_c2w[..., 2:3] * tmp_input_cameras.intrinsic[..., 2:3, 2:3].sign()
        tmp_input_rgbd_images.camera = tmp_input_cameras

        # make sure intrinsic is all positive
        tmp_input_cameras.intrinsic = tmp_input_cameras.intrinsic.abs()

        tmp_output_cameras = copy.deepcopy(output_cameras)

        # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
        tmp_output_cameras.H_c2w[..., 1:2] = \
            tmp_output_cameras.H_c2w[..., 1:2] * tmp_output_cameras.intrinsic[..., 1:2, 1:2].sign()
        tmp_output_cameras.H_c2w[..., 2:3] = \
            tmp_output_cameras.H_c2w[..., 2:3] * tmp_output_cameras.intrinsic[..., 2:3, 2:3].sign()
        # make sure intrinsic is all positive
        tmp_output_cameras.intrinsic = tmp_output_cameras.intrinsic.abs()

        if gt_rgbd_images is not None:
            tmp_gt_rgbd_images = copy.deepcopy(gt_rgbd_images)
            tmp_gt_cameras = tmp_gt_rgbd_images.camera
            # if intrinsic[2,2] is negative, we flip the y and z axises in H_c2w
            tmp_gt_cameras.H_c2w[..., 1:2] = \
                tmp_gt_cameras.H_c2w[..., 1:2] * tmp_gt_cameras.intrinsic[..., 1:2, 1:2].sign()
            tmp_gt_cameras.H_c2w[..., 2:3] = \
                tmp_gt_cameras.H_c2w[..., 2:3] * tmp_gt_cameras.intrinsic[..., 2:3, 2:3].sign()
            tmp_gt_rgbd_images.camera = tmp_gt_cameras

            # make sure intrinsic is all positive
            tmp_gt_cameras.intrinsic = tmp_gt_cameras.intrinsic.abs()
        else:
            tmp_gt_rgbd_images = gt_rgbd_images

        try:
            with torch.no_grad():
                out_dict = infer.render_rgbd_using_ibrnet(
                    input_rgbd_images=tmp_input_rgbd_images,
                    output_rgbd_images=tmp_gt_rgbd_images,
                    output_cameras=tmp_output_cameras,
                    chunk_size=ibrnet_chunk_size,
                )
                total_time = out_dict['total_time']
                rgbd_images_ibrnet = out_dict['rgbd_image']

                if rgbd_images_ibrnet is not None:
                    # save npbgpp results
                    ibrnet_dir = os.path.join(output_dir, f'ibrnet')
                    rgbd_images_ibrnet.save(
                        output_dir=ibrnet_dir,
                        overwrite=save_settings['overwrite'],
                        save_png=save_settings['save_png'],
                        save_pt=save_settings['save_pt'],
                        save_gif=save_settings['save_gif'],
                        gif_fps=save_settings['gif_fps'],
                        background_color=save_settings['background_color'],
                    )

                    if total_time is not None:
                        filename = os.path.join(ibrnet_dir, 'total_time.json')
                        with open(filename, 'w') as f:
                            json.dump(
                                dict(total_time=total_time), f, indent=2)

                result_rgbd_dict['ibrnet'] = rgbd_images_ibrnet
                total_time_dict[f'ibrnet'] = total_time

        except:
            print(traceback.format_exc())
            result_rgbd_dict['ibrnet'] = None
            total_time_dict[f'ibrnet'] = None

    # compare and save
    preset_orders = ['surfel', 'poisson', 'neural points', 'npbg++', 'nglod', 'ngp', 'dsnerf', 'ibrnet', 'ours']
    preset_display_names = ['vis. splatting', 'poisson recon.', 'neural points', 'npbg++', 'nglod', 'ngp', 'dsnerf',
                            'ibrnet', 'ours']
    result_rgbd_images = []
    result_rgbd_images_names = []
    for i, preset in enumerate(preset_orders):
        if preset == 'ours':
            for idx in range(len(pointersect_result_names)):
                name = pointersect_result_names[idx]
                if name in result_rgbd_dict and result_rgbd_dict[name] is not None:
                    result_rgbd_images.append(result_rgbd_dict[name])
                    if len(pointersect_result_names) == 1:
                        result_rgbd_images_names.append(preset_display_names[i])
                    else:
                        result_rgbd_images_names.append(f'{name}')

        elif preset == 'surfel':
            for idx in range(len(surfel_result_names)):
                name = surfel_result_names[idx]
                if name in result_rgbd_dict and result_rgbd_dict[name] is not None:
                    result_rgbd_images.append(result_rgbd_dict[name])
                    if len(surfel_result_names) == 1:
                        result_rgbd_images_names.append(preset_display_names[i])
                    else:
                        result_rgbd_images_names.append(f'{name}')
        elif preset == 'nglod':
            for idx in range(len(nglod_result_names)):
                name = nglod_result_names[idx]
                if name in result_rgbd_dict and result_rgbd_dict[name] is not None:
                    result_rgbd_images.append(result_rgbd_dict[name])
                    if len(nglod_result_names) == 1:
                        result_rgbd_images_names.append(preset_display_names[i])
                    else:
                        result_rgbd_images_names.append(f'{name}')
        elif preset == 'ngp':
            for idx in range(len(ngp_result_names)):
                name = ngp_result_names[idx]
                if name in result_rgbd_dict and result_rgbd_dict[name] is not None:
                    result_rgbd_images.append(result_rgbd_dict[name])
                    if len(ngp_result_names) == 1:
                        result_rgbd_images_names.append(preset_display_names[i])
                    else:
                        result_rgbd_images_names.append(f'{name}')
        elif preset == 'dsnerf':
            for idx in range(len(dsnerf_result_names)):
                name = dsnerf_result_names[idx]
                if name in result_rgbd_dict and result_rgbd_dict[name] is not None:
                    result_rgbd_images.append(result_rgbd_dict[name])
                    if len(dsnerf_result_names) == 1:
                        result_rgbd_images_names.append(preset_display_names[i])
                    else:
                        result_rgbd_images_names.append(f'{name}')
        else:
            if preset in result_rgbd_dict and result_rgbd_dict[preset] is not None:
                result_rgbd_images.append(result_rgbd_dict[preset])
                result_rgbd_images_names.append(preset_display_names[i])

    if gt_rgbd_images is not None:
        print(f'computing metrics')
        rgb_metric = ['psnr', 'ssim', 'lpips']
        depth_metric = ['rmse']
        normal_metric = ['avg_angle']
        hit_metric = ['accuracy']
        pcd_metric = []
        filtered_pcd_metric = []

        filename = os.path.join(output_dir, 'metric.json')
        metric_dict = infer.compute_metrics_for_rgbd_images(
            rgbd_images=result_rgbd_images,
            ref_rgbd_image=gt_rgbd_images,
            ref_mesh=gt_mesh,
            names=result_rgbd_images_names,
            rgb_metric=rgb_metric,
            depth_metric=depth_metric,
            normal_metric=normal_metric,
            hit_metric=hit_metric,
            pcd_metric=pcd_metric,
            filtered_pcd_metric=filtered_pcd_metric,
            output_filename=filename,
        )
        metric_statistics_dict = dict()
        print('metrics:')
        pprint(metric_dict)
        print('')
        print('summary:')
        for category_name, dict_name, metric_names in [
            ['rgb', 'rgb_err_dicts', rgb_metric],
            ['depth', 'depth_err_dicts', depth_metric],
            ['normal', 'normal_err_dicts', normal_metric],
            ['hit', 'hit_err_dicts', hit_metric],
            ['pcd', 'pcd_err_dicts', pcd_metric],
            ['filtered_pcd', 'filtered_pcd_err_dicts', filtered_pcd_metric],
        ]:
            print(f'{category_name}:')
            for metric_name in metric_names:
                metric_statistics_dict[metric_name] = dict()
                print(f'  {metric_name}:')
                for name in result_rgbd_images_names:
                    try:
                        avg = metric_dict[dict_name][name][f'avg_{metric_name}']
                        std = metric_dict[dict_name][name][f'std_{metric_name}']
                    except:
                        avg = -1
                        std = -1

                    print(f'    {name}: {avg:.4f} +- {std:.4f}')
                    metric_statistics_dict[metric_name][name] = [avg, std]

    else:
        metric_dict = None
        metric_statistics_dict = None

    filename = os.path.join(output_dir, 'metric.json')
    if metric_dict is not None:
        with open(filename, 'w') as f:
            json.dump(
                metric_dict,
                f,
                indent=2,
                cls=print_and_save.NumpyJsonEncoder,
            )

    filename = os.path.join(output_dir, 'metric_stat.json')
    if metric_statistics_dict is not None:
        with open(filename, 'w') as f:
            json.dump(
                metric_statistics_dict,
                f,
                indent=2,
                cls=print_and_save.NumpyJsonEncoder,
            )

    print(f'saving summary images', flush=True)
    compare_dir = os.path.join(output_dir, 'compare')
    compare_rgbd_images = infer.compare_rgbd_images(
        rgbd_images=result_rgbd_images,
        names=result_rgbd_images_names,
        ref_rgbd_image=gt_rgbd_images,
        idx_ref=save_settings['idx_ref'],
        output_dir=compare_dir,
        overwrite=save_settings['overwrite'],
        save_png=save_settings['save_png'],
        save_pt=save_settings['save_pt'],
        save_gif=save_settings['save_gif'],
        gif_fps=save_settings['gif_fps'],
        background_color=save_settings['background_color'],
        font_color=None,
        font_size=save_settings['font_size'],
        ncols=6,
    )

    filename = os.path.join(output_dir, 'timing.json')
    with open(filename, 'w') as f:
        json.dump(
            dict(
                total_time=total_time_dict,
                num_images=output_cameras.H_c2w.size(0) * output_cameras.H_c2w.size(1),
                width_px=output_cameras.width_px,
                height_px=output_cameras.height_px,
            ),
            f, indent=2)

    for name in total_time_dict:
        try:
            print(
                f'{name}: \n'
                f'  total: {total_time_dict[name]:.3f} secs\n'
                f'  per {output_cameras.height_px}x{output_cameras.width_px} image: '
                f'{total_time_dict[name] / output_cameras.H_c2w.size(0) / output_cameras.H_c2w.size(1):.3f} secs')
        except:
            pass

    return dict(
        metric_dict=metric_dict,
        metric_statistics_dict=metric_statistics_dict,
        total_time_dict=total_time_dict,
    )


def render_hypersim(
        hypersim_volume_id: int,
        hypersim_scene_id: int,
        hypersim_image_subsample: int = 4,
        hypersim_num_images_per_item: int = None,
        hypersim_tonemap: bool = True,
        hypersim_tonemap_gamma: float = 1. / 2.2,
        hypersim_tonemap_percentile: float = 90,
        hypersim_tonemap_percentile_target: float = 0.7,
        hypersim_tonemap_scale: float = None,
        hypersim_num_hold_out_items: int = 0,
        hypersim_valid_only: bool = False,
        hypersim_dataset_root_dir: str = 'datasets/hypersim',
        hypersim_camera_csv_filename: str = 'datasets/hypersim/metadata_camera_parameters.csv',
        **kwargs,
):
    """Render rgbd images from ARKitScenes dataset."""

    scene_dir, scene_name = hypersim_dataset.get_hypersim_scene_info(
        volume_id=hypersim_volume_id,
        scene_id=hypersim_scene_id,
        dataset_root_dir=hypersim_dataset_root_dir,
    )
    all_camera_info = hypersim_dataset.load_hypersim_camera_info(
        filename=hypersim_camera_csv_filename,
    )

    dset = hypersim_dataset.HypersimDataset(
        scene_dir=scene_dir,
        camera_info=all_camera_info[scene_name],
        cam_idx=-1,
        num_images_per_item=hypersim_num_images_per_item,
        image_subsample=hypersim_image_subsample,
        tonemap=hypersim_tonemap,
        tonemap_gamma=hypersim_tonemap_gamma,
        tonemap_percentile=hypersim_tonemap_percentile,
        tonemap_percentile_target=hypersim_tonemap_percentile_target,
        tonemap_scale=hypersim_tonemap_scale,
        num_hold_out_items=0,
    )
    print(f'hypersim scene {scene_name}, intrinsic:')
    print(dset.intrinsic)

    output_dir = copy.deepcopy(kwargs.get('output_dir', None))
    ori_kwargs = copy.deepcopy(kwargs)

    # render each image set
    for i in range(len(dset)):
        kwargs = copy.deepcopy(ori_kwargs)

        data_dict = dset[i]
        input_rgbd_image: RGBDImage = data_dict['rgbd_image']

        n_imgs = input_rgbd_image.rgb.size(1)
        print(f'number of input images = {n_imgs}')

        if 'pr_setting' in kwargs:
            kwargs['pr_setting']['ray_radius'] = 0.1
        else:
            kwargs['pr_setting'] = dict(ray_radius=0.1)

        if 'save_settings' in kwargs:
            kwargs['save_settings']['save_pt'] = False
            # kwargs['save_settings']['save_gif'] = False
            # kwargs['save_settings']['save_ply'] = False
            # kwargs['save_settings']['save_png'] = False
        else:
            kwargs['save_settings'] = dict(
                save_pt=False,
                save_gif=False,
                save_ply=False,
                save_png=False,
            )

        if output_dir is not None:
            out_dir = os.path.join(output_dir, f'{i}')
            kwargs['output_dir'] = out_dir
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'config.json'), 'w') as f:
                json.dump(kwargs, f, indent=2)

        render_npbgpp = kwargs.get('render_npbgpp', False)  #
        if 'render_npbgpp' in kwargs:
            del kwargs['render_npbgpp']
        render_poisson = kwargs.get('render_poisson', False)  #
        if 'render_poisson' in kwargs:
            del kwargs['render_poisson']

        if not hypersim_valid_only:
            render_rgbd(
                input_rgbd_images=input_rgbd_image,  # (1, q)
                **kwargs,
                render_npbgpp=render_npbgpp,
                render_poisson=render_poisson,
                densify_neural_points=False,
                force_same_intrinsic=render_npbgpp,
            )

        if hypersim_num_hold_out_items > 0:
            print(f'rendering hold out set..', flush=True)

            all_index = torch.arange(input_rgbd_image.rgb.size(1), device=input_rgbd_image.rgb.device)  # (q,)
            tmp_input_rgbd_image = input_rgbd_image.index_select(
                dim=1, index=all_index[:-hypersim_num_hold_out_items])  # (1, q)
            tmp_gt_output_rgbd_image = input_rgbd_image.index_select(
                dim=1, index=all_index[-hypersim_num_hold_out_items:])  # (1, q)
            print(f'num input rgbd: {tmp_input_rgbd_image.rgb.size(1)}', flush=True)
            print(f'num output rgbd: {tmp_gt_output_rgbd_image.rgb.size(1)}', flush=True)
            del kwargs['n_output_imgs']
            del kwargs['output_camera_trajectory_mode']

            if output_dir is not None:
                out_dir = os.path.join(output_dir, f'{i}_valid')
                kwargs['output_dir'] = out_dir
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, 'config.json'), 'w') as f:
                    json.dump(kwargs, f, indent=2)

            render_rgbd(
                input_rgbd_images=tmp_input_rgbd_image,
                n_output_imgs=tmp_gt_output_rgbd_image.rgb.size(1),
                output_camera_trajectory_mode='gt',
                render_npbgpp=render_npbgpp,
                render_poisson=render_poisson,
                densify_neural_points=False,
                gt_rgbd_images=tmp_gt_output_rgbd_image,
                **kwargs,
            )

    kwargs['output_dir'] = output_dir


def batch_render_mesh(
        dataset_name: str,  # 'sketchfab-small', 'tex-models', 'shapenet', 'sketchfab-small-debug'
        config_dict: T.Dict[str, T.Any],
):
    """
    batch render multiple meshes
    """
    mesh_filename_dict: T.Dict[str, T.Any] = dataset_helper.get_dataset_mesh_filename_config()

    # download dataset
    dataset_filename_dict = mesh_filename_dict[dataset_name]
    filename_dict = dataset_helper.gather_and_clean_dataset(
        dataset_name=dataset_name,
        dataset_root_dir=dataset_filename_dict['dataset_root_dir'],
        cleaned_root_dir=dataset_filename_dict.get('cleaned_root_dir', None),
        clean_mesh=dataset_filename_dict.get('clean_mesh', False),
        train_mesh_filenames=dataset_filename_dict['train'],
        test_mesh_filenames=dataset_filename_dict.get('test', None),
        rank=0,
        world_size=1,
        printout=True,
    )
    mesh_filenames = filename_dict['test_mesh_filenames']

    print('mesh filenames:')
    for name in mesh_filenames:
        print(f'  {name}')
        assert os.path.exists(name)

    output_dir_root = config_dict.get('output_dir', 'batch_output')
    metric_dicts = []
    for i, mesh_filename in enumerate(mesh_filenames):
        config_dict['mesh_filename'] = mesh_filename
        dir_name = mesh_filename.replace('/', '_')
        config_dict['output_dir'] = os.path.join(
            output_dir_root,
            dir_name,
        )
        print(f"Rendering {config_dict['mesh_filename']}")
        print(f"  output_dir: {config_dict['output_dir']}")
        stime = timer()
        metric_dict = render_mesh(**config_dict)
        print(f"Finished {config_dict['mesh_filename']}")
        print(f"  total used: {(timer() - stime) / 60.:.3f} mins")

        metric_dicts.append(metric_dict)

    if len(metric_dicts) == 0:
        return

    # compile general statistics
    # metric_dict:
    #     rgb_err_dicts:
    #         model_name:
    #             metric_name:
    #                 list of vals
    #     depth_err_dicts:
    #     normal_err_dicts:
    #     hit_err_dicts:

    # collect model_names
    err_dict_names = [
        'rgb_err_dicts',
        'depth_err_dicts',
        'normal_err_dicts',
        'hit_err_dicts',
        'pcd_err_dicts',
        'filtered_pcd_err_dicts',
    ]
    metric_dict = metric_dicts[0]['metric_dict']
    model_names = set()
    for err_dict_name in err_dict_names:
        if metric_dict[err_dict_name] is not None:
            mnames = set(list(metric_dict[err_dict_name].keys()))
            model_names = model_names.union(mnames)
    print(f'model_names: {model_names}')

    # collect metric_names
    metric_names = dict()
    for err_dict_name in err_dict_names:
        metric_names[err_dict_name] = set()
        for mname in model_names:
            if metric_dict[err_dict_name][mname] is not None:
                me_names = set(list(metric_dict[err_dict_name][mname].keys()))
                metric_names[err_dict_name] = metric_names[err_dict_name].union(me_names)
    print(f'metric_names: {metric_names}')

    # collect all values
    all_value_dict = dict()
    #     rgb_err_dicts:
    #         model_name:
    #             metric_name:
    #                 list of vals
    #     depth_err_dicts:
    #     normal_err_dicts:
    #     hit_err_dicts:
    for i in range(len(metric_dicts)):
        metric_dict = metric_dicts[i]['metric_dict']
        for err_dict_name in err_dict_names:
            if err_dict_name not in all_value_dict:
                all_value_dict[err_dict_name] = dict()
            for mname in model_names:
                if mname not in all_value_dict[err_dict_name]:
                    all_value_dict[err_dict_name][mname] = dict()  # metric_name -> arr
                for metric_name in metric_names[err_dict_name]:
                    if metric_name not in all_value_dict[err_dict_name][mname]:
                        all_value_dict[err_dict_name][mname][metric_name] = []

                    try:
                        arr = metric_dict[err_dict_name][mname][metric_name]
                        if isinstance(arr, np.ndarray):
                            arr = np.reshape(arr, -1).tolist()
                        elif isinstance(arr, torch.Tensor):
                            arr = arr.detach().cpu().reshape(-1).numpy().tolist()

                        if isinstance(arr, (float, int)):
                            arr = [arr]
                        all_value_dict[err_dict_name][mname][metric_name] += arr
                    except:
                        pass
    # compute statistics
    statistics_dict = dict()
    #     rgb_err_dicts:
    #         metric_name:
    #             model_name:
    #                 avg
    #                 std
    #     depth_err_dicts:
    #     normal_err_dicts:
    #     hit_err_dicts:
    for err_dict_name in err_dict_names:
        statistics_dict[err_dict_name] = dict()
        for metric_name in metric_names[err_dict_name]:
            statistics_dict[err_dict_name][metric_name] = dict()
            for mname in model_names:
                statistics_dict[err_dict_name][metric_name][mname] = dict()
                arr = all_value_dict[err_dict_name][mname][metric_name]
                arr = np.array(arr)
                statistics_dict[err_dict_name][metric_name][mname]['avg'] = np.mean(arr)
                statistics_dict[err_dict_name][metric_name][mname]['std'] = np.std(arr)

    # save and print
    filename = os.path.join(output_dir_root, 'config.json')
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=2)

    filename = os.path.join(output_dir_root, 'all_values_dict.json')
    with open(filename, 'w') as f:
        json.dump(all_value_dict, f, indent=2)

    filename = os.path.join(output_dir_root, 'statistics.json')
    with open(filename, 'w') as f:
        json.dump(statistics_dict, f, indent=2)

    lines = []
    print('Statistics:')
    for err_dict_name in err_dict_names:
        lines.append(f'{err_dict_name}:')
        for metric_name in metric_names[err_dict_name]:
            lines.append(f'  {metric_name}:')
            for mname in model_names:
                lines.append(
                    f'    {mname}: '
                    f'{statistics_dict[err_dict_name][metric_name][mname]["avg"]:.4f} +- '
                    f'{statistics_dict[err_dict_name][metric_name][mname]["std"]:.4f} ')

    filename = os.path.join(output_dir_root, 'printout.txt')
    with open(filename, 'w') as f:
        line = '\n'.join(lines)
        f.write(line)

    for line in lines:
        print(line)


def render_arkitscenes(
        ply_filenames: T.List[str],
        arkitscenes_indiv_voxel_downsampling: bool = True,
        **kwargs,
):
    """
    Render the lidar scans in ARKitScenes dataset.

    Args:
        ply_filenames:
            list of ply filenames containing the point cloud

    """

    # load the point clouds
    voxel_downsample_cell_width = kwargs.get('voxel_downsample_cell_width', -1)
    voxel_downsample_sigma = kwargs.get('voxel_downsample_sigma', 0.5)

    all_point_clouds = []
    for i in range(len(ply_filenames)):
        ply_filename = ply_filenames[i]
        o3d_pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(ply_filename)

        point_cloud = PointCloud.from_o3d_pcd(o3d_pcd=o3d_pcd)
        max_xyz = point_cloud.xyz_w.max(dim=1)[0][0]  # (3,)
        min_xyz = point_cloud.xyz_w.min(dim=1)[0][0]  # (3,)
        size_xyz = max_xyz - min_xyz  # (3,)
        print(f'max_xyz: = {max_xyz}')
        print(f'min_xyz: = {min_xyz}')
        print(f'num_cells = {size_xyz / voxel_downsample_cell_width}')

        if arkitscenes_indiv_voxel_downsampling:
            point_cloud = point_cloud.voxel_downsampling(
                cell_width=voxel_downsample_cell_width,
                sigma=voxel_downsample_sigma,
            )
        all_point_clouds.append(point_cloud)

    input_point_cloud = PointCloud.cat(point_clouds=all_point_clouds, dim=1)  # (b=1, n)

    # shift point cloud to center
    delta_xyz_w = input_point_cloud.xyz_w[0].mean(dim=0)  # (3,)
    input_point_cloud.xyz_w = input_point_cloud.xyz_w - delta_xyz_w.reshape(1, 1, 3)

    if 'input_point_cloud' in kwargs:
        del kwargs['input_point_cloud']

    render_point_cloud(
        input_point_cloud=input_point_cloud,
        **kwargs,
    )


def render_point_cloud(
        input_point_cloud: T.Union[PointCloud, str],
        output_dir: str,
        model_filename: str,
        n_output_imgs: int,
        output_camera_trajectory_mode: str,
        # for pointersect
        k: int,
        # for comparison with baselines
        render_pointersect: bool = True,  # True,
        render_surfel: bool = True,
        render_poisson: bool = True,
        densify_neural_points: bool = False,
        # other settings
        rnd_seed: int = 0,
        output_camera_trajectory_params: T.Dict[str, T.Any] = None,
        output_camera_setting: T.Dict[str, T.Any] = None,
        save_settings: T.Dict[str, T.Any] = None,
        pr_setting: T.Dict[str, T.Any] = None,
        model_loading_settings: T.Dict[str, T.Any] = None,
        max_ray_chunk_size: int = int(1e4),  # k=40: int(4e4),
        max_pr_chunk_size: int = -1,
        max_model_chunk_size: int = -1,
        th_hit_prob: float = 0.5,
        surfel_point_size: float = 1.,
        neural_point_upsample_ratio_x48: int = 1,
        voxel_downsample_cell_width: float = -1,
        voxel_downsample_sigma: float = 0.5,
        output_camera_idxs: T.List[int] = None,
        test_plane_normal: bool = False,
        **kwargs,
):
    setting_dict = get_settings(
        output_camera_trajectory_params=output_camera_trajectory_params,
        output_camera_setting=output_camera_setting,
        save_settings=save_settings,
        pr_setting=pr_setting,
        model_loading_settings=model_loading_settings,
    )
    output_camera_trajectory_params = setting_dict['output_camera_trajectory_params']
    output_camera_setting = setting_dict['output_camera_setting']
    pr_setting = setting_dict['pr_setting']
    save_settings = setting_dict['save_settings']
    model_loading_settings = setting_dict['model_loading_settings']

    if os.path.exists(output_dir) and not save_settings['overwrite']:
        raise RuntimeError
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(input_point_cloud, str):
        # read pcd from file
        assert os.path.exists(input_point_cloud), f'{input_point_cloud} not exist'
        o3d_pcd = o3d.io.read_point_cloud(
            filename=input_point_cloud,
            remove_nan_points=True,
            remove_infinite_points=True,
        )
        # convert o3d_pcd to point cloud
        input_point_cloud = PointCloud.from_o3d_pcd(o3d_pcd=o3d_pcd)

    # voxel downsampling since we overlapped many point clouds
    input_point_cloud = input_point_cloud.voxel_downsampling(
        cell_width=voxel_downsample_cell_width,
        sigma=voxel_downsample_sigma,
        bidx=0,
    )

    # save input point_cloud
    print(f'saving input point cloud', flush=True)
    pcd_dir = os.path.join(output_dir, 'input_pcd')
    input_point_cloud.save(
        output_dir=pcd_dir,
        overwrite=save_settings['overwrite'],
        save_ply=True,
        save_pt=False,
    )

    data_device = torch.device('cpu')
    model_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create output camera
    output_camera_trajectory = CameraTrajectory(
        mode=output_camera_trajectory_mode,
        n_imgs=n_output_imgs,
        total=1,  # use all
        rng_seed=rnd_seed,
        params=output_camera_trajectory_params,
    )

    output_cameras = output_camera_trajectory.get_camera(
        fov=output_camera_setting['fov'],
        width_px=output_camera_setting['width_px'],
        height_px=output_camera_setting['height_px'],
        device=data_device,
    )

    if output_camera_idxs is not None:
        if isinstance(output_camera_idxs, int):
            output_camera_idxs = [output_camera_idxs]

        output_cameras = output_cameras.index_select(
            dim=1,
            index=torch.tensor(
                output_camera_idxs,
                dtype=torch.long,
                device=data_device,
            )
        )

    # save output camera trajectory
    print(f'saving output cameras', flush=True)
    output_camera_dir = os.path.join(output_dir, 'output_camera')
    output_cameras.save(
        output_dir=output_camera_dir,
        overwrite=save_settings['overwrite'],
        save_ply=save_settings['save_ply'],
        save_individual_ply=save_settings['save_ply'],
        save_pt=True,
        world_frame_size=1.,
        camera_frame_size=0.5,
        scene_meshes=None,
    )

    # sys.exit()
    print(f'main rendering', flush=True)
    main_render(
        render_pointersect=render_pointersect,
        render_npbgpp=False,
        render_surfel=render_surfel,
        render_nglod=False,
        render_ngp=False,
        render_dsnerf=False,
        render_ibrnet=False,
        render_poisson=render_poisson,
        densify_neural_points=densify_neural_points,
        input_rgbd_images=None,
        input_point_cloud=input_point_cloud,
        output_cameras=output_cameras,
        model_filename=model_filename,
        k=k,
        th_hit_prob=th_hit_prob,
        max_ray_chunk_size=max_ray_chunk_size,
        max_pr_chunk_size=max_pr_chunk_size,
        max_model_chunk_size=max_model_chunk_size,
        output_camera_setting=output_camera_setting,
        pr_setting=pr_setting,
        model_loading_settings=model_loading_settings,
        save_settings=save_settings,
        data_device=data_device,
        model_device=model_device,
        output_dir=output_dir,
        gt_rgbd_images=None,
        surfel_point_size=surfel_point_size,
        neural_point_upsample_ratio_x48=neural_point_upsample_ratio_x48,
        test_plane_normal=test_plane_normal,
    )


def render_point_clouds(
        input_point_clouds: T.Union[T.List[T.Union[PointCloud, str]], T.Tuple[T.Union[PointCloud, str]]],
        output_dir: str,
        **kwargs,
):
    if not isinstance(input_point_clouds, (list, tuple)):
        input_point_clouds = [input_point_clouds]

    total = len(input_point_clouds)
    for i in range(total):
        render_point_cloud(
            input_point_cloud=input_point_clouds[i],
            output_dir=os.path.join(output_dir, f'{i}'),
            **kwargs,
        )


def launch(
        config_filename: str,
):
    """
    launch the inference, using the parameters in the config yaml file.

    Args:
        config_filename:
            yaml filename
    """

    with open(config_filename) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    assert 'procedure' in config_dict
    procedure = config_dict['procedure']
    del config_dict['procedure']

    repo_root = os.path.normpath(os.path.join(__file__, '../../..'))


    # model filename
    default_model_pth_filename = os.path.normpath(
        os.path.join(
            repo_root, 'pointersect/checkpoint/epoch700.pth'))

    model_filename = config_dict['model_filename']
    if model_filename is None or model_filename == 'null':
        model_filename = default_model_pth_filename

    if isinstance(model_filename, str):
        model_filename = [model_filename]

    for i in range(len(model_filename)):
        if model_filename[i] is None or model_filename[i] == 'null':
            model_filename[i] = default_model_pth_filename
    config_dict['model_filename'] = model_filename

    if procedure == 'render_mesh':
        render_mesh(**config_dict)
    elif procedure == 'render_hypersim':
        render_hypersim(**config_dict)
    elif procedure == 'batch_render_mesh':
        batch_render_mesh(
            dataset_name=config_dict['dataset_name'],
            config_dict=config_dict,
        )
    elif procedure == 'render_arkitscenes':
        render_arkitscenes(
            **config_dict,
        )
    elif procedure == 'render_point_cloud':
        render_point_cloud(
            **config_dict,
        )
    elif procedure == 'render_point_clouds':
        render_point_clouds(
            **config_dict,
        )
    else:
        raise NotImplementedError

    print(f'Finished')


def render_pcd_with_pointersect(
        input_point_cloud: str,
        output_dir: str,
        model_filename: T.Optional[T.Union[str, T.List[str]]] = None,
        k: int = 40,
        output_camera_trajectory: T.Optional[str] = 'spiral',
        fov: float = 30,
        width_px: int = 200,
        height_px: int = 200,
        n_output_imgs: T.Optional[int] = None,
        ray_chunk_size_ratio: float = 1.,
        render_surfel: bool = True,
        render_poisson: bool = True,
        **kwargs,
):
    """
    Convenient function to render a point cloud with pointersect.

    Args:
        input_point_cloud:
            the filename of a ply file containing the point cloud
        output_dir:
            folder storing all outputs
        model_filename:
            a pt checkpoint file containing the pretrained pointersect model.
            If `None`, the function uses the default pointersect model.
        k:
            number of neighboring points used in the rendering per ray
        output_camera_trajectory:
            The camera trajectory to render the point cloud.
            It can be a string containing the preset trajectory
            (see :py:`pointersect.structures.CameraTrajectory`),
            or it can be a json file in the format described below.
            If `None`, it uses `spiral` with 144 images.

            Json file format:
                H_c2w:
                    (b, q, 4, 4), a nested list containing the camera pose in the world coord.
                    `b` is the batch dimension, `q` is number of camera poses in a batch.
                    For example, `H_c2w[i,j]` is the 4x4 camera pose matrix that converts
                    a point in the camera coordinate to the world coordinate.

        fov:
            the horizontal field of view in degree of the output images
        width_px:
            number of horizontal pixels in the rendered image
        height_px:
            number of vertial pixels in the rendered image

        n_output_imgs:
            Total number of output images. It will interpolate the camera trajectory uniformly.
            If `None` and `output_camera_trajectory` is provided as a json file, it will use the
            camera poses in the json file.

        ray_chunk_size_ratio:
            The setting controls the number of rays to render per batch. We use some simple
            logic to determine the number to try to fill the memory as much as possible.
            If run out of memory, please decrease the number. For example, if set to `0.5`,
            it will render half of the number of rays in a batch at once.

        render_surfel:
            whether to render the point cloud using visibility splatting

        render_poisson:
            whether to render the point cloud by first reconstructing a mesh using
            screened poisson reconstruction.
    """

    repo_root = os.path.normpath(os.path.join(__file__, '../../..'))

    # load the default settings
    config_filename = os.path.normpath(os.path.join(
        repo_root, 'pointersect/inference/configs/config_default.yaml'))
    with open(config_filename) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    config_dict['input_point_cloud'] = input_point_cloud
    config_dict['output_dir'] = output_dir

    # model filename
    default_model_pth_filename = os.path.normpath(os.path.join(
            repo_root, 'pointersect/checkpoint/epoch700.pth'))

    if model_filename is None:
        model_filename = default_model_pth_filename

    if isinstance(model_filename, str):
        model_filename = [model_filename]

    for i in range(len(model_filename)):
        if model_filename[i] is None or model_filename[i] == 'null':
            model_filename[i] = default_model_pth_filename
    config_dict['model_filename'] = model_filename

    # k
    config_dict['k'] = k

    # ray chunk size
    max_ray_chunk_size = inference_utils.get_pointersect_max_ray_chunk_size(k=config_dict['k'])
    config_dict['max_ray_chunk_size'] = max(1, int(max_ray_chunk_size * float(ray_chunk_size_ratio)))

    # trajectory
    if output_camera_trajectory is not None:
        config_dict['output_camera_trajectory_mode'] = output_camera_trajectory

    # camera setting
    config_dict['output_camera_setting']['fov'] = fov
    config_dict['output_camera_setting']['width_px'] = width_px
    config_dict['output_camera_setting']['height_px'] = height_px

    # replace n_output_imgs to the default value if not a camera path
    _, ext = os.path.splitext(config_dict['output_camera_trajectory_mode'])
    if len(ext) == 0:
        if n_output_imgs is None:
            n_output_imgs = 36
    config_dict['n_output_imgs'] = n_output_imgs

    config_dict['render_surfel'] = render_surfel
    config_dict['render_poisson'] = render_poisson

    for key in kwargs:
        config_dict[key] = kwargs[key]

    return render_point_cloud(**config_dict)


def main_pcd():
    fire.Fire(render_pcd_with_pointersect)

def main_full():
    fire.Fire(launch)


if __name__ == '__main__':
    fire.Fire()
