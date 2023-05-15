#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
import argparse
import os
import typing as T
import torch

import pointersect.meta_script.meta_script_utils as meta_utils
from pointersect.data.genlist import (
    get_tex_model_list,
)


def get_defualt_config() -> T.Dict[str, T.Any]:
    filename = os.path.normpath(
        os.path.join(
            os.path.abspath(__file__),
            '../configs/pbnr_default.yaml',
        ))
    return meta_utils.read_config(filename)


def get_defualt_mesh_filename_config() -> T.Dict[str, T.Any]:
    filename = os.path.normpath(
        os.path.join(
            os.path.abspath(__file__),
            '../configs/mesh_filenames.yaml',
        ))
    print(f'default mesh filename config: {filename}')
    return meta_utils.read_config(filename)


def get_mesh_filenames(
        dataset_name: str,
) -> (list, list):
    """
    Returns:
        two list of mesh filenames (for training and test)
    """

    mesh_filename_dict = get_defualt_mesh_filename_config()

    if dataset_name.lower() == 'tex':
        train_filename = \
            get_tex_model_list(
                setting='train',
                num_classes=3,
                rnd_seed=-1
            )
        test_filename = \
            get_tex_model_list(
                setting='test',
                num_classes=3,
                rnd_seed=-1
            )
        dataset_root_dir = 'datasets/tex-models'
    elif dataset_name.lower() == 'shapenet':
        train_filename = mesh_filename_dict['shapenet']['train']
        test_filename = mesh_filename_dict['shapenet']['test']
        dataset_root_dir = mesh_filename_dict['shapenet']['dataset_root_dir']
    elif dataset_name.lower() == 'sketchfab':
        train_filename = mesh_filename_dict['sketchfab']['train']
        test_filename = mesh_filename_dict['sketchfab']['test']
        dataset_root_dir = mesh_filename_dict['sketchfab']['dataset_root_dir']
    elif dataset_name.lower() == 'sketchfab-small':
        train_filename = mesh_filename_dict['sketchfab-small']['train']
        test_filename = mesh_filename_dict['sketchfab-small']['test']
        dataset_root_dir = mesh_filename_dict['sketchfab-small']['dataset_root_dir']
    else:
        raise NotImplementedError

    return train_filename, test_filename, dataset_root_dir


def set_dataset(
        config_dict: T.Dict[str, T.Any],
        dataset_name: str,
        total: int = 10000,
        max_angle: float = 180,
        # max_translate_ratio: float = 6.0,
        # ray_perturbation_angle: float = 0.75,
        width_px: int = 300,
        height_px: int = 300,
        target_width_px: int = 80,
        target_height_px: int = 80,
        k: int = 40,
        local_max_angle: float = 3.,
        n_imgs: int = 30,
        min_subsample: int = 1,
        max_subsample: int = 30,
        min_k_ratio: float = 1.,
        max_k_ratio: float = 1.,
        mesh_scale: float = 1.,
        min_r: float = 0.5,  # min_r used for output camera
        max_r: float = 3.,  # max_r used for output camera
        rand_r: float = 0.,
        texture_mode: str = 'ori',  # 'files', 'imagenet'
        texture_crop_method: T.Union[int, str] = 'ori',  # or an int p indiciating the min p * p crop
        texture_filenames: T.List[str] = None,
        render_method: str = 'ray_cast',  # 'rasterization'
        num_threads: int = 10,
        use_bucket_sampler: bool = True,
        mix_meshes: bool = False,
        min_num_mesh: int = 1,
        max_num_mesh: int = 2,
        radius_scale: float = 2.,
        total_combined: int = None,
) -> T.Dict[str, T.Any]:
    config_dict['dataset_info']['dataset_name'] = dataset_name
    mesh_filename, test_mesh_filename, dataset_root_dir = get_mesh_filenames(dataset_name)
    config_dict['dataset_info']['dataset_root_dir'] = dataset_root_dir
    config_dict['dataset_info']['mesh_filename'] = mesh_filename
    config_dict['dataset_info']['test_mesh_filename'] = test_mesh_filename
    config_dict['dataset_info']['total'] = total
    config_dict['dataset_info']['max_angle'] = max_angle
    # config_dict['dataset_info']['max_translate_ratio'] = max_translate_ratio
    # config_dict['dataset_info']['ray_perturbation_angle'] = ray_perturbation_angle
    config_dict['dataset_info']['width_px'] = width_px
    config_dict['dataset_info']['height_px'] = height_px
    config_dict['dataset_info']['target_width_px'] = target_width_px
    config_dict['dataset_info']['target_height_px'] = target_height_px

    config_dict['dataset_info']['k'] = k
    config_dict['dataset_info']['local_max_angle'] = local_max_angle
    config_dict['dataset_info']['n_imgs'] = n_imgs

    config_dict['dataset_info']['min_subsample'] = min_subsample
    config_dict['dataset_info']['max_subsample'] = max_subsample
    config_dict['dataset_info']['min_k_ratio'] = min_k_ratio
    config_dict['dataset_info']['max_k_ratio'] = max_k_ratio
    config_dict['dataset_info']['mesh_scale'] = mesh_scale
    config_dict['dataset_info']['min_r'] = min_r
    config_dict['dataset_info']['max_r'] = max_r
    config_dict['dataset_info']['rand_r'] = rand_r

    config_dict['dataset_info']['texture_mode'] = texture_mode
    config_dict['dataset_info']['texture_crop_method'] = texture_crop_method
    config_dict['dataset_info']['texture_filenames'] = texture_filenames

    config_dict['dataset_info']['render_method'] = render_method
    config_dict['dataset_info']['num_threads'] = num_threads
    config_dict['dataset_info']['use_bucket_sampler'] = use_bucket_sampler

    # for mix and match meshes
    config_dict['dataset_info']['mix_meshes'] = mix_meshes
    config_dict['dataset_info']['min_num_mesh'] = min_num_mesh
    config_dict['dataset_info']['max_num_mesh'] = max_num_mesh
    config_dict['dataset_info']['radius_scale'] = radius_scale
    config_dict['dataset_info']['total_combined'] = total_combined

    return config_dict


def set_optim_info(
        config_dict: T.Dict[str, T.Any],
        random_drop_rgb_rate: float = 0.5,
        random_drop_sample_feature_rate: float = 0.5,
        optim_method: str = 'adam',
        loss_weight_t: float = 10.,
        loss_weight_t_l1: float = 0.,
        loss_weight_normal: float = 1.,
        loss_weight_normal_l1: float = 0.,
        loss_weight_plane_normal: float = 0.,
        loss_weight_plane_normal_l1: float = 0.,
        loss_weight_hit: float = 1.0,
        loss_weight_rgb: float = 1.0,
        loss_weight_rgb_normal: float = 0,
        loss_weight_rgb_normal_dot: float = 0,
        loss_weight_rgb_normal_dot_l1: float = 0,
        pcd_noise_std: float = 0,
) -> T.Dict[str, T.Any]:
    config_dict['optim_info']['random_drop_rgb_rate'] = random_drop_rgb_rate
    config_dict['optim_info']['random_drop_sample_feature_rate'] = random_drop_sample_feature_rate
    config_dict['optim_info']['optim_method'] = optim_method
    config_dict['optim_info']['loss_weight_t'] = loss_weight_t
    config_dict['optim_info']['loss_weight_t_l1'] = loss_weight_t_l1
    config_dict['optim_info']['loss_weight_normal'] = loss_weight_normal
    config_dict['optim_info']['loss_weight_normal_l1'] = loss_weight_normal_l1
    config_dict['optim_info']['loss_weight_plane_normal'] = loss_weight_plane_normal
    config_dict['optim_info']['loss_weight_plane_normal_l1'] = loss_weight_plane_normal_l1
    config_dict['optim_info']['loss_weight_hit'] = loss_weight_hit
    config_dict['optim_info']['loss_weight_rgb'] = loss_weight_rgb
    config_dict['optim_info']['loss_weight_rgb_normal'] = loss_weight_rgb_normal
    config_dict['optim_info']['loss_weight_rgb_normal_dot'] = loss_weight_rgb_normal_dot
    config_dict['optim_info']['loss_weight_rgb_normal_dot_l1'] = loss_weight_rgb_normal_dot_l1
    config_dict['optim_info']['pcd_noise_std'] = pcd_noise_std

    return config_dict


def set_model_info(
        config_dict: T.Dict[str, T.Any],
        dim: int = 512,
        dropout: float = 0.1,
        num_layers: int = 4,
        use_rgb_as_input: bool = True,
        use_dpsuv_as_input: bool = True,
        use_zdir_as_input: bool = True,
        num_heads: int = 4,
        dim_input_layers: T.List[int] = None,  # dimension of the linear layers (nLayer-1)
        use_vdir_as_input: bool = False,  # if true, use camera viewing direction (1 vector, 3 dim) as input
        use_rgb_indicator: bool = False,  # whether to add a binary indicator saying input has valid rgb
        use_feature_indicator: bool = False,  # whether to add a binary indicator saying input has valid feature
        estimate_surface_normal_weights: bool = False,
) -> T.Dict[str, T.Any]:
    config_dict['model_info']['dim_feature'] = dim
    config_dict['model_info']['dim_mlp'] = dim
    config_dict['model_info']['dropout'] = dropout
    config_dict['model_info']['num_layers'] = num_layers
    config_dict['model_info']['use_rgb_as_input'] = use_rgb_as_input
    config_dict['model_info']['use_dpsuv_as_input'] = use_dpsuv_as_input
    config_dict['model_info']['use_zdir_as_input'] = use_zdir_as_input
    config_dict['model_info']['num_heads'] = num_heads

    config_dict['model_info']['dim_input_layers'] = dim_input_layers
    config_dict['model_info']['use_vdir_as_input'] = use_vdir_as_input
    config_dict['model_info']['use_rgb_indicator'] = use_rgb_indicator
    config_dict['model_info']['use_feature_indicator'] = use_feature_indicator

    config_dict['model_info']['estimate_surface_normal_weights'] = estimate_surface_normal_weights

    return config_dict


def set_process_info(
        config_dict: T.Dict[str, T.Any],
        key: str,
        val: T.Any,
):
    config_dict['process_info'][key] = val
    return config_dict


if __name__ == '__main__':

    config_filename = 'pointersect/meta_script/configs/pbnr.yaml'

    # create config file
    config_dict = get_defualt_config()

    # set dataset
    config_dict = set_dataset(
        config_dict=config_dict,
        dataset_name='sketchfab-small',
        total=10000,
        max_angle=180,
        width_px=300,
        height_px=300,
        target_width_px=50,
        target_height_px=50,
        k=40,
        local_max_angle=3.,
        n_imgs=30,
        min_subsample=1,
        max_subsample=10,
        min_k_ratio=0.3,
        max_k_ratio=5.,
        mesh_scale=1.,
        min_r=0.5,
        max_r=3.,
        rand_r=0.5,
        texture_mode='imagenet',
        texture_crop_method='20-200',
        render_method='ray_cast',
        num_threads=10,
        use_bucket_sampler=True,
        mix_meshes=False,
        min_num_mesh=1,
        max_num_mesh=2,
        radius_scale=2.,
        total_combined=None,
    )

    # set model
    config_dict = set_model_info(
        config_dict=config_dict,
        dim=64,  # 512,
        dropout=0.1,
        num_layers=4,
        use_rgb_as_input=True,
        use_dpsuv_as_input=True,
        use_zdir_as_input=True,
        use_vdir_as_input=True,
        use_rgb_indicator=True,
        use_feature_indicator=True,
        dim_input_layers=[32, 64],
        estimate_surface_normal_weights=False,
    )

    # set optim
    config_dict = set_optim_info(
        config_dict=config_dict,
        random_drop_rgb_rate=0.5,  # 0
        random_drop_sample_feature_rate=0.5,  # 0
        optim_method='adam_tf',  # 'adam'
        loss_weight_rgb_normal=0.,
        loss_weight_rgb_normal_dot=0.,
        pcd_noise_std=0.,
    )

    # save to artifact
    config_dict['process_info']['output_dir'] = 'artifacts'

    # ------------------------------------------

    name = 'loss_weight_plane_normal'
    vals = [0, ]

    ngpus = torch.cuda.device_count()
    assert ngpus > 0

    task_urls = []
    for val in vals:

        config_dict['optim_info'][name] = val

        # ------------------------------------------
        model_info = config_dict['model_info']
        if model_info['use_rgb_indicator']:
            assert model_info['use_rgb_as_input']
        if model_info['use_feature_indicator']:
            assert model_info['use_dist_as_input'] or model_info['use_zdir_as_input'] or \
                   model_info['use_dps_as_input'] or model_info['use_dpsuv_as_input'] or \
                   model_info['use_vdir_as_input']

        # write config file
        meta_utils.write_config_file(
            filename=config_filename,
            config_dict=config_dict,
        )

        # create command
        param_dict = dict(
            config_filename=config_filename,
        )
        switch_names = []

        cmd = meta_utils.compile_command(
            script_filename='pointersect/script/train_v2.py',
            num_gpus=ngpus,
            params=param_dict,
            switch_names=switch_names,
            use_xvfb=True,
        )

        print(cmd)

    print('summary: -------------------------------')
    for i in range(len(vals)):
        print(f'{name} = {vals[i]}')
        if len(task_urls) > 0:
            print(f'{task_urls[i]}')
