#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#


import argparse
import typing as T
from timeit import default_timer as timer

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from cdslib.core.data.dataloader import distributed_bucket_sampler
from cdslib.core.nn import FocalLoss
from cdslib.core.optim import TFOptimizer
from cdslib.core.script.base_train import BaseTrainProcess
from cdslib.core.utils.multigpu_utils import reduce_tensor
from cdslib.core.utils.print_and_save import Logger
from plib import utils
from pointersect.data import dataset_helper
from pointersect.data import mesh_dataset_v2
from pointersect.inference import infer
from pointersect.inference import structures
from pointersect.models import pointersect


class TrainPointersectProcess(BaseTrainProcess):

    def __init__(
            self,
            ## dataset_info
            dataset_name: str = 'tex',  # name of the dataset to train on, determines which dataset to download
            dataset_root_dir: str = 'datasets/tex-models',  # where the meshes are
            mesh_filename: T.Union[str, T.List[str]] = 'bunny.obj',
            test_mesh_filename: str = 'cat.obj',
            batch_size: int = 2,
            n_target_imgs: int = 2,
            n_imgs: int = 3,
            width_px: int = 200,
            height_px: int = 200,
            target_width_px: int = 20,
            target_height_px: int = 20,
            fov: int = 60.,
            max_angle: float = 30.,
            local_max_angle: float = 3.,
            max_translate_ratio: float = 2.0,  # not used
            ray_perturbation_angle: float = 3,  # not used
            total: int = 10000,
            pcd_subsample: int = 1,  # not used, replaced by min_subsample
            dataset_rng_seed: int = 0,
            k: int = 40,
            randomize_translate: bool = False,
            # not used  # whether translation amount is randomized, see utils.rectify_points
            ray_radius: float = 0.1,  # radius of the ray, used in pr
            num_threads: int = 0,
            train_cam_path_mode: str = 'random',  # not used  # random/circle # support different camera trajectory
            generate_point_cloud_input: bool = False,  # not used
            clean_mesh: bool = True,  # not used  # if true, clean the obj file
            cleaned_root_dir: str = 'datasets/cleaned_models',  # not used  # where the cleaned obj meshes are saved
            skip_existed_cleaned_mesh: bool = False,  # if true, will not clean the obj file again if existed
            render_method: str = 'ray_cast',  # 'ray_cast', 'rasterization'
            min_subsample: int = 1,
            max_subsample: int = 1,  # None: same as min_subsample
            min_k_ratio: float = 1.,
            max_k_ratio: float = 1.,  # None: same as max_k_ratio
            mesh_scale: float = 1.,
            min_r: float = 0.5,
            max_r: float = 3.,
            rand_r: float = 0.,
            texture_mode: str = 'ori',  # 'files', 'imagenet'
            texture_crop_method: T.Union[int, str] = 'ori',  # or an int p indiciating the min p * p crop
            texture_filenames: T.List[str] = None,
            use_bucket_sampler: bool = True,
            mix_meshes: bool = False,
            min_num_mesh: int = 1,
            max_num_mesh: int = 2,
            radius_scale: float = 2.,
            total_combined: int = None,
            ## model_info
            learn_dist: bool = False,
            num_layers: int = 4,  # 4,  # 3,
            dim_feature: int = 512,  # 256,
            num_heads: int = 4,
            encoding_type: str = 'pos',  # pos/ siren  # support different ways of position encoding
            positional_encoding_num_functions: int = 10,  # to turn off position encoding, set to 0
            positional_encoding_include_input: bool = True,
            positional_encoding_log_sampling: bool = True,
            nonlinearity: str = 'silu',
            dim_mlp: int = 512,  # 1024,  # 512,
            dropout: float = 0.1,
            direction_param: str = 'norm_vec',
            estimate_surface_normal_weights: bool = False,
            estimate_image_rendering_weights: bool = True,
            use_rgb_as_input: bool = False,
            use_dist_as_input: bool = False,  # if true, use |x|,|y|,|z| and sqrt(x^2+y^2) in ray space as input
            use_zdir_as_input: bool = False,  # if true, use camera viewing direction (2 vector, 3 dim) as input
            use_dps_as_input: bool = False,  # if true, use local frame width (1 value, 1 dim) as input
            use_dpsuv_as_input: bool = False,  # if true, use local frame (2 vectors, 6 dim) as input
            use_layer_norm: bool = False,  # if true, enable layer norm
            use_pr: bool = False,  # if true, use pr to find neighbor points within a fixed distance to ray
            use_additional_invalid_token: bool = False,  # if true, an extra invalid token will be used in transformer
            dim_input_layers: T.List[int] = None,
            use_vdir_as_input: bool = False,
            use_rgb_indicator: bool = False,  # whether to add a binary indicator saying input has valid rgb
            use_feature_indicator: bool = False,  # whether to add a binary indicator saying input has valid feature
            ## optim_info
            optim_method: str = 'adam',  # 'adam_tf'
            learning_rate: float = 1.0e-4,
            lr_factor: float = 0.1,
            num_warmup_steps: int = 4000,
            max_grad_val: float = 1.0,
            use_amp: bool = False,
            loss_weight_t: float = 10.,
            loss_weight_t_l1: float = 0.,
            loss_weight_normal: float = 1.,
            loss_weight_normal_l1: float = 0.,
            loss_weight_plane_normal: float = 1.,
            loss_weight_plane_normal_l1: float = 0.,
            loss_weight_hit: float = 1.0,
            loss_weight_rgb: float = 1.0,
            loss_weight_rgb_normal: float = 0,
            loss_weight_rgb_normal_dot: float = 0,
            loss_weight_rgb_normal_dot_l1: float = 0,
            loss_rgb_type: str = 'l1',  # 'l2'
            focal_loss_gamma: float = 2.0,
            focal_loss_alpha: float = 0.5,
            learn_ray_rgb: bool = True,
            random_drop_rgb_rate: float = 0,  # probability that the rgb will be randomly dropped
            random_drop_sample_feature_rate: float = 0,  # probability that zdir, dps, dpsuv will be randomly dropped
            pcd_noise_std: float = 0,  # std of the gaussian noise added to the input point cloud
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # set default values
        self.dataset_info = dict(
            dataset_name=dataset_name,
            dataset_root_dir=dataset_root_dir,
            mesh_filename=mesh_filename,
            test_mesh_filename=test_mesh_filename,
            batch_size=batch_size,
            n_target_imgs=n_target_imgs,
            n_imgs=n_imgs,
            width_px=width_px,
            height_px=height_px,
            target_width_px=target_width_px,
            target_height_px=target_height_px,
            fov=fov,
            max_angle=max_angle,
            local_max_angle=local_max_angle,
            max_translate_ratio=max_translate_ratio,
            ray_perturbation_angle=ray_perturbation_angle,
            total=total,
            pcd_subsample=pcd_subsample,
            dataset_rng_seed=dataset_rng_seed,
            k=k,
            randomize_translate=randomize_translate,
            ray_radius=ray_radius,
            num_threads=num_threads,
            train_cam_path_mode=train_cam_path_mode,
            generate_point_cloud_input=generate_point_cloud_input,
            clean_mesh=clean_mesh,
            cleaned_root_dir=cleaned_root_dir,
            skip_existed_cleaned_mesh=skip_existed_cleaned_mesh,
            render_method=render_method,
            max_subsample=max_subsample if max_subsample is not None else min_subsample,
            min_subsample=min_subsample,
            min_k_ratio=min_k_ratio,
            max_k_ratio=max_k_ratio if max_k_ratio is not None else min_k_ratio + 1e-6,
            mesh_scale=mesh_scale,
            rand_r=rand_r,
            min_r=min_r,
            max_r=max_r,
            texture_mode=texture_mode,
            texture_crop_method=texture_crop_method,
            texture_filenames=texture_filenames,
            use_bucket_sampler=use_bucket_sampler,
            mix_meshes=mix_meshes,
            min_num_mesh=min_num_mesh,
            max_num_mesh=max_num_mesh,
            radius_scale=radius_scale,
            total_combined=total_combined,
        )

        if self.dataset_info.get('max_subsample', 1) != self.dataset_info.get('min_subsample', 1):
            assert self.dataset_info['batch_size'] == 1

        self.model_info = dict(
            learn_dist=learn_dist,
            num_layers=num_layers,
            dim_feature=dim_feature,
            num_heads=num_heads,
            encoding_type=encoding_type,
            positional_encoding_num_functions=positional_encoding_num_functions,
            positional_encoding_include_input=positional_encoding_include_input,
            positional_encoding_log_sampling=positional_encoding_log_sampling,
            nonlinearity=nonlinearity,
            dim_mlp=dim_mlp,
            dropout=dropout,
            direction_param=direction_param,
            estimate_surface_normal_weights=estimate_surface_normal_weights,
            estimate_image_rendering_weights=estimate_image_rendering_weights,
            use_rgb_as_input=use_rgb_as_input,
            use_dist_as_input=use_dist_as_input,
            use_zdir_as_input=use_zdir_as_input,
            use_dps_as_input=use_dps_as_input,
            use_dpsuv_as_input=use_dpsuv_as_input,
            use_layer_norm=use_layer_norm,
            use_pr=use_pr,
            use_additional_invalid_token=use_additional_invalid_token,
            dim_input_layers=dim_input_layers,
            use_vdir_as_input=use_vdir_as_input,
            use_rgb_indicator=use_rgb_indicator,
            use_feature_indicator=use_feature_indicator,
        )
        if self.model_info['use_rgb_indicator']:
            assert self.model_info['use_rgb_as_input']
        if self.model_info['use_feature_indicator']:
            assert self.model_info['use_dist_as_input'] or self.model_info['use_zdir_as_input'] or \
                   self.model_info['use_dps_as_input'] or self.model_info['use_dpsuv_as_input'] or \
                   self.model_info['use_vdir_as_input']

        self.optim_info = dict(
            optim_method=optim_method,
            learning_rate=learning_rate,
            lr_factor=lr_factor,
            num_warmup_steps=num_warmup_steps,
            use_amp=use_amp,
            max_grad_val=max_grad_val,
            loss_weight_t=loss_weight_t,
            loss_weight_t_l1=loss_weight_t_l1,
            loss_weight_normal=loss_weight_normal,
            loss_weight_normal_l1=loss_weight_normal_l1,
            loss_weight_hit=loss_weight_hit,
            loss_weight_rgb=loss_weight_rgb,
            loss_weight_rgb_normal=loss_weight_rgb_normal,
            loss_weight_rgb_normal_dot=loss_weight_rgb_normal_dot,
            loss_weight_rgb_normal_dot_l1=loss_weight_rgb_normal_dot_l1,
            loss_weight_plane_normal=loss_weight_plane_normal,
            loss_weight_plane_normal_l1=loss_weight_plane_normal_l1,
            loss_rgb_type=loss_rgb_type,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_alpha=focal_loss_alpha,
            learn_ray_rgb=learn_ray_rgb,
            random_drop_rgb_rate=random_drop_rgb_rate,
            random_drop_sample_feature_rate=random_drop_sample_feature_rate,
            pcd_noise_std=pcd_noise_std,
        )

        self.model: T.Union[torch.nn.Module, None] = None  # this is the model
        self.losses_name = set()
        self.outputs_name = set()
        self.global_step = 0
        self.nan_count = 0

        self._register_var_to_save(
            [
                "dataset_info", "model_info", "optim_info", "losses_name", "outputs_name",
            ])  # dataset_info can change when resuming, so only save for future reference

        self._register_var_to_load([])  # these settings should not change when resuming, so reload

        # load options
        self.load_options(filename=self.process_info['config_filename'])

        self._register_output(
            [
                "pointersect_record",
            ])

        self._register_loss(
            [
                "loss_t",
                "loss_t_l1",
                "loss_normal",
                "loss_normal_l1",
                "loss_plane_normal",
                "loss_plane_normal_l1",
                "loss_total",
                "loss_hit",
                "loss_rgb",
                'loss_rgb_normal',
                'loss_rgb_normal_dot',
                'loss_rgb_normal_dot_l1',
                "rmse_theta",
                "rmse_theta_l1",
                "rmse_t",
                "rmse_plane_theta",
                "rmse_plane_theta_l1",
            ])

    def get_dataloaders(self):
        """This function is called after setup_assets."""

        self.logger.info(f'Creating datasets...')

        # camera settings
        input_camera_setting = dict(
            width_px=self.dataset_info['width_px'],
            height_px=self.dataset_info['height_px'],
            fov=self.dataset_info['fov'],
        )
        # we want input point cloud covers the entire mesh (with various sampling rate)
        input_camera_trajectory_params = dict(
            mode='random',
            min_r=self.dataset_info.get('mesh_scale', 1.),
            max_r=self.dataset_info.get('mesh_scale', 1.) * 3,
            max_angle=self.dataset_info['max_angle'],
            rand_r=self.dataset_info['rand_r'],
            local_max_angle=self.dataset_info['local_max_angle'],
            r_freq=1,
            max_translate_ratio=self.dataset_info['max_translate_ratio'],
        )
        output_camera_setting = dict(
            width_px=self.dataset_info['target_width_px'],
            height_px=self.dataset_info['target_height_px'],
            fov=self.dataset_info['fov'],
        )
        output_camera_trajectory_params = dict(
            mode='random',
            min_r=self.dataset_info.get('min_r', 0.5),
            max_r=self.dataset_info.get('max_r', 3),
            max_angle=self.dataset_info['max_angle'],
            rand_r=self.dataset_info['rand_r'],
            local_max_angle=self.dataset_info['local_max_angle'],
            r_freq=1,
            max_translate_ratio=self.dataset_info['max_translate_ratio'],
        )

        dataset_dict = dataset_helper.get_dataset(
            dataset_name=self.dataset_info['dataset_name'],
            dataset_info=self.dataset_info,
            input_camera_setting=input_camera_setting,
            input_camera_trajectory_params=input_camera_trajectory_params,
            output_camera_setting=output_camera_setting,
            output_camera_trajectory_params=output_camera_trajectory_params,
            rank=self.process_info['rank'],
            world_size=self.process_info['global_world_size'],
            printout=self.process_info['rank'] == 0,
        )
        dataset: mesh_dataset_v2.MeshConcatDataset = dataset_dict['dataset']
        val_dataset: mesh_dataset_v2.MeshConcatDataset = dataset_dict['val_dataset']
        test_dataset: mesh_dataset_v2.MeshConcatDataset = dataset_dict['test_dataset']

        # get dataloader
        self.logger.info('Creating dataloaders...')

        collate_fn = mesh_dataset_v2.MeshDatasetCollate()

        # add sampler to support distributed run
        if self.dataset_info.get('use_bucket_sampler', False):
            # use distributed bucket sampler to group batches with similar number of pixels
            seq_lens = dataset.get_all_num_pixels()
            self.train_sampler = distributed_bucket_sampler.DistributedBucketSampler(
                seq_lens=seq_lens,  # a list containing the sequence lengths of samples in dataset
                batch_size=self.dataset_info['batch_size'],
                num_replicas=self.process_info['global_world_size'] if self.process_info['distributed_run'] else 1,
                rank=self.process_info['global_rank'] if self.process_info['distributed_run'] else 0,
                drop_last=False,  # False is recommended
                bucket_boundaries=10,  # can be the number of bins or the bin edges
                seed=self.process_info['random_seed'],
                shuffle=True,
            )
            seq_lens = val_dataset.get_all_num_pixels()
            self.valid_sampler = distributed_bucket_sampler.DistributedBucketSampler(
                seq_lens=seq_lens,  # a list containing the sequence lengths of samples in dataset
                batch_size=self.dataset_info['batch_size'],
                num_replicas=self.process_info['global_world_size'] if self.process_info['distributed_run'] else 1,
                rank=self.process_info['global_rank'] if self.process_info['distributed_run'] else 0,
                drop_last=False,  # False is recommended
                bucket_boundaries=10,  # can be the number of bins or the bin edges
                seed=self.process_info['random_seed'],
                shuffle=True,
            )
            if test_dataset is not None:
                seq_lens = test_dataset.get_all_num_pixels()
                self.test_sampler = distributed_bucket_sampler.DistributedBucketSampler(
                    seq_lens=seq_lens,  # a list containing the sequence lengths of samples in dataset
                    batch_size=self.dataset_info['batch_size'],
                    num_replicas=self.process_info['global_world_size'] if self.process_info[
                        'distributed_run'] else 1,
                    rank=self.process_info['global_rank'] if self.process_info['distributed_run'] else 0,
                    drop_last=False,  # False is recommended
                    bucket_boundaries=10,  # can be the number of bins or the bin edges
                    seed=self.process_info['random_seed'],
                    shuffle=True,
                )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                num_workers=self.dataset_info['num_threads'],
                collate_fn=collate_fn,
                pin_memory=True,
                batch_sampler=self.train_sampler,
            )
            val_dataloader = torch.utils.data.DataLoader(
                dataset,
                num_workers=self.dataset_info['num_threads'],
                collate_fn=collate_fn,
                pin_memory=True,
                batch_sampler=self.valid_sampler,
            )
            if test_dataset is None:
                test_dataloader = None
            else:
                test_dataloader = torch.utils.data.DataLoader(
                    dataset,
                    num_workers=self.dataset_info['num_threads'],
                    collate_fn=collate_fn,
                    pin_memory=True,
                    batch_sampler=self.test_sampler,
                )

        else:
            # use typical pytorch sampler
            if self.process_info['distributed_run']:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset=dataset,
                    num_replicas=self.process_info['global_world_size'],
                    rank=self.process_info['rank'],
                    seed=self.process_info['random_seed'],
                    shuffle=True,
                    drop_last=False,
                )
                self.valid_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset=val_dataset,
                    num_replicas=self.process_info['global_world_size'],
                    rank=self.process_info['rank'],
                    seed=self.process_info['random_seed'],
                    shuffle=True,
                    drop_last=False,
                )
                if test_dataset is not None:
                    self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset=test_dataset,
                        num_replicas=self.process_info['global_world_size'],
                        rank=self.process_info['rank'],
                        seed=self.process_info['random_seed'],
                        shuffle=True,
                        drop_last=False,
                    )
            else:
                self.train_sampler = None
                self.valid_sampler = None
                if test_dataset is not None:
                    self.test_sampler = None

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.dataset_info['batch_size'],
                shuffle=(self.train_sampler is None),
                num_workers=self.dataset_info['num_threads'],
                collate_fn=collate_fn,
                pin_memory=True,
                sampler=self.train_sampler,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.dataset_info['batch_size'],
                shuffle=(self.valid_sampler is None),
                num_workers=self.dataset_info['num_threads'],
                collate_fn=collate_fn,
                pin_memory=True,
                sampler=self.valid_sampler,
            )
            if test_dataset is None:
                test_dataloader = None
            else:
                test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self.dataset_info['batch_size'],
                    shuffle=(self.test_sampler is None),
                    num_workers=self.dataset_info['num_threads'],
                    collate_fn=collate_fn,
                    pin_memory=True,
                    sampler=self.test_sampler,
                )

        return dataloader, val_dataloader, test_dataloader

    def construct_models(self):
        # get model
        self.model_info['dim_point_feature'] = 0
        # determine input dimension based on which features are used
        if self.model_info['use_rgb_as_input']:
            self.model_info['dim_point_feature'] += 3
        if self.model_info['use_dist_as_input']:
            self.model_info['dim_point_feature'] += 4
        if self.model_info['use_zdir_as_input']:
            self.model_info['dim_point_feature'] += 3
        if self.model_info['use_dps_as_input']:
            self.model_info['dim_point_feature'] += 1
        if self.model_info['use_dpsuv_as_input']:
            self.model_info['dim_point_feature'] += 6
        if self.model_info['use_vdir_as_input']:
            self.model_info['dim_point_feature'] += 3
        if self.model_info['use_rgb_indicator']:
            self.model_info['dim_point_feature'] += 1
        if self.model_info['use_feature_indicator']:
            self.model_info['dim_point_feature'] += 1

        self.model = pointersect.SimplePointersect(**self.model_info)

    def construct_optimizers(self):

        if self.optim_info['optim_method'] in {'adam', 'adam_tf'}:
            optim_method = torch.optim.Adam
        elif self.optim_info['optim_method'] == 'radam':
            optim_method = torch.optim.RAdam
        elif self.optim_info['optim_method'] == 'nadam':
            optim_method = torch.optim.NAdam
        else:
            raise NotImplementedError

        self._optimizer = optim_method(
            self.model.parameters(),
            lr=self.optim_info['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        if self.optim_info['optim_method'].endswith("_tf"):
            self.optimizer = TFOptimizer(
                optimizer=self._optimizer,
                model_size=self.model_info['dim_feature'],
                factor=self.optim_info['lr_factor'],
                warmup=self.optim_info['num_warmup_steps'],
                init_step=self.total_batch_count,
            )
        else:
            self.optimizer = self._optimizer

        if self.optim_info['use_amp']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # loss
        self.loss_l1 = torch.nn.L1Loss(reduction='none')
        self.loss_mse = torch.nn.MSELoss(reduction='none')
        # self.loss_cos_fn = torch.nn.CosineSimilarity(dim=-1, eps=1.e-8)
        self.loss_fn_hit = FocalLoss(
            gamma=self.optim_info['focal_loss_gamma'],
            alpha=self.optim_info['focal_loss_alpha'],
            reduction='none',
        )

        if self.optim_info['loss_rgb_type'] == 'vgg':
            self.loss_vgg = None
            raise NotImplementedError

    def _step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
            update: bool,
    ):
        """
        Args:
            epoch:
            bidx:
            batch:
                input_rgbd_images:
                    RGBDImage, (b, q, h, w),  images along q should be used to create point cloud
                ray:
                    Ray, (b, q=n_target_img, ho, wo)  target rays
                ray_gt_dict:
                    ray_rgbs: (b, q=n_target_img, ho, wo, 3)
                    ray_ts: (b, q=n_target_img, ho, wo)
                    surface_normals_w: (b, q=n_target_img, ho, wo, 3)
                    hit_map: (b, q=n_target_img, ho, wo)  1 if hit a surface, 0 otherwise
            update:

        Returns:

        """

        loss = 0
        total_stime = timer()
        with torch.autocast(
                device_type='cuda' if self.process_info['n_gpus'] > 0 else 'cpu',
                enabled=self.optim_info['use_amp'],
        ):
            input_rgbd_images: structures.RGBDImage = batch['input_rgbd_images'].to(device=self.device)  # (b, q, h, w)
            ray: structures.Ray = batch['ray'].to(device=self.device)  # (b, qo, ho, wo)
            ray_gt_dict = batch['ray_gt_dict']
            gt_ray_rgb: torch.Tensor = ray_gt_dict['ray_rgbs'].to(device=self.device)  # (b, qo, ho, wo, 3)
            gt_ray_ts: torch.Tensor = ray_gt_dict['ray_ts'].to(device=self.device)  # (b, qo, ho, wo)
            gt_surface_normals_w: torch.Tensor = ray_gt_dict['surface_normals_w'].to(
                device=self.device)  # (b, qo, ho, wo, 3)
            gt_hit_map: torch.Tensor = ray_gt_dict['hit_map'].to(device=self.device)  # (b, qo, ho, wo)

            input_point_cloud: structures.PointCloud = input_rgbd_images.get_pcd(
                subsample=1,
                remove_background=(input_rgbd_images.rgb.size(0) == 1),
            )  # (b, n, 3)

            # add noise to input point cloud
            if self.optim_info['pcd_noise_std'] > 1.e-6:
                noise = torch.randn_like(input_point_cloud.xyz_w) * self.optim_info['pcd_noise_std']
                input_point_cloud.xyz_w = input_point_cloud.xyz_w + noise

            # determine k to use
            r_k_ratio = torch.rand(1).item() * \
                        (self.dataset_info.get('max_k_ratio', 1.) - self.dataset_info.get('min_k_ratio', 1.)) + \
                        self.dataset_info.get('min_k_ratio', 1.)
            k = max(10, int(self.dataset_info.get('k', 40) * r_k_ratio))
            # self.logger.info(f'k = {k}, subsample = {r_subsample}')
            self.logger.info(f'k = {k}, input_rgbd_images.shape = {input_rgbd_images.rgb.shape}')

            # run infer
            compute_stime = timer()
            out_dict = infer.intersect_pcd_and_ray(
                point_cloud=input_point_cloud,
                camera_rays=ray,
                model=self.model,
                k=k,
                t_min=0. - 1e-4,
                t_max=1.e10,
                max_pr_chunk_size=-1,
                max_model_chunk_size=-1,
                pr_grid_size=100,
                pr_grid_width=2.1 * self.dataset_info.get('mesh_scale', 1.),
                pr_grid_center=0.,
                pr_ray_radius=self.dataset_info.get('ray_radius', 0.1) * self.dataset_info.get('mesh_scale', 1.),
                th_hit_prob=0.5,
                random_drop_rgb_rate=self.optim_info['random_drop_rgb_rate'],
                random_drop_sample_feature_rate=self.optim_info['random_drop_sample_feature_rate'],
                rgb_drop_val=0.5,
            )

            self.pointersect_record: structures.PointersectRecord = out_dict['pointersect_record']
            rgb_random_drop_mask = out_dict['rgb_random_drop_mask']  # (b, qo, ho, wo), bool

            compute_time = timer() - compute_stime

            # compute loss
            loss_stime = timer()

            # hit
            self.loss_hit = self.loss_fn_hit(
                self.pointersect_record.ray_hit_logit,  # (b, qo, ho, wo)
                gt_hit_map.float(),  # (b, qo, ho, wo)
            ).mean()

            # for other losses, we do not care what happens when gt_hit_map is False
            valid_mask = gt_hit_map  # bool,  (b, qo, ho, wo)
            invalid_mask = torch.logical_not(valid_mask)  # (b, qo, ho, wo)
            total_valid = valid_mask.float().sum()

            self.logger.info(f'gt hit ratio = {total_valid / valid_mask.numel() * 100.:.2f} %')

            # sometimes the camera observes nothing, set valid num to 1 to avoid being divided by zero
            if total_valid == 0:
                self.logger.info(f'this batch contains no valid sample!')
                total_valid = torch.ones_like(total_valid)

            if self.model_info['learn_dist']:
                assert False, 'we do not support distribution learning anymore'
            else:
                self.loss_t = self.loss_mse(
                    self.pointersect_record.ray_t,  # (b, qo, ho, wo)
                    gt_ray_ts.clone().masked_fill_(invalid_mask, 0.),  # (b, qo, ho, wo)
                ).masked_fill(invalid_mask, 0.).sum() / total_valid

                self.loss_t_l1 = self.loss_l1(
                    self.pointersect_record.ray_t,  # (b, qo, ho, wo)
                    gt_ray_ts.clone().masked_fill_(invalid_mask, 0.),  # (b, qo, ho, wo)
                ).masked_fill(invalid_mask, 0.).sum() / total_valid

            # surface normal loss is the sin(theta)^2
            if self.model_info['learn_dist']:
                assert False
            else:
                # l2
                loss_normal = torch.cross(
                    self.pointersect_record.intersection_surface_normal_w,  # (b, qo, ho, wo, 3)
                    gt_surface_normals_w.clone().masked_fill_(invalid_mask.unsqueeze(-1), 0.),  # (b, qo, ho, wo, 3)
                    dim=-1,
                ).pow(2).sum(dim=-1)  # (b, m)  sin(theta)^2
                self.loss_normal = loss_normal.masked_fill(invalid_mask, 0.).sum() / total_valid
                self.rmse_theta = torch.asin(self.loss_normal.pow(0.5)) * 180. / torch.pi  # (degree)

                # l1
                loss_normal_l1 = torch.cross(
                    self.pointersect_record.intersection_surface_normal_w,  # (b, qo, ho, wo, 3)
                    gt_surface_normals_w.clone().masked_fill_(invalid_mask.unsqueeze(-1), 0.),  # (b, qo, ho, wo, 3)
                    dim=-1,
                ).abs().sum(dim=-1)  # (b, m)  sin(theta)^2
                self.loss_normal_l1 = loss_normal_l1.masked_fill(invalid_mask, 0.).sum() / total_valid
                self.rmse_theta_l1 = torch.asin(self.loss_normal_l1) * 180. / torch.pi  # (degree)

            if self.model_info['estimate_surface_normal_weights'] \
                    and self.optim_info['loss_weight_plane_normal'] > 1.e-6:
                tmp_invalid_mask = torch.logical_or(
                    invalid_mask,
                    torch.logical_not(self.pointersect_record.valid_plane_normal_mask),
                )
                tmp_total_valid = torch.logical_not(tmp_invalid_mask).float().sum().clamp(min=1.)

                # currently only use sine loss
                # l2
                loss_normal = torch.cross(
                    self.pointersect_record.intersection_plane_normals_w,  # (b, qo, ho, wo, 3)
                    gt_surface_normals_w.clone().masked_fill_(tmp_invalid_mask.unsqueeze(-1), 0.),  # (b, qo, ho, wo, 3)
                    dim=-1,
                ).pow(2).sum(dim=-1)  # (b, qo, ho, wo)  sin(theta)^2

                self.loss_plane_normal = loss_normal.masked_fill(tmp_invalid_mask, 0.).sum() / tmp_total_valid
                self.rmse_plane_theta = torch.asin(self.loss_plane_normal.pow(0.5)) * 180. / torch.pi  # (degree)

                # l1
                loss_normal_l1 = torch.cross(
                    self.pointersect_record.intersection_plane_normals_w,  # (b, qo, ho, wo, 3)
                    gt_surface_normals_w.clone().masked_fill_(tmp_invalid_mask.unsqueeze(-1), 0.),  # (b, qo, ho, wo, 3)
                    dim=-1,
                ).abs().sum(dim=-1)  # (b, qo, ho, wo)  sin(theta)^2

                self.loss_plane_normal_l1 = loss_normal_l1.masked_fill(tmp_invalid_mask, 0.).sum() / tmp_total_valid
                self.rmse_plane_theta_l1 = torch.asin(self.loss_plane_normal_l1) * 180. / torch.pi  # (degree)

            # rgb
            if self.optim_info['learn_ray_rgb']:
                assert self.pointersect_record.intersection_rgb is not None

                if rgb_random_drop_mask is not None:
                    tmp_invalid_mask = torch.logical_or(invalid_mask, rgb_random_drop_mask)  # (b, qo, ho, wo)
                    tmp_total_valid = torch.logical_not(tmp_invalid_mask).float().sum().clamp(min=1.)
                else:
                    tmp_invalid_mask = invalid_mask  # (b, qo, ho, wo)
                    tmp_total_valid = total_valid

                if self.optim_info['loss_rgb_type'] == 'l2':
                    self.loss_rgb = self.loss_mse(
                        gt_ray_rgb,  # (b, qo, ho, wo, 3)
                        self.pointersect_record.intersection_rgb,  # (b, qo, ho, wo, 3)
                    )  # (b, m, 3)
                elif self.optim_info['loss_rgb_type'] == 'l1':
                    self.loss_rgb = self.loss_l1(
                        gt_ray_rgb,  # (b, qo, ho, wo, 3)
                        self.pointersect_record.intersection_rgb,  # (b, qo, ho, wo, 3)
                    )  # (b, m, 3)
                elif self.optim_info['loss_rgb_type'] == 'vgg':
                    self.loss_rgb = self.loss_vgg(
                        gt_ray_rgb,  # (b, qo, ho, wo, 3)
                        self.pointersect_record.intersection_rgb,  # (b, qo, ho, wo, 3)
                    )  # (,)
                else:
                    raise NotImplementedError

                if self.optim_info['loss_rgb_type'] in {'l2', 'l1'}:
                    self.loss_rgb = self.loss_rgb.masked_fill(
                        tmp_invalid_mask.unsqueeze(-1),  # (b, m, 1)
                        0.
                    ).sum() / tmp_total_valid

                if self.optim_info['loss_weight_rgb_normal'] > 1e-6:
                    tmp2_invalid_mask = torch.logical_or(
                        tmp_invalid_mask,
                        self.pointersect_record.valid_neighbor_idx_mask.sum(-1) < 2,  # (b, qo, ho, wo)
                    ).detach()  # (b, qo, ho, wo)
                    tmp2_total_valid = torch.logical_not(tmp2_invalid_mask).float().sum().clamp(min=1.)

                    # randomly select two neighbor points based on blending weight
                    probs = self.pointersect_record.blending_weights  # (b, qo, ho, wo, k)
                    b, *m_shape, _k = probs.shape
                    ridx = torch.multinomial(
                        input=probs.reshape(-1, probs.size(-1)).t(),  # (bqhw, k)
                        num_samples=2,
                        replacement=False,
                    ).reshape(b, *m_shape, 2).detach()  # (b, qo, ho, wo, 2)  long

                    n = input_point_cloud.xyz_w.size(-2)

                    # gather index
                    nidxs = torch.gather(
                        input=self.pointersect_record.neighbor_point_idxs,  # (b, qo, ho, wo, k)
                        dim=-1,
                        index=ridx
                    ).detach()  # (b, qo, ho, wo, 2)

                    # gather point
                    npoints_w = torch.gather(
                        input=input_point_cloud.xyz_w.reshape(
                            b, *([1] * len(m_shape)), n, 3).expand(
                            b, *m_shape, n, 3).to(device=self.device),
                        # (b, qo, ho, wo, n, 3)
                        dim=-2,
                        index=nidxs.unsqueeze(-1).expand(-1, *([-1] * len(m_shape)), -1, 3),  # (b, qo, ho, wo, 2, 3)
                    )  # (b, qo, ho, wo, 2, 3)

                    # form two vectors
                    ds = npoints_w - self.pointersect_record.intersection_xyz_w.unsqueeze(-2)  # (b, qo, ho, wo, 2, 3)
                    ds = torch.nn.functional.normalize(ds, p=2., dim=-1)  # (b, qo, ho, wo, k, 2, 3)

                    # use the two points and the intersection point to compute normal
                    ns = torch.cross(
                        ds[..., 0, :],  # (b, qo, ho, wo, 3)
                        ds[..., 1, :],  # (b, qo, ho, wo, 3)
                        dim=-1
                    )  # (b, qo, ho, wo, 3)
                    ns = torch.nn.functional.normalize(ns, p=2., dim=-1)

                    # loss = dot(ns, gt_ns).^2
                    self.loss_rgb_normal = torch.sum(
                        ns * gt_surface_normals_w.clone().masked_fill_(tmp2_invalid_mask.unsqueeze(-1), 0.),
                        dim=-1,
                    ) ** 2  # (b, qo, ho, wo)

                    self.loss_rgb_normal = self.loss_rgb_normal.masked_fill(
                        tmp2_invalid_mask, 0.).sum() / tmp2_total_valid

                    self.logger.info(f'self.loss_rgb_normal = {self.loss_rgb_normal}')

                if self.optim_info['loss_weight_rgb_normal_dot'] > 1e-6:
                    b, *m_shape, _k = self.pointersect_record.neighbor_point_idxs.shape
                    n = input_point_cloud.xyz_w.size(-2)

                    # gather point
                    npoints_w = torch.gather(
                        input=input_point_cloud.xyz_w.reshape(
                            b, *([1] * len(m_shape)), n, 3).expand(
                            b, *m_shape, n, 3,
                        ).to(device=self.device),
                        # (b, qo, ho, wo, n, 3)
                        dim=-2,
                        index=self.pointersect_record.neighbor_point_idxs.unsqueeze(-1).expand(
                            -1, *([-1] * len(m_shape)), -1, 3).detach(),  # (b, qo, ho, wo, k, 3)
                    )  # (b, qo, ho, wo, k, 3)

                    # form vectors on plane
                    ds = npoints_w - self.pointersect_record.intersection_xyz_w.unsqueeze(-2)  # (b, qo, ho, wo, k, 3)
                    ds = torch.nn.functional.normalize(ds, p=2., dim=-1)  # (b, qo, ho, wo, k, 3)

                    ds = ds.masked_fill(
                        torch.logical_not(self.pointersect_record.valid_neighbor_idx_mask.unsqueeze(-1)),
                        0.,
                    )

                    # dot product between ds and surface normal
                    ds_dot_ns = torch.sum(
                        ds * gt_surface_normals_w.clone().masked_fill_(tmp_invalid_mask.unsqueeze(-1), 0.).unsqueeze(
                            -2),
                        dim=-1,
                    )  # (b, qo, ho, wo, k)
                    # l2
                    self.loss_rgb_normal_dot = ds_dot_ns ** 2  # (b, qo, ho, wo, k)
                    self.loss_rgb_normal_dot = self.loss_rgb_normal_dot * self.pointersect_record.blending_weights
                    self.loss_rgb_normal_dot = self.loss_rgb_normal_dot.masked_fill(
                        tmp_invalid_mask.unsqueeze(-1), 0.).sum() / tmp_total_valid

                    self.loss_rgb_normal_dot_l1 = ds_dot_ns.abs()  # (b, qo, ho, wo, k)
                    self.loss_rgb_normal_dot_l1 = self.loss_rgb_normal_dot_l1 * self.pointersect_record.blending_weights
                    self.loss_rgb_normal_dot_l1 = self.loss_rgb_normal_dot_l1.masked_fill(
                        tmp_invalid_mask.unsqueeze(-1), 0.).sum() / tmp_total_valid

            self.loss_total = \
                self.optim_info['loss_weight_t'] * self.loss_t + \
                self.optim_info['loss_weight_t_l1'] * self.loss_t_l1 + \
                self.optim_info['loss_weight_normal'] * self.loss_normal + \
                self.optim_info['loss_weight_normal_l1'] * self.loss_normal_l1 + \
                self.optim_info['loss_weight_hit'] * self.loss_hit

            if self.model_info['estimate_surface_normal_weights'] \
                    and self.optim_info['loss_weight_plane_normal'] > 1.e-6:
                self.loss_total = \
                    self.loss_total + \
                    self.optim_info['loss_weight_plane_normal'] * self.loss_plane_normal + \
                    self.optim_info['loss_weight_plane_normal_l1'] * self.loss_plane_normal_l1

            if self.optim_info['learn_ray_rgb']:
                self.loss_total = self.loss_total + \
                                  self.optim_info['loss_weight_rgb'] * self.loss_rgb

                if self.optim_info['loss_weight_rgb_normal'] > 1e-6:
                    self.loss_total = self.loss_total + \
                                      self.optim_info['loss_weight_rgb_normal'] * self.loss_rgb_normal

                if self.optim_info['loss_weight_rgb_normal_dot'] > 1e-6:
                    self.loss_total = \
                        self.loss_total + \
                        self.optim_info['loss_weight_rgb_normal_dot'] * self.loss_rgb_normal_dot + \
                        self.optim_info['loss_weight_rgb_normal_dot_l1'] * self.loss_rgb_normal_dot_l1

            loss = self.loss_total
            loss_etime = timer()
            loss_time = loss_etime - loss_stime

            if not torch.isfinite(loss) or torch.isnan(loss):
                self.logger.info('loss is wrong!')

            if self.loss_t > 1e4:
                self.logger.info('loss t is too large?')

        update_stime = timer()
        if update:

            # clear the temporary gradient storage of all parameters
            self.optimizer.zero_grad()

            if self.optim_info['use_amp']:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            backward_time = timer() - update_stime

            # gradient clipping (self.parameters contains the parameters of the generator)
            if self.optim_info['use_amp']:
                self.optimizer.unscale(scaler=self.scaler)

            try:
                # try to catch nan if gracefully ignore the batch
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.optim_info['max_grad_val'],
                    error_if_nonfinite=True,
                )

                if self.optim_info['use_amp']:
                    # update parameters
                    self.optimizer.step(scaler=self.scaler)
                    # should be called after all optimizers had been stepped.
                    self.scaler.update()
                else:
                    # update parameters
                    self.optimizer.step()

            except:
                self.nan_count += 1
                self.logger.info(
                    f'gradient has nan, skip the update of the batch. '
                    f'nan_count = {self.nan_count} / global_step = {self.global_step} = '
                    f'{self.nan_count / (self.global_step + 1) * 100.:.3g}%'
                )
                pass

        update_time = timer() - update_stime

        total_time = timer() - total_stime

        # gather loss
        loss_dict = self._gather_losses_by_losses_name()

        # add learning rate into loss_dict
        loss_dict.update(self.get_current_lrs())

        if self.process_info['rank'] == 0:
            self.logger.info(
                f'total = {total_time:.2f} secs: '
                f'forward = {compute_time / total_time * 100:.1f}% '
                f'loss = {loss_time / total_time * 100:.1f}% '
                f'backward = {update_time / total_time * 100:.1f}% ')

        # show info to confirm that every rank receive different batches
        self.logger.info(f'local rank total loss = {loss}')

        return loss_dict

    def _visualize_gt_est(
            self,
            bidx: int,
            batch: T.Any,
            logger: Logger,
            global_step: int,
            figsize=None,
            dpi=150,
            main_tag='',
    ):
        """Visualize the results computed in _step. """

        ray_gt_dict = batch['ray_gt_dict']
        gt_ray_rgb: torch.Tensor = ray_gt_dict['ray_rgbs'][bidx]  # (qo, ho, wo, 3)
        gt_ray_ts: torch.Tensor = ray_gt_dict['ray_ts'][bidx]  # (qo, ho, wo)
        gt_surface_normals_w: torch.Tensor = ray_gt_dict['surface_normals_w'][bidx]  # ( qo, ho, wo, 3)
        gt_hit_map: torch.Tensor = ray_gt_dict['hit_map'][bidx]  # (qo, ho, wo)

        # plot gt and est side by side
        valid_mask = gt_hit_map  # (qo, h, w)
        invalid_mask = torch.logical_not(valid_mask)  # (qo, h, w)

        ## input rgb images
        input_rgb_images = batch['input_rgbd_images'].rgb[bidx]  # (q, h, w, 3)
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figsize, dpi=dpi)
        fig, ax = utils.plot_multiple_images(
            imgs=input_rgb_images,  # (q, h, w, 3)
            dpi=dpi,
            mode='tile',
            fig=fig,
            ax=axs,
            colorbar=False,
            ncols=6,
        )
        tag = f'{main_tag}input_rgb_images_{bidx}'
        logger.add_figure(tag=tag, figure=fig, global_step=global_step, close=True)

        ## surface normal
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figsize, dpi=dpi)
        fig_idx = 0
        fig, ax = utils.plot_multiple_images(
            imgs=gt_surface_normals_w,  # (qo, ho, wo, 3)
            dpi=dpi,
            mode='horizontal',
            fig=fig,
            ax=axs[fig_idx],
            colorbar=False,
            valrange=[-1., 1.],
        )
        fig_idx += 1
        fig, ax = utils.plot_multiple_images(
            imgs=self.pointersect_record.intersection_surface_normal_w[bidx],  # (n_target_img, h, w, 3)
            dpi=dpi,
            mode='horizontal',
            fig=fig,
            ax=axs[fig_idx],
            colorbar=False,
            valrange=[-1., 1.],
        )
        fig_idx += 1
        fig, ax = utils.plot_multiple_images(
            imgs=torch.linalg.vector_norm(
                torch.cross(
                    self.pointersect_record.intersection_surface_normal_w[bidx],
                    gt_surface_normals_w.to(device=self.pointersect_record.intersection_surface_normal_w.device),
                    dim=-1,
                ), dim=-1,
            ).masked_fill(invalid_mask.to(device=self.pointersect_record.intersection_surface_normal_w.device), 0),
            dpi=dpi,
            mode='horizontal',
            fig=fig,
            ax=axs[fig_idx],
            colorbar=False,
        )
        tag = f'{main_tag}surface_normal_{bidx}'
        logger.add_figure(tag=tag, figure=fig, global_step=global_step, close=True)

        ## ts
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figsize, dpi=dpi)
        fig_idx = 0
        fig, ax = utils.plot_multiple_images(
            imgs=gt_ray_ts * gt_hit_map.float(),  # (qo, ho, wo)
            dpi=dpi,
            mode='horizontal',
            fig=fig,
            ax=axs[fig_idx],
            colorbar=True,
        )
        fig_idx += 1

        fig, ax = utils.plot_multiple_images(
            imgs=self.pointersect_record.ray_t[bidx] * (self.pointersect_record.ray_hit[bidx]).float(),  # (qo, ho, wo)
            dpi=dpi,
            mode='horizontal',
            fig=fig,
            ax=axs[fig_idx],
            colorbar=True,
        )
        tag = f'{main_tag}ts_{bidx}'
        logger.add_figure(tag=tag, figure=fig, global_step=global_step, close=True)

        ## hit
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figsize, dpi=dpi)
        fig_idx = 0
        fig, ax = utils.plot_multiple_images(
            imgs=gt_hit_map.float(),  # (qo, ho, wo)
            dpi=dpi,
            mode='horizontal',
            fig=fig,
            ax=axs[fig_idx],
            colorbar=True,
        )
        fig_idx += 1
        fig, ax = utils.plot_multiple_images(
            imgs=self.pointersect_record.ray_hit[bidx].float(),  # (qo, ho, wo)
            dpi=dpi,
            mode='horizontal',
            fig=fig,
            ax=axs[fig_idx],
            colorbar=True,
        )
        tag = f'{main_tag}hit_{bidx}'
        logger.add_figure(tag=tag, figure=fig, global_step=global_step, close=True)

        ## geometry weights
        if self.pointersect_record.geometry_weights is not None:
            fig, axs = plt.subplots(
                nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figsize, dpi=dpi)
            # fig_idx = 0

            fig, ax = utils.plot_multiple_images(
                imgs=self.pointersect_record.geometry_weights[bidx].reshape(
                    self.pointersect_record.geometry_weights.size(1),
                    -1,
                    self.pointersect_record.geometry_weights.size(-1),
                ).transpose(-1, -2),  # (qo, ho, wo, k) -> (qo, ho*wo, k) -> (qo, k, ho*wo)
                dpi=dpi,
                mode='horizontal',
                fig=fig,
                ax=axs,
                colorbar=True,
            )
            # fig_idx += 1
            tag = f'{main_tag}geometry_weights_{bidx}'
            logger.add_figure(tag=tag, figure=fig, global_step=global_step, close=True)

        ## blending weights
        if self.pointersect_record.blending_weights is not None:
            fig, axs = plt.subplots(
                nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figsize,
                dpi=dpi)
            # fig_idx = 0

            fig, ax = utils.plot_multiple_images(
                imgs=self.pointersect_record.blending_weights[bidx].reshape(
                    self.pointersect_record.blending_weights.size(1),
                    -1,
                    self.pointersect_record.blending_weights.size(-1),
                ).transpose(-1, -2),  # (qo, ho, wo, k) -> (qo, ho*wo, k) -> (qo, k, ho*wo)
                dpi=dpi,
                mode='horizontal',
                fig=fig,
                ax=axs,
                colorbar=True,
            )
            # fig_idx += 1
            tag = f'{main_tag}blending_weights_{bidx}'
            logger.add_figure(tag=tag, figure=fig, global_step=global_step, close=True)

        ## ray_rgb
        if self.pointersect_record.intersection_rgb is not None:
            fig, axs = plt.subplots(
                nrows=2, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figsize, dpi=dpi)
            fig_idx = 0
            fig, ax = utils.plot_multiple_images(
                imgs=gt_ray_rgb * gt_hit_map.unsqueeze(-1).float(),  # (qo, ho, wo, 3)
                dpi=dpi,
                mode='horizontal',
                fig=fig,
                ax=axs[fig_idx],
                colorbar=True,
            )
            fig_idx += 1
            fig, ax = utils.plot_multiple_images(
                imgs=self.pointersect_record.intersection_rgb[bidx] * (self.pointersect_record.ray_hit[bidx]).unsqueeze(
                    -1).float(),
                dpi=dpi,
                mode='horizontal',
                fig=fig,
                ax=axs[fig_idx],
                colorbar=True,
            )
            tag = f'{main_tag}ray_rgb_{bidx}'
            logger.add_figure(tag=tag, figure=fig, global_step=global_step, close=True)

    def get_current_lrs(self):
        """
        Get a dictionary containing the current learning rate of each optimizer
        """
        lr_dict = dict()
        lr_dict['lr'] = self._optimizer.param_groups[0]['lr']
        return lr_dict

    def _register_output(
            self,
            var_name: T.Union[str, T.List[str]],
    ):
        """
        Register output so that it will be reset by `reset_outputs`

        Args:
            var_name:
                var name (str) or list of var names
        """
        self._register(var_name=var_name, target=self.outputs_name)

    def _register_loss(
            self,
            var_name: T.Union[str, T.List[str]],
    ):
        """
        Register scalar loss so that it will be gathered by :py:`_gather_loss`

        Args:
            var_name:
                loss name (str) or list of loss names
        """
        self._register(var_name=var_name, target=self.losses_name)

    def _register(
            self,
            var_name: T.Union[str, T.List[str]],
            target: T.Set[str],
    ):
        """
        Register the var_name to the target (set of var names)

        Args:
            var_name:
                name (str) or list of names of the variable to register
            target:
                the set to register the var namee
        """
        if isinstance(var_name, str):
            var_name = [var_name]

        for name in var_name:
            target.add(name)

    def _gather_losses_by_losses_name(self, return_float=True) -> T.Dict[str, float]:
        """
        Return the values in losses_name in a dictionary.
        """
        loss_dict = dict()

        if not hasattr(self, 'losses_name'):
            return loss_dict

        for name in sorted(self.losses_name):
            loss = getattr(self, name, None)
            if loss is None:
                continue
            else:
                loss_dict[name] = loss

        loss_dict = self._gather_losses(
            loss_dict=loss_dict,
            return_float=return_float,
        )
        return loss_dict

    def _gather_losses(
            self,
            loss_dict: T.Dict[str, torch.Tensor],
            return_float: bool = True,
    ) -> T.Dict[str, float]:
        """
        Return the values in losses_name in a dictionary.
        """
        if loss_dict is None:
            return None

        out_dict = dict()
        keys = sorted(list(loss_dict.keys()))
        for name in keys:  # sorted to support dist.reduce
            val = loss_dict[name]
            if val is None:
                continue

            if not isinstance(val, torch.Tensor):
                out_dict[name] = val
            else:
                if self.process_info['distributed_run']:
                    reduced_loss = reduce_tensor(val.detach()) / self.process_info['n_gpus']
                    out_dict[name] = reduced_loss
                else:
                    out_dict[name] = val.detach()

        if return_float:
            for key, val in out_dict.items():
                if isinstance(val, torch.Tensor):
                    out_dict[key] = val.detach().cpu().item()
                elif isinstance(val, np.ndarray):
                    out_dict[key] = val.item()
                elif isinstance(val, int):
                    out_dict[key] = float(val)

        return out_dict

    def reset_outputs(self):
        """
        Set the outputs of the model to None.
        """
        for name in self.outputs_name:
            setattr(self, name, None)

    def reset_losses(self):
        """
        Set the losses to None.
        """
        for name in self.losses_name:
            setattr(self, name, None)

    def train_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ):

        self.reset_losses()
        self.reset_outputs()

        all_loss_dict = self._step(
            epoch=epoch,
            bidx=bidx,
            batch=batch,
            update=True,
        )
        return all_loss_dict

    def validation_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ):
        self.reset_losses()
        self.reset_outputs()

        with torch.no_grad():
            all_loss_dict = self._step(
                epoch=epoch,
                bidx=bidx,
                batch=batch,
                update=False,
            )
        return all_loss_dict

    def test_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ):
        self.reset_losses()
        self.reset_outputs()

        with torch.no_grad():
            all_loss_dict = self._step(
                epoch=epoch,
                bidx=bidx,
                batch=batch,
                update=False,
            )
        return all_loss_dict

    def visualize_train_step(
            self,
            epoch: int,
            total_batch_count: int,
            bidx: int,
            batch: T.Any,
            out_dict: T.Dict[str, T.Any],
            logger: Logger,
            batch_time: float,
            step_time: float,
            epoch_time: float,
    ):
        self.logger.info(
            f'batch uses: {batch_time:.4f} secs, '
            f'(step uses: {step_time:.4f} secs, '
            f'{step_time / batch_time * 100.:.2f}%)'
        )

        batch_size = batch['ray'].origins_w.size(0)
        total = min(10, batch_size)
        plot_stime = timer()
        for bidx in range(total):
            logger.info(f'plotting {bidx}..')
            self._visualize_gt_est(
                bidx=bidx,
                batch=batch,
                logger=logger,
                global_step=total_batch_count,
                figsize=None,
                dpi=300,
                main_tag='train_',
            )
        plot_time = timer() - plot_stime
        logger.info(f'plotting used {plot_time} secs')

    def visualize_validation_step(
            self,
            epoch: int,
            total_batch_count: int,
            bidx: int,
            batch: T.Any,
            out_dict: T.Dict[str, T.Any],
            logger: Logger,
            batch_time: float,
            step_time: float,
            epoch_time: float,
    ):
        self.logger.info(
            f'batch uses: {batch_time:.4f} secs, '
            f'(step uses: {step_time:.4f} secs, '
            f'{step_time / batch_time * 100.:.2f}%)'
        )

        batch_size = batch['ray'].origins_w.size(0)
        total = min(10, batch_size)
        plot_stime = timer()
        for bidx in range(total):
            logger.info(f'plotting {bidx}..')
            self._visualize_gt_est(
                bidx=bidx,
                batch=batch,
                logger=logger,
                global_step=total_batch_count,
                figsize=None,
                dpi=300,
                main_tag='valid_',
            )
        plot_time = timer() - plot_stime
        logger.info(f'plotting used {plot_time} secs')

    def visualize_test_step(
            self,
            epoch: int,
            total_batch_count: int,
            bidx: int,
            batch: T.Any,
            out_dict: T.Dict[str, T.Any],
            logger: Logger,
            batch_time: float,
            step_time: float,
            epoch_time: float,
    ):
        self.logger.info(
            f'batch uses: {batch_time:.4f} secs, '
            f'(step uses: {step_time:.4f} secs, '
            f'{step_time / batch_time * 100.:.2f}%)'
        )

        batch_size = batch['ray'].origins_w.size(0)
        total = min(10, batch_size)
        plot_stime = timer()
        for bidx in range(total):
            logger.info(f'plotting {bidx}..')
            self._visualize_gt_est(
                bidx=bidx,
                batch=batch,
                logger=logger,
                global_step=total_batch_count,
                figsize=None,
                dpi=300,
                main_tag='test_',
            )
        plot_time = timer() - plot_stime
        logger.info(f'plotting used {plot_time} secs')

    def epoch_setup(
            self,
            epoch: int,
            dataloader: T.Sequence[T.Any],
            val_dataloader: T.Sequence[T.Any],
            test_dataloader: T.Sequence[T.Any],
    ):
        """Set up at the beginning of an epoch, before dataloder iterator
         is constructed.  It can be used to setup the batch sampler, etc."""

        # used when distribution learning
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        if self.valid_sampler is not None:
            self.valid_sampler.set_epoch(epoch)
        if self.test_sampler is not None:
            self.test_sampler.set_epoch(epoch)


if __name__ == "__main__":
    matplotlib.use('agg')
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument(
        '--trainer_filename',
        type=str,
        default=None,
        help='previous pth file')
    parser.add_argument(
        '--config_filename',
        type=str,
        default=None,
        help='config yaml/json file')
    parser.add_argument(
        '--num_threads',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--pcd_subsample',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--k',
        type=int,
        default=40,
    )
    parser.add_argument(
        '--visualize_every_num_train_batch',
        type=int,
        default=300,
    )
    parser.add_argument(
        '--visualize_every_num_valid_batch',
        type=int,
        default=30,
    )
    parser.add_argument(
        '--visualize_every_num_test_batch',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--skip_existed_cleaned_mesh',
        action='store_true',
    )

    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--n_gpus', type=int, default=1)
    options = parser.parse_args()

    # torch.autograd.set_detect_anomaly(True)

    with TrainPointersectProcess(
            trainer_filename=options.trainer_filename,
            config_filename=options.config_filename,
            rank=options.rank,
            n_gpus=options.n_gpus,
            save_code=False,  # will be overwritten by config_filename if given
            pcd_subsample=options.pcd_subsample,  # will be overwritten by config_filename if given
            num_threads=options.num_threads,  # will be overwritten by config_filename if given
            k=options.k,  # will be overwritten by config_filename if given
            visualize_every_num_train_batch=options.visualize_every_num_train_batch,
            visualize_every_num_valid_batch=options.visualize_every_num_valid_batch,
            visualize_every_num_test_batch=options.visualize_every_num_test_batch,
            skip_existed_cleaned_mesh=options.skip_existed_cleaned_mesh,  # just for speed, default false
    ) as trainer:
        trainer.run()
