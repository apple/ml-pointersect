#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# The file implements actual functions.

import json
import math
import traceback
import typing as T
from timeit import default_timer as timer

import numpy as np
import open3d as o3d
import torch
from chamferdist import ChamferDistance
from torch import profiler
from tqdm import tqdm

import cdslib.core.utils.print_and_save
from plib import render
from plib import utils
from plib.metrics import get_lpips_model
from pointersect.inference import inference_utils
from pointersect.inference.structures import Camera
from pointersect.inference.structures import CameraTrajectory
from pointersect.inference.structures import Mesh
from pointersect.inference.structures import PointCloud
from pointersect.inference.structures import PointersectRecord
from pointersect.inference.structures import RGBDImage
from pointersect.inference.structures import Ray
from pointersect.models import model_utils
from pointersect.models.pointersect import SimplePointersect


def intersect_pcd_and_ray(
        point_cloud: PointCloud,
        camera_rays: Ray,
        model: SimplePointersect,
        k: int,  # number of neighbor points to use
        t_min: float = 1.0e-8,
        t_max: float = 1.0e10,
        max_pr_chunk_size: int = -1,  # max n_points * n_rays to process in a chunk
        max_model_chunk_size: int = -1,  # max n_points * n_rays to process in a chunk
        pr_grid_size: int = 100,
        pr_grid_width: float = 2.2,
        pr_grid_center: float = 0.,
        pr_ray_radius: float = 0.1,
        th_hit_prob: float = 0.5,  # > th, hit
        printout: bool = False,
        cached_info: T.Union[T.Dict[str, torch.Tensor], None] = None,
        random_drop_rgb_rate: float = 0.,
        random_drop_sample_feature_rate: float = 0.,
        rgb_drop_val: float = 0.5,
        rgb_render_method: str = 'blending_weights',
        enable_timing: bool = False,
) -> T.Dict[str, T.Union[PointersectRecord, T.Dict[str, torch.Tensor], torch.Tensor, None]]:
    """
    Given point cloud (scene representation) and camera rays,
    compute intersection points, surface normal, rgb, etc.

    Procedure:
    - find neighboring points using pr
    - run pointersect model


    cached_info:
        a dictionary containing the grid_cell_to_point information.
    """

    if hasattr(model, 'dim_point_feature'):
        pass
    else:
        # model is DDP, we use the actual model
        model = model.module

    b = point_cloud.xyz_w.size(0)
    _b, *m_shape, _ = camera_rays.origins_w.shape
    assert _b == b
    stime_total = timer()

    pr_params = dict(
        ray_radius=pr_ray_radius,
        grid_size=pr_grid_size,
        grid_center=pr_grid_center,
        grid_width=pr_grid_width,
    )

    ori_device = point_cloud.xyz_w.device
    device_cpu = torch.device('cpu')
    model_device = next(model.parameters()).device
    pr_device = model_device
    timing_info = dict()

    # prepare for pr by inserting a point at inf
    point_cloud.insert_point_at_inf()

    # find neighbor point of each camera ray
    # with torch.no_grad():  # we never need to backprop through finding neighbors
    other_maps = []
    rotate_other_maps = []
    other_maps_name_idx = dict()
    idx = 0

    name_rotates = [
        ('rgb', False),
        ('captured_z_direction_w', True),
        ('captured_view_direction_w', True),
        ('captured_dps', False),
        ('captured_dps_u_w', True),
        ('captured_dps_v_w', True),
    ]

    for name, rotate in name_rotates:
        arr = getattr(point_cloud, name, None)
        if arr is not None:
            other_maps.append(arr.to(device=pr_device))
            rotate_other_maps.append(rotate)
            other_maps_name_idx[name] = idx
            idx += 1

    with profiler.record_function('find_neighbors_and_rectify'):
        out_dict = model_utils.find_neighbors_and_rectify(
            points_w=point_cloud.xyz_w.to(device=pr_device),
            ray_origins_w=camera_rays.origins_w.reshape(b, -1, 3).to(device=pr_device),  # (b, m, 3)
            ray_directions_w=camera_rays.directions_w.reshape(b, -1, 3).to(device=pr_device),  # (b, m, 3)
            k=k,
            t_min=t_min,
            t_max=t_max,
            t_init=None,  # (b, m', 1)
            other_maps=other_maps,
            rotate_other_maps=rotate_other_maps,
            translate_other_maps=None,
            max_chunk_size=max_pr_chunk_size,
            pr_params=pr_params,
            randomize_translate=False,
            printout=printout,
            cached_info=cached_info,
            valid_mask=point_cloud.valid_mask,
            enable_timing=enable_timing,
        )
    if out_dict.get('timing_info', None) is not None:
        timing_info.update(out_dict['timing_info'])
    # print(f'pr uses {total_time_neighbor:.2f} secs')
    cached_info = out_dict.get('cached_info', None)
    # points_n  # (b, m, k, 3) rectified points in the transformed coord
    # ts_shift  # (b, m)  # (ts_n = ts - ts_shift)
    # Rs_w2n   # (b, m, 3, 3)  from world coord to the transformed coord
    # translation_w2n # (b, m, 3, 1)  from the world coord to the transformed coord
    # neighbor_other_maps  # list of (b, m, k, d) associated with each point
    # neighbor_num  # (b, m) number of neighbor points for each ray

    # handle memory
    bm = np.prod(list(out_dict['points_n'].shape[:2]))
    if max_model_chunk_size > 0 and bm > max_model_chunk_size:
        # we are going to do things in chunks, move data to cpu
        out_dict = utils.to_device(out_dict, device_cpu)

    points_n = out_dict['points_n']  # (b, m, k, 3)  in the transformed coord
    ts_shift = out_dict['ts_shift']  # (b, m)  # (ts_new = ts - ts_shift)
    Rs_w2n = out_dict['Rs_w2n']  # (b, m, 3, 3)  from world coord to the transformed coord
    # translation_w2n = out_dict['translation_w2n']  # (b, m, 3, 1)  from the world coord to the transformed coord
    Rs_n2w = Rs_w2n.transpose(-1, -2)  # (b, m, 3, 3)  from local coord to the world coord
    neighbor_num = out_dict.get('neighbor_num', None)  # (b, m) number of valid neighbor points of each ray
    valid_mask = out_dict.get('valid_mask', None)  # (b, m, k) bool, considered neighbor_num

    # random dropping rgb and features
    with torch.no_grad():
        rgb_random_drop_mask = None
        feature_random_drop_mask = None
        if 'rgb' in other_maps_name_idx:
            if random_drop_rgb_rate > 0:
                mask_size = points_n.size()  # (b, m, k, 3)
                mask_size = mask_size[:-2]  # (b, m,)
                rgb_random_drop_mask = torch.rand(mask_size, device=points_n.device) < random_drop_rgb_rate  # (b, m,)
                out_dict['neighbor_other_maps'][other_maps_name_idx['rgb']][rgb_random_drop_mask] = rgb_drop_val

        # all other features share the same mask
        if random_drop_sample_feature_rate > 0:
            mask_size = points_n.size()  # (b, m, k, 3)
            mask_size = mask_size[:-2]  # (b, m,)
            feature_random_drop_mask = torch.rand(mask_size, device=points_n.device) < random_drop_sample_feature_rate

            for name in other_maps_name_idx:
                if name == 'rgb':
                    continue
                out_dict['neighbor_other_maps'][other_maps_name_idx[name]][feature_random_drop_mask] = 0

    # compile point feature for pointersect model
    feature_name_usage = [
        ('rgb', getattr(model, 'use_rgb_as_input', False)),
        ('captured_z_direction_w', getattr(model, 'use_zdir_as_input', False)),
        ('captured_dps', getattr(model, 'use_dps_as_input', False)),
        ('captured_dps_u_w', getattr(model, 'use_dpsuv_as_input', False)),
        ('captured_dps_v_w', getattr(model, 'use_dpsuv_as_input', False)),
        ('captured_view_direction_w', getattr(model, 'use_vdir_as_input', False)),
    ]  # order is important

    points_features = []
    for name, use_feature in feature_name_usage:
        if use_feature:
            idx = other_maps_name_idx.get(name, None)
            if idx is not None:
                arr = out_dict['neighbor_other_maps'][idx]  # (b, m, k, dim)
                if arr.ndim == 3:
                    arr = arr.unsqueeze(0)
            else:
                if name == 'rgb':
                    arr = torch.ones_like(points_n) * 0.5  # (b, m, k, 3)
                elif name in {
                    'normal_w',
                    'captured_z_direction_w',
                    'captured_dps_u_w',
                    'captured_dps_v_w',
                    'captured_view_direction_w',
                }:
                    arr = torch.zeros_like(points_n)
                elif name in {'captured_dps'}:
                    arr = torch.zeros(*(points_n.shape[:-1]), 1, dtype=points_n.dtype, device=points_n.device)
                else:
                    raise NotImplementedError
            points_features.append(arr)

    # specifically preserve rgb and neighbor point idx before deleting out_dict
    idx = other_maps_name_idx['rgb']
    points_rgb = out_dict['neighbor_other_maps'][idx]  # (b, m, k, 3)
    neighbor_point_idxs = out_dict['neighbor_info']['sorted_idxs']  # (b, m, k)  long
    del out_dict

    if getattr(model, 'use_dist_as_input', False):
        points_square = points_n * points_n  # (b, m, k, 3)
        points_dist_square = torch.sum(points_square[..., :2], -1, keepdim=True)
        points_dist_features = torch.cat((points_square, points_dist_square), dim=3)
        points_dist_features = torch.sqrt(points_dist_features)
        points_features.append(points_dist_features)
        del points_square
        del points_dist_square

    if getattr(model, 'use_rgb_indicator', False):
        if rgb_random_drop_mask is not None:
            indicator = torch.logical_not(rgb_random_drop_mask).to(dtype=torch.float, device=points_n.device)  # (b, m,)
            indicator = indicator.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k, 1)  # (b, m, k, 1)
        elif 'rgb' in other_maps_name_idx:
            indicator = torch.ones(*points_n.shape[:-1], 1).to(
                dtype=torch.float, device=points_n.device)  # (b, m, k, 1)
        else:
            indicator = torch.zeros(*points_n.shape[:-1], 1).to(
                dtype=torch.float, device=points_n.device)  # (b, m, k, 1)
        points_features.append(indicator)

    if getattr(model, 'use_feature_indicator', False):
        if feature_random_drop_mask is not None:
            indicator = torch.logical_not(feature_random_drop_mask).to(
                dtype=torch.float, device=points_n.device)  # (b, m,)
            indicator = indicator.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k, 1)  # (b, m, k, 1)
        elif 'captured_z_direction_w' in other_maps_name_idx or 'captured_view_direction_w' in other_maps_name_idx:
            indicator = torch.ones(*points_n.shape[:-1], 1).to(
                dtype=torch.float, device=points_n.device)  # (b, m, k, 1)
        else:
            indicator = torch.zeros(*points_n.shape[:-1], 1).to(
                dtype=torch.float, device=points_n.device)  # (b, m, k, 1)
        points_features.append(indicator)

    if len(points_features) == 0:
        points_features = None
    else:
        points_features = torch.cat(points_features, dim=-1)  # (b, m, k, dim)

    dim_point_feature = model.dim_point_feature
    if dim_point_feature > 0:
        assert points_features is not None
        assert dim_point_feature == points_features.size(-1), \
            f'{dim_point_feature} vs {points_features.shape}'

    # estimate intersection points and surface normal
    total_time_model = 0
    if max_model_chunk_size == -1 or bm <= max_model_chunk_size:
        # we handle the chunk outside pointersect
        if enable_timing:
            torch.cuda.synchronize()
        stime_model = timer()
        with profiler.record_function('model'):
            result_dict = model(
                points=points_n.to(device=model_device),  # (b, m, k, 3)
                additional_features=points_features.to(device=model_device) if points_features is not None else None,
                neighbor_num=neighbor_num.to(device=model_device) if neighbor_num is not None else None,
                printout=printout,
                max_chunk_size=-1,
                valid_mask=valid_mask.to(device=model_device) if valid_mask is not None else None,  # (b, m, k) bool
            )
        if enable_timing:
            torch.cuda.synchronize()
        total_time_model += (timer() - stime_model)

        result_dict['est_ray_rgb'] = None
        if point_cloud.rgb is not None:
            _b, _m, _k, _3 = points_rgb.shape  # (b, m, k, 3)
            if rgb_render_method == 'blending_weights' and result_dict['blending_weights'] is not None:
                result_dict['est_ray_rgb'] = (
                        result_dict['blending_weights'].unsqueeze(-1) * points_rgb
                ).sum(-2)  # (b, m, 3)
            elif rgb_render_method.startswith('gaussian-'):
                # get gaussian std
                sigma = float(rgb_render_method.split('gaussian-', 1)[1])
                # compute intersection point (in the normalized coordinate)
                ro_n = torch.zeros_like(points_rgb)  # (b, m, k, 3)
                rd_n = torch.zeros_like(points_rgb)  # (b, m, k, 3)
                rd_n[..., 2] = 1
                x0_n = ro_n + result_dict['ts'].reshape(_b, _m, 1, 1) * rd_n  # (b, m, k, 3)
                logits = (-1. * ((points_n - x0_n) ** 2).sum(dim=-1)) / (2. * (sigma ** 2))  # (b, m, k)
                if valid_mask is not None:
                    logits = logits.masked_fill(
                        torch.logical_not(logits.to(device=model_device)), -torch.inf)  # (b, m, k)
                rgb_weights = torch.nn.functional.softmax(logits, dim=-1)  # (b, m, k)

                result_dict['est_ray_rgb'] = (
                        rgb_weights.unsqueeze(-1) * points_rgb
                ).sum(-2)  # (b, m, 3)

            else:
                raise NotImplementedError

        # convert intersection points and surface normal to world coord
        result_dict['ts'] = result_dict['ts'] + ts_shift.to(device=model_device)
        result_dict['surface_normals'] = (
                Rs_n2w @ result_dict['surface_normals'].unsqueeze(-1)).squeeze(-1)  # (b, m', 3)
        result_dict['est_hits'] = result_dict['hit_logits'].sigmoid() > th_hit_prob  # (b, m')

        if result_dict['est_plane_normals'] is not None:
            result_dict['est_plane_normals'] = (Rs_n2w @ result_dict['est_plane_normals'].unsqueeze(-1)).squeeze(
                -1)  # (b, m', 3)

    else:
        assert rgb_render_method == 'blending_weights'

        # chunk b_m
        *b_m, n, dim = points_n.shape
        points_n = points_n.reshape(-1, n, dim)
        num_chunks = math.ceil(bm / max_model_chunk_size)
        chunk_dim = 0
        points_n_list = torch.chunk(points_n, chunks=num_chunks, dim=chunk_dim)

        ts_shift_list = torch.chunk(
            ts_shift.reshape(-1), chunks=num_chunks, dim=chunk_dim,
        )  # (bm,)

        Rs_n2w_list = torch.chunk(
            Rs_n2w.reshape(-1, 3, 3), chunks=num_chunks, dim=chunk_dim,
        )  # (bm, 3, 3)

        if point_cloud.rgb is not None:
            rgb_list = torch.chunk(
                points_rgb.reshape(-1, *points_rgb.shape[-2:]), chunks=num_chunks, dim=chunk_dim,
            )  # (bm, k, 3)
        else:
            rgb_list = None

        if points_features is not None:
            points_features = points_features.reshape(-1, *points_features.shape[-2:])
            points_features_list = torch.chunk(points_features, chunks=num_chunks, dim=chunk_dim)
        else:
            points_features_list = None

        if neighbor_num is not None:
            neighbor_num = neighbor_num.reshape(-1)  # (b, m) -> bm
            neighbor_num_list = torch.chunk(neighbor_num, chunks=num_chunks, dim=chunk_dim)
        else:
            neighbor_num_list = None

        if valid_mask is not None:
            valid_mask = valid_mask.reshape(-1, valid_mask.size(-1))  # (b, m, k) -> (bm, k)
            valid_mask_list = torch.chunk(valid_mask, chunks=num_chunks, dim=chunk_dim)  # (bm', k)
        else:
            valid_mask_list = None

        # go through each chunk
        print('run pointersect in chunks:')
        result_dict = []
        for jj in range(len(points_n_list)):
            # on model_device
            if enable_timing:
                torch.cuda.synchronize()
            stime_model = timer()
            with profiler.record_function('model'):
                sub_result_dict = model(
                    points=points_n_list[jj].to(device=model_device),
                    additional_features=points_features_list[jj].to(
                        device=model_device) if points_features_list is not None else None,
                    neighbor_num=neighbor_num_list[jj].to(
                        device=model_device) if neighbor_num_list is not None else None,
                    printout=printout,
                    max_chunk_size=-1,
                    valid_mask=valid_mask_list[jj].to(device=model_device) if valid_mask_list is not None else None,
                )
            if enable_timing:
                torch.cuda.synchronize()
            total_time_model += (timer() - stime_model)

            # convert intersection points and surface normal to world coord
            sub_result_dict['ts'] = sub_result_dict['ts'] + ts_shift_list[jj].to(device=model_device)  # (bm')
            sub_result_dict['surface_normals'] = (
                    Rs_n2w_list[jj].to(device=model_device) @ sub_result_dict['surface_normals'].unsqueeze(-1)
            ).squeeze(-1)  # (bm', 3)
            sub_result_dict['est_hits'] = sub_result_dict['hit_logits'].sigmoid() > th_hit_prob  # (bm')

            if sub_result_dict['est_plane_normals'] is not None:
                sub_result_dict['est_plane_normals'] = (
                        Rs_n2w_list[jj].to(device=model_device) @ sub_result_dict['est_plane_normals'].unsqueeze(-1)
                ).squeeze(-1)  # (bm', 3)

            if sub_result_dict['blending_weights'] is not None and point_cloud.rgb is not None:
                sub_result_dict['est_ray_rgb'] = (
                        sub_result_dict['blending_weights'].unsqueeze(-1) * rgb_list[jj].to(device=model_device)
                ).sum(-2)  # (bm', 3)
            else:
                sub_result_dict['est_ray_rgb'] = None

            sub_result_dict = utils.to_device(sub_result_dict, device=device_cpu)
            result_dict.append(sub_result_dict)

        # combine chunk results and reshape to the original shape
        result_dict = utils.cat_dict(result_dict, dim_dict=0)
        for key in result_dict:
            arr = result_dict[key]
            if arr is not None and isinstance(arr, torch.Tensor):
                result_dict[key] = arr.reshape(*b_m, *arr.shape[1:])

    timing_info['model'] = total_time_model

    # est_ts_n = result_dict['ts']  # (b, m')
    est_ts_w = result_dict['ts']  # (b, m')
    # est_surface_normals_n = result_dict['surface_normals']  # (b, m', 3)  # last dimension is positive, normalized
    est_surface_normals_w = result_dict['surface_normals']  # (b, m', 3)  # last dimension is positive, normalized
    est_hit_logits = result_dict['hit_logits']  # (b, m')
    est_hits = result_dict['est_hits']  # (b, m')
    est_ray_rgbs = result_dict['est_ray_rgb']  # (b, m', 3)
    # est_plane_normals_n = result_dict['est_plane_normals']  # (b, m', 3)
    est_plane_normals_w = result_dict['est_plane_normals']  # (b, m', 3)
    geometry_weights = result_dict['geometry_weights']  # (b, m', k)
    blending_weights = result_dict['blending_weights']  # (b, m', k)
    out_weights = result_dict['out_weights']  # (b, m , k+1)
    valid_neighbor_idx_mask = valid_mask  # (bm, k)
    valid_est_plane_normal_mask = result_dict['valid_est_plane_normal_mask']  # (b, m')
    del result_dict

    # convert rendered results back to m_shape
    est_ts_w = est_ts_w.reshape(b, *m_shape)  # (b, m, h, w)
    est_surface_normals_w = est_surface_normals_w.reshape(b, *m_shape, 3)  # (b, m, h, w, 3)
    est_hits = est_hits.reshape(b, *m_shape)  # (b, m, h, w)
    est_hit_logits = est_hit_logits.reshape(b, *m_shape)  # (b, m, h, w)
    if est_plane_normals_w is not None:
        est_plane_normals_w = est_plane_normals_w.reshape(b, *m_shape, 3)  # (b, m, h, w, 3)
        geometry_weights = geometry_weights.reshape(b, *m_shape, k)  # (b, m, h, w, k)
    if est_ray_rgbs is not None:
        est_ray_rgbs = est_ray_rgbs.reshape(b, *m_shape, 3)  # (b, m, h, w, 3)
        blending_weights = blending_weights.reshape(b, *m_shape, k)  # (b, m, h, w, k)
    if valid_est_plane_normal_mask is not None:
        valid_est_plane_normal_mask = valid_est_plane_normal_mask.reshape(b, *m_shape)  # (b, m, h, w)

    model_attn_weights = out_weights.reshape(b, *m_shape, k + 1, -1)  # (b, m, h, w, k+1, num_layers)
    neighbor_num = neighbor_num.reshape(b, *m_shape)  # (b, *m)
    neighbor_point_idxs = neighbor_point_idxs.reshape(b, *m_shape, k)  # (b, *m, k)
    valid_neighbor_idx_mask = valid_neighbor_idx_mask.reshape(b, *m_shape, k)  # (b, *m, k)
    intersection_xyz_w = \
        camera_rays.origins_w + \
        est_ts_w.to(device=ori_device).unsqueeze(-1) * camera_rays.directions_w  # (b, *m, 3)

    if rgb_random_drop_mask is not None:
        rgb_random_drop_mask = rgb_random_drop_mask.reshape(b, *m_shape)  # (b, *m)

    if feature_random_drop_mask is not None:
        feature_random_drop_mask = feature_random_drop_mask.reshape(b, *m_shape)  # (b, *m)

    timing_info['total_intersect_pcd_and_ray'] = timer() - stime_total

    pointersect_record = PointersectRecord(
        intersection_xyz_w=intersection_xyz_w.to(device=ori_device),  # (b, *m, 3)
        intersection_surface_normal_w=est_surface_normals_w.to(device=ori_device),  # (b, *m, 3)
        intersection_rgb=est_ray_rgbs.to(device=ori_device),  # (b, *m, 3)
        blending_weights=blending_weights.to(device=ori_device),  # (b, *m, k)
        neighbor_point_idxs=neighbor_point_idxs.to(device=ori_device),  # (b, *m, k)
        neighbor_point_valid_len=neighbor_num.to(device=ori_device),  # (b, *m)
        ray_t=est_ts_w.to(device=ori_device),  # (b, *m)
        ray_hit=est_hits.to(device=ori_device),  # (b, *m),
        ray_hit_logit=est_hit_logits.to(device=ori_device),  # (b, *m)
        model_attn_weights=model_attn_weights.to(device=ori_device),  # (b, *m, k+1, n_layers)
        intersection_plane_normals_w=est_plane_normals_w.to(
            device=ori_device) if est_plane_normals_w is not None else None,  # (b, *m, 3)
        geometry_weights=geometry_weights.to(device=ori_device) if geometry_weights is not None else None,  # (b, *m k)
        valid_neighbor_idx_mask=valid_neighbor_idx_mask.to(
            device=ori_device) if valid_neighbor_idx_mask is not None else None,
        valid_plane_normal_mask=valid_est_plane_normal_mask.to(
            device=ori_device) if valid_est_plane_normal_mask is not None else None,  # (b, *m)
    )

    return dict(
        pointersect_record=pointersect_record.to(device=ori_device),
        cached_info=cached_info,
        rgb_random_drop_mask=rgb_random_drop_mask.to(device=ori_device) if rgb_random_drop_mask is not None else None,
        # (b, *m)
        feature_random_drop_mask=feature_random_drop_mask.to(
            device=ori_device) if feature_random_drop_mask is not None else None,  # (b, *m)
        timing_info=timing_info,
    )


def render_point_cloud_camera_using_pointersect(
        model_filename: str,
        k: int,
        point_cloud: PointCloud,
        output_cameras: Camera,
        pr_setting: T.Optional[T.Dict[str, T.Any]],
        model_device: torch.device,
        data_device: torch.device,
        th_hit_prob: float = 0.5,
        max_ray_chunk_size: int = int(4e4),
        max_pr_chunk_size: int = -1,
        max_model_chunk_size: int = -1,
        model: SimplePointersect = None,
        output_camera_setting: T.Optional[T.Dict[str, T.Any]] = None,
        model_loading_settings: T.Optional[T.Dict[str, T.Any]] = None,
        print_out: bool = False,
        rgb_render_method: str = 'blending_weights',
        num_samples_per_pixel: int = 1,
) -> PointersectRecord:
    """
    Given point cloud and cameras, use pointersect to render the point cloud.

    Args:
        model_filename:
            filename of the pretrained model checkpoint
        k:
            number of neighbor points used by the model
        point_cloud:
            input point cloud
        output_cameras:
            output cameras
        output_camera_setting:
            'ray_offsets': default: 'center'
        model_loading_settings:
            loss_name='test_epoch_loss_hit',
            loss_smooth_window=100,
            loss_smooth_std=30.,
        pr_setting:
            ray_radius=0.1,  # if <0, set to grid_width/grid_size *2
            grid_size=100,
            grid_center=0,
            grid_width=2.2 * mesh_scale,
        model_device:
            the device to load the model
        data_device:
            the device to load the data
        th_hit_prob:
            probability to consider a hit
        max_ray_chunk_size:
            max number of rays to send to the pointersect model at once
        max_pr_chunk_size:
            set to -1
        max_model_chunk_size:
            set to -1
        num_samples_per_pixel:
            number of rays to sample from a random location within each pixel
            (to avoid aliasing). If num_samples_per_pixel = 1, always sample
            from the center of the pixel.
            If num_samples_per_pixel > 1, we sum the results.

    Returns:
        PointersectRecord:  (b, q, h, w)
    """
    if model_loading_settings is None:
        model_loading_settings = dict()
    if output_camera_setting is None:
        output_camera_setting = dict()

    # load model
    if model is None:
        model, model_info = inference_utils.load_pointersect(
            filename=model_filename,
            device=model_device,
        )
    else:
        model.to(device=model_device)
        model_info = dict(
            model_filename=model_filename,
        )

    # render point cloud from output camera viewpoints
    # (b=1, q) -> (b=1, q, h, w)
    if num_samples_per_pixel == 1:
        output_camera_rays = output_cameras.generate_camera_rays(
            offsets='center',  # output_camera_setting.get('ray_offsets', 'center'),
            device=data_device,
        )  # (b=1, q, h, w)
    else:
        output_camera_rays = [
            output_cameras.generate_camera_rays(
                offsets='rand',
                device=data_device,
            )  # (b=1, q, h, w)
            for _ in range(num_samples_per_pixel)]
        output_camera_rays = Ray.cat(output_camera_rays, dim=1)  # (b=1, qr, h, w)

    # chunk rays
    _b, _q, _h, _w, _d = output_camera_rays.origins_w.shape
    _qhw = _q * _h * _w
    print(f'total number of points = {point_cloud.xyz_w.numel() // 3}')
    print(f'total rays to render = {_b * _qhw}')

    # adjust pr_setting for the point cloud
    xyz_max, _ = point_cloud.xyz_w.max(dim=-2)  # (b, 3)
    xyz_min, _ = point_cloud.xyz_w.min(dim=-2)  # (b, 3)

    stime = timer()
    result_dict = render_point_cloud_ray_using_pointersect(
        model=model,
        k=k,
        point_cloud=point_cloud,
        rays=output_camera_rays,  # (b, q, h, w)
        pr_setting=pr_setting,
        th_hit_prob=th_hit_prob,
        max_ray_chunk_size=max_ray_chunk_size,
        max_pr_chunk_size=max_pr_chunk_size,
        max_model_chunk_size=max_model_chunk_size,
        print_out=print_out,
        rgb_render_method=rgb_render_method,
    )
    total_time = timer() - stime
    print(f'finished')
    print(f'average speed: ')
    print(f'  {_b * _qhw / total_time:.3f} rays per second')
    print(f'  {total_time / (_b * _q):.3f} seconds per {_h}x{_w} image')

    pointersect_result: PointersectRecord = result_dict['pointersect_result']  # (b, q, h, w) or (b, qr, h, w)
    pointersect_result.model_info = model_info
    pointersect_result.total_time = total_time
    del result_dict

    # chunk pointersect result
    if num_samples_per_pixel > 1:
        pointersect_result: T.List[PointersectRecord] = pointersect_result.chunk(chunks=num_samples_per_pixel, dim=1)
        pointersect_result = PointersectRecord.aggregate(pointersect_result)

    return pointersect_result


def render_point_cloud_ray_using_pointersect(
        model: SimplePointersect,
        k: int,
        point_cloud: PointCloud,
        rays: Ray,
        pr_setting: T.Optional[T.Dict[str, T.Any]],
        th_hit_prob: float = 0.5,
        max_ray_chunk_size: int = int(4e4),
        max_pr_chunk_size: int = -1,
        max_model_chunk_size: int = -1,
        t_min: float = 1.0e-8,
        t_max: float = 1.0e10,
        cached_info=None,
        print_out: bool = False,
        requires_grad: bool = False,
        rgb_render_method: str = 'blending_weights',
        enable_timing: bool = False,
) -> T.Dict[str, T.Any]:
    """
    Given point cloud and rays, use pointersect to render the point cloud.

    Args:
        k:
            number of neighbor points used by the model
        point_cloud:
            input point cloud
        rays:
            (b, *m_shape)
        pr_setting:
            ray_radius=0.1,  # if <0, set to grid_width/grid_size *2
            grid_size=100,
            grid_center=0,
            grid_width=2.2 * mesh_scale,
        model_device:
            the device to load the model
        th_hit_prob:
            probability to consider a hit
        max_ray_chunk_size:
            max number of rays to send to the pointersect model at once
        max_pr_chunk_size:
            set to -1
        max_model_chunk_size:
            set to -1

    Returns:
        PointersectRecord:  (b, q, h, w)
    """

    # chunk rays
    _b, *m_shape, _d = rays.origins_w.shape
    rays.reshape(_b, -1, _d)  # (b=1, q, 3)
    _qhw = np.prod(m_shape)

    # chunk the ray
    chunk_dim = 1
    if max_ray_chunk_size > 0 and _qhw > max_ray_chunk_size:
        chunks = math.ceil(_qhw / max_ray_chunk_size)
        rays_list = rays.chunk(chunks, chunk_dim)
        if print_out:
            print(f'divided into {chunks} chunks of {max_ray_chunk_size} rays')
    else:
        rays_list = [rays]

    if pr_setting is None \
            or pr_setting.get('grid_center', None) is None \
            or pr_setting.get('grid_width', None) is None \
            or pr_setting.get('grid_size', None) is None \
            or pr_setting.get('ray_radius', None) is None:
        with torch.no_grad():
            if not point_cloud.included_point_at_inf:
                xyz_max, _ = point_cloud.xyz_w.max(dim=-2)  # (b, 3)
                xyz_min, _ = point_cloud.xyz_w.min(dim=-2)  # (b, 3)
            else:
                xyz_max, _ = point_cloud.xyz_w[:, 1:].max(dim=-2)  # (b, 3)
                xyz_min, _ = point_cloud.xyz_w[:, 1:].min(dim=-2)  # (b, 3)
            tmp_pr_setting = dict()
            if pr_setting is None or pr_setting.get('grid_center', None) is None:
                tmp_pr_setting['grid_center'] = (xyz_max + xyz_min) / 2  # (b, 3)
            else:
                tmp_pr_setting['grid_center'] = pr_setting['grid_center']
            if pr_setting is None or pr_setting.get('grid_width', None) is None:
                tmp_pr_setting['grid_width'] = (xyz_max - xyz_min) * 1.1  # (b, 3)
                tmp_pr_setting['grid_width'] = tmp_pr_setting['grid_width'].clamp(min=0.1)
            else:
                tmp_pr_setting['grid_width'] = pr_setting['grid_width']
            if pr_setting is None or pr_setting.get('grid_size', None) is None:
                tmp_pr_setting['grid_size'] = max(1, math.floor((point_cloud.xyz_w.size(1) / k) ** (1 / 3)) * 5)
            else:
                tmp_pr_setting['grid_size'] = pr_setting['grid_size']
            if pr_setting is None or pr_setting.get('ray_radius', None) is None:
                tmp_pr_setting['ray_radius'] = 0.1 * tmp_pr_setting['grid_width'].max(dim=-1)[0]
            else:
                tmp_pr_setting['ray_radius'] = pr_setting['ray_radius']
            pr_setting = tmp_pr_setting

    # pointersect
    stime = timer()
    pointersect_results = []

    timing_info = dict()
    for ray_idx in tqdm(range(len(rays_list)), disable=not print_out):
        with torch.set_grad_enabled(requires_grad):
            out_dict = intersect_pcd_and_ray(
                point_cloud=point_cloud,
                camera_rays=rays_list[ray_idx],  # (b, m)
                model=model,
                k=k,
                max_pr_chunk_size=max_pr_chunk_size,  # chunk ray instead of pr
                max_model_chunk_size=max_model_chunk_size,  # chunk ray instead of model
                pr_grid_size=pr_setting['grid_size'],
                pr_grid_width=pr_setting['grid_width'],
                pr_grid_center=pr_setting['grid_center'],
                pr_ray_radius=pr_setting['ray_radius'],
                th_hit_prob=th_hit_prob,
                cached_info=cached_info,
                t_min=t_min,
                t_max=t_max,
                rgb_render_method=rgb_render_method,
                enable_timing=enable_timing,
            )  # (b=1, q, h, w)  on cpu
        pointersect_result = out_dict['pointersect_record']
        cached_info = out_dict['cached_info']
        pointersect_results.append(pointersect_result)
        t_info = out_dict['timing_info']
        for key in t_info:
            if key not in timing_info:
                timing_info[key] = t_info[key]
            else:
                timing_info[key] += t_info[key]
    total_time = timer() - stime
    timing_info['total_render_point_cloud_ray_using_pointersect'] = total_time

    # combine pointersect results
    pointersect_result = PointersectRecord.cat(pointersect_results, dim=chunk_dim)
    pointersect_result.total_time = total_time
    del pointersect_results

    # reshape back to (b, q, h, w)
    pointersect_result.reshape(new_b=_b, new_m_shape=m_shape)
    rays.reshape(_b, *m_shape, _d)  # (b=1, *m, 3)

    return dict(
        pointersect_result=pointersect_result,
        cached_info=cached_info,
        total_time=total_time,
        timing_info=timing_info,
    )


def compute_metrics_for_rgbd_images(
        rgbd_images: T.List[RGBDImage],
        ref_rgbd_image: RGBDImage,
        ref_mesh: Mesh = None,
        names: T.List[str] = None,
        rgb_metric: T.List[str] = 'psnr',
        depth_metric: T.List[str] = 'rmse',
        normal_metric: T.List[str] = 'avg_angle',
        hit_metric: T.List[str] = 'accuracy',
        pcd_metric: T.List[str] = [],
        filtered_pcd_metric: T.List[str] = [],
        th_dot_product: float = 0.5,  # (less than 60 degree)
        output_filename: str = None,
) -> T.Dict[str, T.Any]:
    """
    Compare a list of rgbd_images, create gif, compute difference from ref_rgbd_image if given.
    Note we assume the camera used to capture the rgbd images are the same.

    Args:
        rgbd_images:
            list of rgbd_image to compare.
        ref_rgbd_image:
            reference rgbd image to compute the error against
        names:
            name of the rgbd_images. If None, it will become their indexes.
        rgb_metric:
            'psnr'
        depth_metric:
            'rmse',
        normal_metric:
            'avg_angle'
        hit_metric:
            'accuracy'
        pcd_metric:
            'rmse_signed_distance' (need ref_mesh to be given)
            'max_signed_distance' (need ref_mesh to be given)
            'rmse_distance' (need ref_mesh to be given)
            'max_distance' (need ref_mesh to be given)

    Returns:
        rgb_err_dicts:
            name -> error dict for rgb (metric_name -> val (b,q)). err_dict will be None if input/gt not presented
        depth_err_dicts:
            name -> error dict for deoth (metric_name -> val (b,q)). err_dict will be None if input/gt not presented
        normal_err_dicts:
            name -> error dict for normal (metric_name -> val (b,q)). err_dict will be None if input/gt not presented
        hit_err_dicts:
            name -> error dict for hit (metric_name -> val (b,q)). err_dict will be None if input/gt not presented
        pcd_err_dicts:
            name -> error dict for pcd (metric_name -> val (b,q)). err_dict will be None if input/gt not presented

    Procedure:
        - before adding the name to the image, compute the error to the reference
        - create tmp rgb, depth, normal_w, hit_map if not None.  If one content is None, skip the image
    """
    assert ref_rgbd_image is not None

    if isinstance(rgb_metric, str):
        rgb_metric = [rgb_metric]
    if isinstance(depth_metric, str):
        depth_metric = [depth_metric]
    if isinstance(normal_metric, str):
        normal_metric = [normal_metric]
    if isinstance(hit_metric, str):
        hit_metric = [hit_metric]
    if isinstance(pcd_metric, str):
        pcd_metric = [pcd_metric]

    if names is None or len(names) == 0:
        names = [f'{i}' for i in range(len(rgbd_images))]
    assert len(names) == len(rgbd_images)

    # get hit map (to identify valid pixels)
    hit_maps = [rgbd.hit_map for rgbd in rgbd_images]  # (b, q, h, w,)

    # compute errors
    # rgb: (b, q, h, w, 3)
    rgbs = [rgbd.rgb for rgbd in rgbd_images]  # (b, q, h, w, 3)
    gt = ref_rgbd_image.rgb  # (b, q, h, w, 3)
    gt_hit_map = ref_rgbd_image.hit_map  # (b, q, h, w)
    if gt_hit_map is None:
        gt_hit_map = torch.ones_like(gt[..., 0])  # (b, q, h, w)

    rgb_err_dicts = dict()  # a list containing the err_dict for each input rgbd_img
    for i in range(len(rgbs)):
        arr = rgbs[i]
        if arr is None or gt is None:
            rgb_err_dicts[names[i]] = None
            continue
        hit_map = hit_maps[i] if hit_maps[i] is not None else torch.ones(*arr.shape[:-1], device=arr.device)
        # valid_mask = gt_hit_map + hit_map
        valid_mask = torch.logical_and(gt_hit_map, hit_map)
        err_dict = dict()
        if 'lpips' in rgb_metric:
            lpips_device = torch.device('cuda')
            lpips_model = get_lpips_model(device=lpips_device)
        else:
            lpips_model = None
            lpips_device = None
        for metric_name in rgb_metric:
            if metric_name == 'psnr':
                err = inference_utils.compute_psnr(
                    arr=arr * hit_map.unsqueeze(-1) + (1 - hit_map.float()).unsqueeze(-1).expand_as(arr),
                    ref=gt * gt_hit_map.unsqueeze(-1) + (1 - gt_hit_map.float()).unsqueeze(-1).expand_as(gt),
                    # assume background is white
                    ndim_b=2,
                    max_val=1.,
                    valid_mask=None,  # calculate the full image to be fair to all methods
                    # valid_mask=valid_mask,  # calculate only on the valid hit region to be fair to all methods
                )  # (b, q)
                err_dict[metric_name] = err
                err = err[torch.isfinite(err)]
                err_dict[f'avg_{metric_name}'] = err.mean()
                err_dict[f'std_{metric_name}'] = err.std()
            elif metric_name == 'ssim':
                err = inference_utils.compute_ssim(
                    arr=arr * hit_map.unsqueeze(-1) + (1 - hit_map.float()).unsqueeze(-1).expand_as(arr),
                    ref=gt * gt_hit_map.unsqueeze(-1) + (1 - gt_hit_map.float()).unsqueeze(-1).expand_as(gt),
                    # assume background is white
                    ndim_b=2,
                )  # (b, q)
                err_dict[metric_name] = err
                err = err[torch.isfinite(err)]
                err_dict[f'avg_{metric_name}'] = err.mean()
                err_dict[f'std_{metric_name}'] = err.std()

            elif metric_name == 'lpips':
                err = inference_utils.compute_lpips(
                    arr=arr * hit_map.unsqueeze(-1) + (1 - hit_map.float()).unsqueeze(-1).expand_as(arr),
                    ref=gt * gt_hit_map.unsqueeze(-1) + (1 - gt_hit_map.float()).unsqueeze(-1).expand_as(gt),
                    # assume background is white
                    ndim_b=2,
                    lpips_model=lpips_model,
                    device=lpips_device,
                )  # (b, q)
                err_dict[metric_name] = err
                err = err[torch.isfinite(err)]
                err_dict[f'avg_{metric_name}'] = err.mean()
                err_dict[f'std_{metric_name}'] = err.std()
            else:
                raise NotImplementedError

        rgb_err_dicts[names[i]] = err_dict

    # depth: (b, q, h, w)
    device_chamfer = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    depths = [rgbd.depth for rgbd in rgbd_images]  # list of (b, q, h, w,)
    rays = [rgbd.camera.generate_camera_rays() for rgbd in rgbd_images]  # list of (b, q, h, w)
    gt = ref_rgbd_image.depth  # (b, q, h, w)
    depth_err_dicts = dict()  # a list containing the err_dict for each input rgbd_img
    for i in range(len(depths)):
        arr = depths[i]  # (b, q, h, w,)
        ray = rays[i]  # (b, q, h, w)
        if arr is None or gt is None:
            depth_err_dicts[names[i]] = None
            continue
        hit_map = hit_maps[i] if hit_maps[i] is not None else torch.ones(
            *arr.shape, dtype=torch.bool, device=arr.device)
        # valid_mask = torch.logical_or(gt_hit_map, hit_map)
        valid_mask = torch.logical_and(gt_hit_map, hit_map)
        err_dict = dict()
        for metric_name in depth_metric:
            if metric_name in err_dict:
                continue

            if metric_name == 'rmse':
                err = inference_utils.compute_rmse(
                    arr=arr * hit_map,
                    ref=gt * gt_hit_map,
                    ndim_b=2,
                    valid_mask=valid_mask,
                )  # (b, q)
                err_dict[metric_name] = err
                err = err[torch.isfinite(err)]
                err_dict[f'avg_{metric_name}'] = err.mean()
                err_dict[f'std_{metric_name}'] = err.std()

            elif metric_name in {
                'chamfer_est2gt', 'chamfer_gt2est', 'chamfer_symmetric',
                'silhouette_chamfer_est2gt', 'silhouette_chamfer_gt2est', 'silhouette_chamfer_symmetric',
                'valid_chamfer_est2gt', 'valid_chamfer_gt2est', 'valid_chamfer_symmetric',
            }:
                chamferDist = ChamferDistance()

                # we compute the distance for each view (q) separately and then merge
                # silhouette uses the gt_hip_map (not the estimated hit_map)
                all_dist_unmasked_est2gts = []
                all_dist_unmasked_gt2ests = []
                all_dist_unmasked_symmetrics = []
                all_dist_masked_est2gts = []
                all_dist_masked_gt2ests = []
                all_dist_masked_symmetrics = []
                all_dist_valid_est2gts = []
                all_dist_valid_gt2ests = []
                all_dist_valid_symmetrics = []
                for ib in range(arr.size(0)):
                    dist_unmasked_est2gts = []
                    dist_unmasked_gt2ests = []
                    dist_unmasked_symmetrics = []
                    dist_masked_est2gts = []
                    dist_masked_gt2ests = []
                    dist_masked_symmetrics = []
                    dist_valid_est2gts = []
                    dist_valid_gt2ests = []
                    dist_valid_symmetrics = []
                    for iq in range(arr.size(1)):
                        arr_q = arr[ib, iq]  # (h, w)
                        gt_q = gt[ib, iq]  # (h, w)

                        point_q_w = ray.origins_w[ib, iq] + arr_q.unsqueeze(-1) * ray.directions_w[ib, iq]  # (h, w, 3)
                        # print(f'point_q_w.shape = {point_q_w.shape}')
                        point_q_w = point_q_w.reshape(-1, 3)  # (hw, 3)
                        gt_q_w = ray.origins_w[ib, iq] + gt_q.unsqueeze(-1) * ray.directions_w[ib, iq]  # (h, w, 3)
                        # print(f'gt_q_w.shape = {gt_q_w.shape}')
                        gt_q_w = gt_q_w.reshape(-1, 3)  # (hw, 3)

                        est_hit_map_q = hit_map[ib, iq].reshape(-1)  # (hw)
                        total_est_hit_map_q = est_hit_map_q.sum().clamp(min=1)
                        # print(f'est_hit_map_q.shape = {est_hit_map_q}')
                        gt_hit_map_q = gt_hit_map[ib, iq].reshape(-1)  # (hw)
                        total_gt_hit_map_q = gt_hit_map_q.sum().clamp(min=1)
                        # print(f'gt_hit_map_q.shape = {gt_hit_map_q}')
                        valid_mask_q = valid_mask[ib, iq].reshape(-1)  # (hw)
                        total_valid_mask_q = valid_mask_q.sum().clamp(min=1)
                        # print(f'valid_mask_q.shape = {valid_mask_q}')

                        # unmasked chamfer: est -> gt
                        dist_unmasked_est2gt = chamferDist(
                            point_q_w[est_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                            gt_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                        ).detach().cpu()  # (,)
                        dist_unmasked_est2gt = dist_unmasked_est2gt / total_est_hit_map_q
                        # unmasked chamfer: gt -> est
                        dist_unmasked_gt2est = chamferDist(
                            gt_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                            point_q_w[est_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                        ).detach().cpu()  # (,)
                        dist_unmasked_gt2est = dist_unmasked_gt2est / total_gt_hit_map_q
                        dist_unmasked_symmetric = dist_unmasked_est2gt + dist_unmasked_gt2est

                        # silhouette_masked chamfer: est -> gt
                        dist_masked_est2gt = chamferDist(
                            point_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                            gt_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                        ).detach().cpu()  # (,)
                        dist_masked_est2gt = dist_masked_est2gt / total_gt_hit_map_q
                        # silhouette_masked chamfer: gt -> est
                        dist_masked_gt2est = chamferDist(
                            gt_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                            point_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                        ).detach().cpu()  # (,)
                        dist_masked_gt2est = dist_masked_gt2est / total_gt_hit_map_q
                        dist_masked_symmetric = dist_masked_est2gt + dist_masked_gt2est

                        # both_masked chamfer: est -> gt
                        dist_valid_est2gt = chamferDist(
                            point_q_w[valid_mask_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                            gt_q_w[valid_mask_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                        ).detach().cpu()  # (,)
                        dist_valid_est2gt = dist_valid_est2gt / total_valid_mask_q
                        # both_masked chamfer: gt -> est
                        dist_valid_gt2est = chamferDist(
                            gt_q_w[valid_mask_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                            point_q_w[valid_mask_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                        ).detach().cpu()  # (,)
                        dist_valid_gt2est = dist_valid_gt2est / total_valid_mask_q
                        dist_valid_symmetric = dist_valid_est2gt + dist_valid_gt2est

                        dist_unmasked_est2gts.append(dist_unmasked_est2gt)
                        dist_unmasked_gt2ests.append(dist_unmasked_gt2est)
                        dist_unmasked_symmetrics.append(dist_unmasked_symmetric)
                        dist_masked_est2gts.append(dist_masked_est2gt)
                        dist_masked_gt2ests.append(dist_masked_gt2est)
                        dist_masked_symmetrics.append(dist_masked_symmetric)
                        dist_valid_est2gts.append(dist_valid_est2gt)
                        dist_valid_gt2ests.append(dist_valid_gt2est)
                        dist_valid_symmetrics.append(dist_valid_symmetric)

                    dist_unmasked_est2gts = torch.stack(dist_unmasked_est2gts, dim=0)  # (q,)
                    dist_unmasked_gt2ests = torch.stack(dist_unmasked_gt2ests, dim=0)  # (q,)
                    dist_unmasked_symmetrics = torch.stack(dist_unmasked_symmetrics, dim=0)  # (q,)
                    dist_masked_est2gts = torch.stack(dist_masked_est2gts, dim=0)  # (q,)
                    dist_masked_gt2ests = torch.stack(dist_masked_gt2ests, dim=0)  # (q,)
                    dist_masked_symmetrics = torch.stack(dist_masked_symmetrics, dim=0)  # (q,)
                    dist_valid_est2gts = torch.stack(dist_valid_est2gts, dim=0)  # (q,)
                    dist_valid_gt2ests = torch.stack(dist_valid_gt2ests, dim=0)  # (q,)
                    dist_valid_symmetrics = torch.stack(dist_valid_symmetrics, dim=0)  # (q,)

                    all_dist_unmasked_est2gts.append(dist_unmasked_est2gts)
                    all_dist_unmasked_gt2ests.append(dist_unmasked_gt2ests)
                    all_dist_unmasked_symmetrics.append(dist_unmasked_symmetrics)
                    all_dist_masked_est2gts.append(dist_masked_est2gts)
                    all_dist_masked_gt2ests.append(dist_masked_gt2ests)
                    all_dist_masked_symmetrics.append(dist_masked_symmetrics)
                    all_dist_valid_est2gts.append(dist_valid_est2gts)
                    all_dist_valid_gt2ests.append(dist_valid_gt2ests)
                    all_dist_valid_symmetrics.append(dist_valid_symmetrics)

                all_dist_unmasked_est2gts = torch.stack(all_dist_unmasked_est2gts, dim=0)  # (b, q)
                all_dist_unmasked_gt2ests = torch.stack(all_dist_unmasked_gt2ests, dim=0)  # (b, q)
                all_dist_unmasked_symmetrics = torch.stack(all_dist_unmasked_symmetrics, dim=0)  # (b, q)
                all_dist_masked_est2gts = torch.stack(all_dist_masked_est2gts, dim=0)  # (b, q)
                all_dist_masked_gt2ests = torch.stack(all_dist_masked_gt2ests, dim=0)  # (b, q)
                all_dist_masked_symmetrics = torch.stack(all_dist_masked_symmetrics, dim=0)  # (b, q)
                all_dist_valid_est2gts = torch.stack(all_dist_valid_est2gts, dim=0)  # (b, q)
                all_dist_valid_gt2ests = torch.stack(all_dist_valid_gt2ests, dim=0)  # (b, q)
                all_dist_valid_symmetrics = torch.stack(all_dist_valid_symmetrics, dim=0)  # (b, q)

                for mname, err in [
                    ['chamfer_est2gt', all_dist_unmasked_est2gts],
                    ['chamfer_gt2est', all_dist_unmasked_gt2ests],
                    ['chamfer_symmetric', all_dist_unmasked_symmetrics],
                    ['silhouette_chamfer_est2gt', all_dist_masked_est2gts],
                    ['silhouette_chamfer_gt2est', all_dist_masked_gt2ests],
                    ['silhouette_chamfer_symmetric', all_dist_masked_symmetrics],
                    ['valid_chamfer_est2gt', all_dist_valid_est2gts],
                    ['valid_chamfer_gt2est', all_dist_valid_gt2ests],
                    ['valid_chamfer_symmetric', all_dist_valid_symmetrics],
                ]:
                    err_dict[mname] = err
                    tmp_err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = tmp_err.mean()
                    err_dict[f'std_{mname}'] = tmp_err.std()
            else:
                raise NotImplementedError

        depth_err_dicts[names[i]] = err_dict

    # normal_w: (b, q, h, w, 3)
    normal_ws = [rgbd.normal_w for rgbd in rgbd_images]  # (b, q, h, w, 3)
    gt = ref_rgbd_image.normal_w
    normal_err_dicts = dict()  # a list containing the err_dict for each input rgbd_img
    for i in range(len(normal_ws)):
        arr = normal_ws[i]
        if arr is None or gt is None:
            normal_err_dicts[names[i]] = None
            continue
        hit_map = hit_maps[i] if hit_maps[i] is not None else torch.ones(*arr.shape[:-1], device=arr.device)
        # valid_mask = torch.logical_or(gt_hit_map, hit_map)
        valid_mask = torch.logical_and(gt_hit_map, hit_map)
        err_dict = dict()
        for metric_name in normal_metric:
            if metric_name == 'avg_angle':
                err = inference_utils.compute_diff_angle(
                    arr=arr * hit_map.unsqueeze(-1),
                    ref=gt * gt_hit_map.unsqueeze(-1),
                    ndim_b=2,
                    normalized=False,
                    valid_mask=valid_mask,
                )  # (b, q)
                err_dict[metric_name] = err
                err = err[torch.isfinite(err)]
                err_dict[f'avg_{metric_name}'] = err.mean()
                err_dict[f'std_{metric_name}'] = err.std()
            else:
                raise NotImplementedError

        normal_err_dicts[names[i]] = err_dict

    # hit_map: (b, q, h, w,)
    # hit_maps = [rgbd.hit_map for rgbd in rgbd_images]  # (b, q, h, w, 3)
    gt = ref_rgbd_image.hit_map
    hit_err_dicts = dict()  # a list containing the err_dict for each input rgbd_img
    for i in range(len(hit_maps)):
        arr = hit_maps[i]
        if arr is None or gt is None:
            hit_err_dicts[names[i]] = None
            continue

        err_dict = dict()
        for metric_name in hit_metric:
            if metric_name == 'accuracy':
                err = inference_utils.compute_accuracy(
                    arr=arr,
                    ref=gt,
                    ndim_b=2,
                    valid_mask=None,
                )  # (b, q)
                err_dict[metric_name] = err
                err = err[torch.isfinite(err)]
                err_dict[f'avg_{metric_name}'] = err.mean()
                err_dict[f'std_{metric_name}'] = err.std()
            else:
                raise NotImplementedError

        hit_err_dicts[names[i]] = err_dict

    # pcd
    print(f'computing pcd metrics..', flush=True)
    pcds = []
    for i in range(len(rgbd_images)):
        try:
            pcd = rgbd_images[i].get_pcd()
        except:
            pcd = None
        pcds.append(pcd)

    pcd_err_dicts = dict()  # a list containing the err_dict for each input rgbd_img
    for i in range(len(pcds)):
        pcd = pcds[i]

        if pcd is None or ref_mesh is None:
            pcd_err_dicts[names[i]] = None
            continue

        err_dict = dict()
        for metric_name in pcd_metric:
            o3d_pcds: T.List[o3d.geometry.PointCloud] = pcd.get_o3d_pcds()
            if metric_name in ['rmse_signed_distance', 'max_signed_distance', 'detail_signed_distance']:
                if metric_name in err_dict:
                    continue

                dists = []
                for b in range(len(o3d_pcds)):
                    o3d_pcd = o3d_pcds[b]  # (n, 3)
                    points = np.asarray(o3d_pcd.points, dtype=np.float32)  # (n, 3)

                    dist = ref_mesh.scene.compute_signed_distance(
                        query_points=points,  # (n, 3)
                    )  # (n, ) open3d.cpu.pybind.core.Tensor
                    dist = dist.numpy()
                    print(f'i={i} b={b}, points.shape={points.shape}, dist.shape={dist.shape}')
                    dists.append(dist)
                dists = torch.from_numpy(np.stack(dists, axis=0)).float()  # (b, n)
                print(f'  dists.shape={dists.shape}')
                if 'rmse_signed_distance' in pcd_metric:
                    mname = 'rmse_signed_distance'
                    err = torch.sqrt((dists ** 2).mean(dim=-1))  # (b,)
                    err_dict[mname] = err.detach().cpu().numpy()  # (b,)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
                if 'max_signed_distance' in pcd_metric:
                    mname = 'max_signed_distance'
                    err, _ = dists.abs().max(dim=-1)  # (b,)
                    err_dict[mname] = err.detach().cpu().numpy()  # (b,)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
                if 'detail_signed_distance' in pcd_metric:
                    # assume b = 1
                    mname = 'detail_signed_distance'
                    err = dists
                    err_dict[mname] = err.detach().cpu().numpy()  # (b, n)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
            elif metric_name in ['rmse_distance', 'max_distance', 'detail_distance']:
                if metric_name in err_dict:
                    continue
                dists = []
                for b in range(len(o3d_pcds)):
                    o3d_pcd = o3d_pcds[b]  # (n, 3)
                    points = np.asarray(o3d_pcd.points, dtype=np.float32)  # (n, 3)

                    dist = ref_mesh.scene.compute_distance(
                        query_points=points,  # (n, 3)
                    )  # (n, ) open3d.cpu.pybind.core.Tensor
                    dist = dist.numpy()
                    dists.append(dist)
                dists = torch.from_numpy(np.stack(dists, axis=0)).float()  # (b, n)
                if 'rmse_distance' in pcd_metric:
                    mname = 'rmse_distance'
                    err = torch.sqrt((dists ** 2).mean(dim=-1))  # (b,)
                    err_dict[mname] = err.detach().cpu().numpy()  # (b,)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
                if 'max_distance' in pcd_metric:
                    mname = 'max_distance'
                    err, _ = dists.abs().max(dim=-1)  # (b,)
                    err_dict[mname] = err.detach().cpu().numpy()  # (b,)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
                if 'detail_distance' in pcd_metric:
                    # assume b = 1
                    mname = 'detail_distance'
                    err = dists
                    err_dict[mname] = err.detach().cpu().numpy()  # (b, n)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
            else:
                raise NotImplementedError

        pcd_err_dicts[names[i]] = err_dict

    # filtered pcd
    print(f'computing filtered pcd metrics..', flush=True)
    pcds = []
    for i in range(len(rgbd_images)):
        try:
            rgbd = rgbd_images[i].clone()
            dot_prod = rgbd.compute_ray_normal_dot_product()  # (b, q, h, w)
            if dot_prod is not None:
                print(f'{names[i]}: dot_prod: max = {dot_prod.max()} min = {dot_prod.min()}')
                angle = dot_prod.abs() >= th_dot_product
                if rgbd.hit_map is not None:
                    rgbd.hit_map = torch.logical_and(rgbd.hit_map, angle)
            else:
                print(f'{names[i]}: dot_prod is None')
            pcd = rgbd.get_pcd()
        except:
            print(f'{names[i]} pcd is None')
            print(traceback.format_exc())
            pcd = None
        pcds.append(pcd)

    filtered_pcd_err_dicts = dict()  # a list containing the err_dict for each input rgbd_img
    for i in range(len(pcds)):
        pcd = pcds[i]

        if pcd is None or ref_mesh is None:
            filtered_pcd_err_dicts[names[i]] = None
            continue

        err_dict = dict()
        for metric_name in filtered_pcd_metric:
            o3d_pcds: T.List[o3d.geometry.PointCloud] = pcd.get_o3d_pcds()
            if metric_name in [
                'rmse_filtered_signed_distance',
                'max_filtered_signed_distance',
                'detail_filtered_signed_distance',
            ]:
                if metric_name in err_dict:
                    continue

                dists = []
                for b in range(len(o3d_pcds)):
                    o3d_pcd = o3d_pcds[b]  # (n, 3)
                    points = np.asarray(o3d_pcd.points, dtype=np.float32)  # (n, 3)

                    dist = ref_mesh.scene.compute_signed_distance(
                        query_points=points,  # (n, 3)
                    )  # (n, ) open3d.cpu.pybind.core.Tensor
                    dist = dist.numpy()
                    print(f'i={i} b={b}, points.shape={points.shape}, dist.shape={dist.shape}')
                    dists.append(dist)
                dists = torch.from_numpy(np.stack(dists, axis=0)).float()  # (b, n)
                print(f'  dists.shape={dists.shape}')
                if 'rmse_filtered_signed_distance' in filtered_pcd_metric:
                    mname = 'rmse_filtered_signed_distance'
                    err = torch.sqrt((dists ** 2).mean(dim=-1))  # (b,)
                    err_dict[mname] = err.detach().cpu().numpy()  # (b,)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
                if 'max_filtered_signed_distance' in filtered_pcd_metric:
                    mname = 'max_filtered_signed_distance'
                    err, _ = dists.abs().max(dim=-1)  # (b,)
                    err_dict[mname] = err.detach().cpu().numpy()  # (b,)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
                if 'detail_filtered_signed_distance' in filtered_pcd_metric:
                    # assume b = 1
                    mname = 'detail_filtered_signed_distance'
                    err = dists
                    err_dict[mname] = err.detach().cpu().numpy()  # (b, n)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
            elif metric_name in [
                'rmse_filtered_distance',
                'max_filtered_distance',
                'detail_filtered_distance',
            ]:
                if metric_name in err_dict:
                    continue
                dists = []
                for b in range(len(o3d_pcds)):
                    o3d_pcd = o3d_pcds[b]  # (n, 3)
                    points = np.asarray(o3d_pcd.points, dtype=np.float32)  # (n, 3)

                    dist = ref_mesh.scene.compute_distance(
                        query_points=points,  # (n, 3)
                    )  # (n, ) open3d.cpu.pybind.core.Tensor
                    dist = dist.numpy()
                    dists.append(dist)
                dists = torch.from_numpy(np.stack(dists, axis=0)).float()  # (b, n)
                if 'rmse_filtered_distance' in filtered_pcd_metric:
                    mname = 'rmse_filtered_distance'
                    err = torch.sqrt((dists ** 2).mean(dim=-1))  # (b,)
                    err_dict[mname] = err.detach().cpu().numpy()  # (b,)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
                if 'max_filtered_distance' in filtered_pcd_metric:
                    mname = 'max_filtered_distance'
                    err, _ = dists.abs().max(dim=-1)  # (b,)
                    err_dict[mname] = err.detach().cpu().numpy()  # (b,)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
                if 'detail_filtered_distance' in filtered_pcd_metric:
                    # assume b = 1
                    mname = 'detail_filtered_distance'
                    err = dists
                    err_dict[mname] = err.detach().cpu().numpy()  # (b, n)
                    err = err[torch.isfinite(err)]
                    err_dict[f'avg_{mname}'] = err.mean()
                    err_dict[f'std_{mname}'] = err.std()
            else:
                raise NotImplementedError(f'{metric_name}')

        filtered_pcd_err_dicts[names[i]] = err_dict

    out_dict = dict(
        rgb_err_dicts=rgb_err_dicts,
        depth_err_dicts=depth_err_dicts,
        normal_err_dicts=normal_err_dicts,
        hit_err_dicts=hit_err_dicts,
        pcd_err_dicts=pcd_err_dicts,
        filtered_pcd_err_dicts=filtered_pcd_err_dicts,
    )

    if output_filename is not None:
        with open(output_filename, 'w') as f:
            json.dump(
                utils.to_numpy(out_dict),
                f,
                indent=2,
                cls=cdslib.core.utils.print_and_save.NumpyJsonEncoder,
            )

    return out_dict


def compare_rgbd_images(
        rgbd_images: T.List[RGBDImage],
        names: T.Optional[T.List[str]] = None,
        ref_rgbd_image: T.Optional[RGBDImage] = None,
        idx_ref: int = -1,
        ref_name: str = 'ground truth',
        ncols: int = -1,
        font_size: int = 24,
        font_color: T.Union[float, T.List[float], None] = None,  # [0,1]
        font_name: str = "DejaVuSans",
        background_color: T.Union[float, T.List[float]] = 1,  # [0,1]
        pad_height_px: int = 30,
        align_width: str = 'center',
        align_height: str = 'center',
        output_dir: str = None,
        overwrite: bool = False,
        save_png: bool = True,
        save_pt: bool = True,
        save_gif: bool = True,
        gif_fps: int = 10,
) -> RGBDImage:
    """
    Compare a list of rgbd_images by creating gif containing each of the rgbd_images
    Note we assume the camera used to capture the rgbd images are the same.

    Args:
        rgbd_images:
            list of rgbd_image to compare.
        names:
            name of the rgbd images. If given, name will be print on top of the images
        ref_rgbd_image:
            reference rgbd image to compute the error against (if given)
        nrows:
            number of rows in the gif
        idx_ref:
            linear idx of the ref in the gif

    Returns:

    Procedure:
        - before adding the name to the image, compute the error to the reference
        - create tmp rgb, depth, normal_w, hit_map if not None.  If one content is None, skip the image
    """

    if idx_ref < 0:
        idx_ref = len(rgbd_images)

    # insert ref image into rgbd_image
    if ref_rgbd_image is not None:
        rgbd_images = [rgbd for rgbd in rgbd_images]  # create a new list
        rgbd_images.insert(idx_ref, ref_rgbd_image)
        names = [n for n in names]  # create a new list
        names.insert(idx_ref, ref_name)

    # find the first non-None image
    img = None
    for i in range(len(rgbd_images)):
        if rgbd_images[i].rgb is not None:
            img = rgbd_images[i].rgb
            break
    if img is None:
        raise RuntimeError

    if isinstance(background_color, (int, float)):
        background_color = [background_color] * 3
    background_img = torch.ones_like(img)  # (b, q, h, w, 3)
    for c in range(3):
        background_img[..., c] = background_color[c]

    rgbs = [rgbd.rgb for rgbd in rgbd_images]  # list of (b, q, h, w, 3)
    depths = [rgbd.depth for rgbd in rgbd_images]  # list of (b, q, h, w,)
    normals = [rgbd.normal_w for rgbd in rgbd_images]  # list of (b, q, h, w, 3)
    hit_maps = [rgbd.hit_map for rgbd in rgbd_images]  # list of (b, q, h, w)
    camera = rgbd_images[0].camera

    # add name to the images
    if names is not None:
        # rgb
        arrs = rgbs
        for i in range(len(arrs)):
            arr = arrs[i]  # (b, q, h, w, 3)
            if arr is None:
                arr = background_img.clone()
            else:
                # handle hit_map
                hit_map = hit_maps[i]  # hit map is not yet padded
                if hit_map is not None:
                    arr = arr * hit_map.unsqueeze(-1) + (1 - hit_map.float()).unsqueeze(-1) * background_img

            arr = (arr.detach().cpu().numpy() * 255).astype(np.uint8)
            arr = render.add_title_to_image(
                image=arr,
                title=names[i],
                font_size=font_size,
                font_color=font_color,
                font_name=font_name,
                background_color=background_color,
                pad_height_px=pad_height_px,
                align_width=align_width,
                align_height=align_height,
            )
            arr = torch.from_numpy(arr).float() / 255.
            arrs[i] = arr  # (b, q, h, w, 3)

        # depth
        arrs = depths
        # determine the global min and max for normalization
        global_min_depth = torch.inf
        global_max_depth = -torch.inf
        for i in range(len(arrs)):
            arr = arrs[i]
            if arr is None:
                continue

            tmp_min = arr.clone()
            tmp_max = arr.clone()
            if hit_maps[i] is not None:
                tmp_min = tmp_min.masked_fill(torch.logical_not(hit_maps[i]), torch.inf)
                tmp_max = tmp_max.masked_fill(torch.logical_not(hit_maps[i]), -torch.inf)
            global_min_depth = min(global_min_depth, tmp_min.min())
            global_max_depth = max(global_max_depth, tmp_max.max())
        assert np.isfinite(global_min_depth)
        assert np.isfinite(global_max_depth)

        for i in range(len(arrs)):
            arr = arrs[i]  # (b, q, h, w)
            if arr is None:
                arr = background_img.clone()
            else:
                # normalize to [0, 1]
                arr = (arr - global_min_depth) / (global_max_depth - global_min_depth)

                # (b, q, h, w,) -> (b, q, h, w, 3)
                arr = arr.unsqueeze(-1).expand(*([-1] * (arr.ndim)), 3)

                # deal with hit_map
                hit_map = hit_maps[i]  # hit map is not yet padded
                if hit_map is not None:
                    arr = arr * hit_map.unsqueeze(-1) + (1 - hit_map.float()).unsqueeze(-1) * background_img

            arr = (arr.detach().cpu().numpy() * 255).astype(np.uint8)
            arr = render.add_title_to_image(
                image=arr,  # (b, q, h, w, 3)
                title=names[i],
                font_size=font_size,
                font_color=font_color,
                font_name=font_name,
                background_color=background_color,
                pad_height_px=pad_height_px,
                align_width=align_width,
                align_height=align_height,
            )
            arr = torch.from_numpy(arr).float() / 255.  # (b, q, h, w, 3)
            arrs[i] = arr  # (b, q, h, w, 3), we will deal with the color channel at the end

        # normal
        arrs = normals
        for i in range(len(arrs)):
            arr = arrs[i]  # (b, q, h, w, 3)
            if arr is None:
                arr = background_img.clone()
            else:
                # [-1, 1] -> [0, 1]
                arr = (arr + 1) * 0.5

                # deal with hit_map
                hit_map = hit_maps[i]  # hit map is not yet padded
                if hit_map is not None:
                    arr = arr * hit_map.unsqueeze(-1) + (1 - hit_map.float()).unsqueeze(-1) * background_img

            arr = (arr.detach().cpu().numpy() * 255).astype(np.uint8)
            arr = render.add_title_to_image(
                image=arr,
                title=names[i],
                font_size=font_size,
                font_color=font_color,
                font_name=font_name,
                background_color=background_color,
                pad_height_px=pad_height_px,
                align_width=align_width,
                align_height=align_height,
            )
            arr = torch.from_numpy(arr).float() / 255.
            # [0, 1] -> [-1, 1]
            arr = (arr - 0.5) * 2
            arrs[i] = arr  # (b, q, h, w, 3)

        # hitmap
        arrs = hit_maps

        for i in range(len(arrs)):
            arr = arrs[i]  # (b, q, h, w)
            if arr is None:
                arr = background_img.clone() * 0  # hardcode background to be 0
            else:
                # (b, q, h, w,) -> (b, q, h, w, 3)
                arr = arr.unsqueeze(-1).expand(*([-1] * (arr.ndim)), 3)

            arr = (arr.detach().cpu().numpy() * 255).astype(np.uint8)
            arr = render.add_title_to_image(
                image=arr,  # (b, q, h, w, 3)
                title=names[i],
                font_size=font_size,
                font_color=None,
                font_name=font_name,
                background_color=0,  # hard-coded to 0, otherwise everything is white
                pad_height_px=pad_height_px,
                align_width=align_width,
                align_height=align_height,
            )
            arr = torch.from_numpy(arr).float() / 255.  # (b, q, h, w, 3)
            arrs[i] = arr  # (b, q, h, w, 3), we will deal with the color channel at the end

    # tile images
    rgbs = render.tile_images(
        images=rgbs,
        ncols=ncols,
        background_color=background_color,
    )  # (b, q, h, w, 3)
    depths = render.tile_images(
        images=depths,
        ncols=ncols,
        background_color=background_color,
    )[..., 0]  # (b, q, h, w)
    normals = render.tile_images(
        images=normals,
        ncols=ncols,
        background_color=background_color,
    )  # (b, q, h, w, 3)
    hit_maps = render.tile_images(
        images=hit_maps,
        ncols=ncols,
        background_color=0,  # hard-coded to 0, otherwise everything is white
    )[..., 0]  # (b, q, h, w)

    # create new rgbd_image
    rgbd_image = RGBDImage(
        rgb=rgbs,
        depth=depths,
        camera=camera,
        normal_w=normals,
        hit_map=hit_maps,
    )

    # save the new rgbd_image
    if output_dir is not None:
        print(f'save comparison results to {output_dir}')
        rgbd_image.save(
            output_dir=output_dir,
            overwrite=overwrite,
            save_png=save_png,
            save_pt=save_pt,
            save_gif=save_gif,
            gif_fps=gif_fps,
            background_color=background_color,
            global_min_depth=0,  # no normalization of depth
            global_max_depth=1,  # no normalization of depth
            hit_only=False,  # since hit map has been added titles
        )

    return rgbd_image


def sample_new_points_using_pointersect(
        point_cloud: PointCloud,
        num_points: int,
        model_filename: str,
        method: str = 'uniform_camera',
        model: SimplePointersect = None,
        k: int = 40,
        width_px: int = 10,
        height_px: int = 10,
        fov: float = 60.,  # degree
        max_ray_chunk_size: int = 10000,
) -> PointCloud:
    """
    Sample new points from the current point cloud.

    Args:
        num_points:
            number of points to sample
        method:
            method to generate querying rays
        model:
        width_px:
        height_px:
        fov:

    Returns:
        new point cloud:  (b, num_points)
    """

    if method == 'uniform_camera':
        # adjust resolution settings
        n_imgs = max(1, num_points // (width_px * height_px))
        n_pixels_per_img = num_points / n_imgs
        width_px = max(2, math.floor(n_pixels_per_img / (width_px * height_px) * width_px))
        width_px = max(2, width_px - (width_px % 2))
        height_px = max(2, math.floor(n_pixels_per_img / width_px))
        height_px = max(2, height_px - (height_px % 2))

        # get mesh scale and center
        if not point_cloud.included_point_at_inf:
            xyz_max = point_cloud.xyz_w.max(dim=-2)[0]  # (b, 3)
            xyz_min = point_cloud.xyz_w.min(dim=-2)[0]  # (b, 3)
        else:
            xyz_max = point_cloud.xyz_w[:, 1:].max(dim=-2)[0]  # (b, 3)
            xyz_min = point_cloud.xyz_w[:, 1:].min(dim=-2)[0]  # (b, 3)
        cs = (xyz_max.mean(dim=0) + xyz_min.mean(dim=0)) / 2  # (3,)
        s = (xyz_max - xyz_min).max().detach().cpu().item() / 2

        # create uniformly placed camera
        camera = CameraTrajectory(
            mode='random',
            n_imgs=n_imgs,
            total=1,
            params=dict(
                max_angle=180,
                min_r=2 * s,
                max_r=2 * s + 1.e-9,
                origin_w=cs.detach().cpu().numpy().tolist(),
                method='LatinHypercube',
            ),
            dtype=point_cloud.xyz_w.dtype,
        ).get_camera(
            fov=fov,
            width_px=width_px,
            height_px=height_px,
            device=point_cloud.xyz_w.device,
        )

        b, q = camera.H_c2w.shape[:2]
        h, w = camera.height_px, camera.width_px

        pointersect_result: PointersectRecord = render_point_cloud_camera_using_pointersect(
            model_filename=model_filename,
            k=k,
            point_cloud=point_cloud,
            output_cameras=camera,
            pr_setting=None,
            model_device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            data_device=point_cloud.xyz_w.device,
            th_hit_prob=0.5,
            max_ray_chunk_size=max_ray_chunk_size,
            model=model,
        )

        rgbd_images = pointersect_result.get_rgbd_image(camera=camera)
        new_point_cloud = rgbd_images.get_pcd()
        return new_point_cloud

    else:
        raise NotImplementedError
