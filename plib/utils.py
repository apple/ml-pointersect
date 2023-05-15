#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
import copy
import torch
import numpy as np
import typing as T
import open3d as o3d
from plib import rigid_motion, render
import matplotlib.pyplot as plt
import math
from timeit import default_timer as timer
from pointersect.pr import pr_utils
from pygltflib.utils import glb2gltf
from pygltflib import GLTF2, BufferFormat
from pygltflib.utils import ImageFormat
import json
import os
from cdslib.core.utils.print_and_save import imagesc


def to_tensor(
        arr: T.Union[np.ndarray, T.List[np.ndarray], T.Dict[str, T.Any]],
        dtype: torch.dtype = None,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Convert each element in arr from np.ndarray to torch.Tensor.
    Note that the output share the same memory as arr.
    """
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
        if dtype is not None:
            arr = arr.to(dtype=dtype)
        return arr
    elif isinstance(arr, torch.Tensor) and dtype is not None:
        arr = arr.to(dtype=dtype)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [to_tensor(x, dtype=dtype) for x in arr]
    elif isinstance(arr, dict):
        out_dict = dict()
        for key, val in arr.items():
            out_dict[key] = to_tensor(val, dtype=dtype)
        return out_dict
    else:
        return arr


def to_numpy(
        arr: T.Union[np.ndarray, T.List[np.ndarray], T.Dict[str, T.Any]],
        dtype: np.dtype = None
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Convert each element in arr from torch.Tensor to numpy ndarray.
    Note that the output share the same memory as arr if on cpu.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr
    elif isinstance(arr, np.ndarray) and dtype is not None:
        arr = arr.astype(dtype)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [to_numpy(x, dtype=dtype) for x in arr]
    elif isinstance(arr, dict):
        out_dict = dict()
        for key, val in arr.items():
            out_dict[key] = to_numpy(val, dtype=dtype)
        return out_dict
    else:
        return arr


def to_device(
        arr: T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]],
        device: torch.device,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Send each torch.Tensor in arr to device.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.to(device=device)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [to_device(x, device=device) for x in arr]
    elif isinstance(arr, dict):
        for key, val in arr.items():
            arr[key] = to_device(val, device=device)
        return arr
    else:
        return arr


def to_dtype(
        arr: T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]],
        dtype: torch.dtype,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Send each torch.Tensor in arr to device.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.to(dtype=dtype)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [to_dtype(x, dtype=dtype) for x in arr]
    elif isinstance(arr, dict):
        for key, val in arr.items():
            arr[key] = to_dtype(val, dtype=dtype)
        return arr
    else:
        return arr


def cat_dict(
        dict_list: T.List[T.Dict[str, T.Union[torch.Tensor, T.List[torch.Tensor]]]],
        dim_dict: T.Union[int, T.Dict[str, int]],
) -> T.Dict[str, torch.Tensor]:
    """
    Given a list of dict, each of which contains torch.Tensor or a list of torch.Tensor,
    we concat along `dim` and create a dict by concat each of them.

    Args:
        dict_list:
        dim_dict:

    Returns:
        a dict having the same keys
    """
    if len(dict_list) == 0:
        return dict()

    if isinstance(dim_dict, int):
        dim = dim_dict
        dim_dict = dict()
        for key in dict_list[0]:
            dim_dict[key] = dim

    out_dict = dict()
    for key in dict_list[0]:
        out_dict[key] = [d[key] for d in dict_list]

    for key in out_dict:
        if out_dict[key][0] is None:
            out_dict[key] = None
        elif isinstance(out_dict[key][0], torch.Tensor):
            out_dict[key] = torch.cat(out_dict[key], dim=dim_dict[key])
        else:  # out_dict[key] is a list of "list of tensor"
            batch_num = len(out_dict[key])
            tensor_num = len(out_dict[key][0])
            feature_list = [''] * tensor_num  # initialize list
            for tensor_id in range(tensor_num):
                feature_list[tensor_id] = torch.cat(
                    [out_dict[key][batch_id][tensor_id] for batch_id in range(batch_num)], dim=dim_dict[key])
            out_dict[key] = feature_list

    return out_dict


def reshape(
        arr: T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]],
        start: int = 0,
        end: int = -1,  # included
        shape: T.Union[int, T.List[int], T.Tuple[int]] = None,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Given a dict or a list containing tensor,
    reshape dimension `start` to `end` to the new shape
    """
    shape = list(shape)

    if isinstance(arr, torch.Tensor):
        arr_shape = arr.shape
        if end < 0:
            end = end + arr.ndim
        new_shape = list(arr_shape[:start]) + list(shape) + list(arr_shape[end + 1:])
        arr = arr.reshape(*new_shape)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [reshape(x, start=start, end=end, shape=shape) for x in arr]
    elif isinstance(arr, dict):
        for key, val in arr.items():
            arr[key] = reshape(val, start=start, end=end, shape=shape)
        return arr
    else:
        return arr


def create_pcd(
        points: T.Union[torch.Tensor, np.ndarray],
        colors: T.Union[torch.Tensor, np.ndarray] = None,
        remove_nan_inf: bool = True,
) -> o3d.geometry.PointCloud:
    """
    Create o3d point cloud from points
    Args:
        points:
            (*, 3)
        colors:
            (*, 3) optional
        remove_nan_inf:

    Returns:
        an o3d point cloud
    """

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    points = points.reshape(-1, 3)  # (n, 3)
    if colors is not None:
        colors = colors.reshape(-1, 3)  # (n, 3)

    # remove any inf or nan points
    if remove_nan_inf:
        idxs = np.all(np.isfinite(points), axis=-1)  # (n,)
        points = points[idxs]
        if colors is not None:
            colors = colors[idxs]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def create_octree(
        points: T.Union[torch.Tensor, np.ndarray, o3d.geometry.PointCloud],
        max_depth: int = 5,
        remove_nan_inf: bool = True,
) -> o3d.geometry.Octree:
    """
    Store the xys points into an octree.

    Args:
        points:
            (*, 3)
        max_depth:
            max depth of the octree. As the tree depth increases,
            internal (and eventually leaf) nodes represents a smaller partition of 3D space.

    Returns:
        an o3d octree
    """
    if isinstance(points, o3d.geometry.PointCloud):
        pcd = points
    elif isinstance(points, (torch.Tensor, np.ndarray)):
        pcd = create_pcd(
            points=points,
            colors=None,
            remove_nan_inf=remove_nan_inf,
        )
    else:
        raise NotImplementedError

    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(
        point_cloud=pcd,
    )
    return octree


def ray_aabb_intersection(
        ray_origin: torch.Tensor,
        ray_direction: torch.Tensor,
        bbox_min_bounds: torch.Tensor,
        bbox_max_bounds: torch.Tensor,
        bbox_scaling_ratio: float = 1.0,
        t_min: float = 0.,
        t_max: float = 1.0e10,
) -> T.Dict[str, T.Any]:
    """
    Check whether a ray intersect with an axis-aligned bounding box.

    Args:
        ray_origin:
            (3,) a point on the ray (with t = 0)
        ray_direction:
            (3,) ray direction
        bbox_min_bounds:
            (3,)  top left corner
        bbox_max_bounds:
            (3,)  bottom right corner.  The bbox encloses bbox_min_bounds to bbox_max_bounds.
        bbox_scaling_ratio:
            a scalar where we will scale the bbox wrt to its center.
        t_min:
            min t to consider
        t_max:
            max t to consider

    Returns:
        is_intersected: True if intersect.
        t0: first intersection t
        t1: second intersection t
    """

    # scale bbox
    bbox_center = 0.5 * (bbox_min_bounds + bbox_max_bounds)
    bbox_min_bounds = bbox_center + (bbox_min_bounds - bbox_center) * bbox_scaling_ratio
    bbox_max_bounds = bbox_center + (bbox_max_bounds - bbox_center) * bbox_scaling_ratio

    inv_ray_direction = 1. / ray_direction  # (3,)
    _t_nears = (bbox_min_bounds - ray_origin) * inv_ray_direction  # (3,)
    _t_fars = (bbox_max_bounds - ray_origin) * inv_ray_direction  # (3,)

    t_nears = torch.where(_t_nears > _t_fars, _t_fars, _t_nears)
    t_fars = torch.where(_t_nears > _t_fars, _t_nears, _t_fars)

    t_nears[torch.isnan(t_nears)] = -torch.inf
    t_fars[torch.isnan(t_fars)] = torch.inf

    t_near = torch.max(t_nears)  # scalar, use fmin to ignore nan, no need for max
    t_far = torch.min(t_fars)  # scalar, use fmax to ignore nan, no need for min

    t_near = torch.max(t_near, torch.ones_like(t_near) * t_min)
    t_far = torch.min(t_far, torch.ones_like(t_far) * t_max)

    is_intersect = t_near <= t_far
    return dict(
        is_intersected=is_intersect,
        t_near=t_near,
        t_far=t_far,
    )


def compute_point_ray_distance_in_chunks(
        points: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        max_chunk_size: int = int(1e8),
) -> T.Dict[str, T.Any]:
    """
    Compute the distance between each point to each ray.

    Args:
        points:
            (*, n, 3)
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)
        max_chunk_size:
            max number of (*, m, n) to avoid using all memory.
            If more than `max_chunk_size`, we will chunk it and use
            for loop for each.   -1: ignored

    Returns:
        dists: (*, m, n) distance between each point to each ray
        projections:  (*, m, n, 3) the projected points on the ray
        ts: (*, m, n) length on ray (can be negative)
    """

    *b_size, n, _ = points.shape
    m = ray_origins.size(-2)

    points = points.reshape(-1, n, 3)  # (b, n, 3)
    ray_origins = ray_origins.reshape(-1, m, 3)  # (b, m, 3)
    ray_directions = ray_directions.reshape(-1, m, 3)  # (b, m, 3)
    b = points.size(0)

    if max_chunk_size < 0:
        max_chunk_size = np.inf
    # check if m*n > or <= max_chunk_size:
    # if mn > max_chunk_size: we chunk along m and for loop on b
    # if mn <= max_chunk_size: we use as large b as possible (chunk along b)
    mn = m * n
    if mn > max_chunk_size:  # chunk along m
        max_m = max(1, int(max_chunk_size / n))
        num_chunks = math.ceil(m / max_m)
        chunk_dim = 1
        ray_origins_chunks = torch.chunk(ray_origins, chunks=num_chunks, dim=chunk_dim)
        ray_directions_chunks = torch.chunk(ray_directions, chunks=num_chunks, dim=chunk_dim)
        points_chunks = [points] * len(ray_origins_chunks)  # reference
    else:
        # chunk along b
        max_b = max(1, int(max_chunk_size / mn))
        num_chunks = math.ceil(b / max_b)
        chunk_dim = 0
        ray_origins_chunks = torch.chunk(ray_origins, chunks=num_chunks, dim=chunk_dim)
        ray_directions_chunks = torch.chunk(ray_directions, chunks=num_chunks, dim=chunk_dim)
        points_chunks = torch.chunk(points, chunks=num_chunks, dim=chunk_dim)

    out_dicts = []
    for i in range(len(ray_origins_chunks)):
        out_dict = compute_point_ray_distance(
            points=points_chunks[i],
            ray_origins=ray_origins_chunks[i],
            ray_directions=ray_directions_chunks[i],
        )
        out_dicts.append(out_dict)

    # concatenate along chunk dimension
    out_dict = cat_dict(
        dict_list=out_dicts,
        dim_dict=chunk_dim,
    )

    # reshape b -> b_shape
    for key in out_dict:
        shape = list(out_dict[key].shape)
        out_dict[key] = torch.reshape(out_dict[key], list(b_size) + shape[1:])

    return out_dict


def compute_point_ray_distance(
        points: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
) -> T.Dict[str, T.Any]:
    """
    Compute the distance between each point to each ray.

    Args:
        points:
            (*, n, 3)
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)

    Returns:
        dists: (*, m, n) distance between each point to each ray
        projections:  (*, m, n, 3) the projected points on the ray
        ts: (*, m, n) length on ray (can be negative)
    """

    points = points.unsqueeze(-3)  # (*, 1, n, 3)
    ray_origins = ray_origins.unsqueeze(-2)  # (*, m, 1, 3)
    ray_directions = ray_directions.unsqueeze(-2)  # (*, m, 1, 3)
    dv = points - ray_origins  # (*, m, n, 3)

    ts = (dv * ray_directions).sum(dim=-1, keepdim=True)  # (*, m, n, 1)  projected length on ray (can be negative)
    projections = ray_origins + ts * ray_directions  # (*, m, n, 3)
    dists = torch.linalg.norm(points - projections, ord=2, dim=-1)  # (*, m, n)

    return dict(
        dists=dists,  # (*, m, n)
        projections=projections,  # (*, m, n, 3)
        ts=ts.squeeze(-1),  # (*, m, n)
    )


def get_k_neighbor_points_in_chunks(
        points: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        k: int,
        t_min: float = 0.,
        t_max: float = 1.e10,
        t_init: torch.Tensor = None,
        max_chunk_size: int = int(1e8),
        pr_params: T.Dict[str, T.Any] = None,
        printout: bool = False,
        cached_info: T.Union[T.Dict[str, torch.Tensor], None] = None,
        valid_mask: torch.Tensor = None,  # (b, n, 1)
) -> T.Dict[str, T.Any]:
    """
    Given n points (xyz) and m rays, return the neighboring points to each ray.

    Args:
        points:
            (*, n, 3)
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)
        k:
            k nearest neighbors
        t_min:
            min t to consider
        t_max:
            max t to consider
        max_chunk_size:
            max number of (*, m, n) to avoid using all memory.
            If more than `max_chunk_size`, we will chunk it and use
            for loop for each.   -1: ignored
        cached_info:
            a dictionary containing the grid cell to point index so pr does not
            need to construct it again.
        valid_mask:
            (b, n, 1) bool, whether the point should be considered in neighbor search

    Returns:
        sorted_dists:
            (*, m, min(k, n))  the distance of the k nearest points to each ray (inf if not within t range)
        sorted_idxs:
            (*, m, min(k, n)) the index of points of the k nearest points
        # dist_dict:
        #     output of :py:compute_point_ray_distance

        cached_info:
            a dict containing the cached info for pr
    """

    *b_size, n, _ = points.shape
    m = ray_origins.size(-2)
    device = points.device

    points = points.reshape(-1, n, 3)  # (b, n, 3)
    ray_origins = ray_origins.reshape(-1, m, 3)  # (b, m, 3)
    ray_directions = ray_directions.reshape(-1, m, 3)  # (b, m, 3)
    b = points.size(0)

    if valid_mask is not None and valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(-1)

    if max_chunk_size < 0:
        max_chunk_size = int(1e13)  # 1e6
    # check if m*n > or <= max_chunk_size:
    # if mn > max_chunk_size: we chunk along m and for loop on b
    # if mn <= max_chunk_size: we use as large b as possible (chunk along b)
    mn = m * n
    # if mn > max_chunk_size:  # chunk along m
    #     max_m = max(1, int(max_chunk_size / n))

    # since we use pr_cuda v3, we support large number of points
    # so we only need to concern about m being too large
    if m > max_chunk_size:  # chunk along m
        max_m = max_chunk_size
        if printout:
            print(f'max_m = {max_m}')
        num_chunks = math.ceil(m / max_m)
        chunk_dim = 1
        ray_origins_chunks = torch.chunk(ray_origins, chunks=num_chunks, dim=chunk_dim)
        ray_directions_chunks = torch.chunk(ray_directions, chunks=num_chunks, dim=chunk_dim)
        points_chunks = [points] * len(ray_origins_chunks)  # reference
        t_init_chunks = torch.chunk(t_init, chunks=num_chunks, dim=chunk_dim) if t_init is not None else None
        valid_mask_chunks = [valid_mask] * len(ray_origins_chunks)
        reuse_cache = True
    else:
        # chunk along b
        # max_b = max(1, int(max_chunk_size / mn))
        max_b = max(1, int(max_chunk_size / m))

        if printout:
            print(f'max_b = {max_b}')
        num_chunks = math.ceil(b / max_b)
        chunk_dim = 0
        ray_origins_chunks = torch.chunk(ray_origins, chunks=num_chunks, dim=chunk_dim)
        ray_directions_chunks = torch.chunk(ray_directions, chunks=num_chunks, dim=chunk_dim)
        points_chunks = torch.chunk(points, chunks=num_chunks, dim=chunk_dim)
        t_init_chunks = torch.chunk(t_init, chunks=num_chunks, dim=chunk_dim) if t_init is not None else None
        valid_mask_chunks = torch.chunk(
            valid_mask, chunks=num_chunks, dim=chunk_dim) if valid_mask is not None else None
        reuse_cache = False if num_chunks > 1 else True

    out_dicts = []
    for i in range(len(ray_origins_chunks)):
        if printout:
            if torch.cuda.is_available():
                print(
                    f'before {i}/{len(ray_origins_chunks)}-th get_k_neighbor_points: '
                    f'{torch.cuda.memory_allocated(device=device) / 2 ** 30} GB'
                )

        stime = timer()
        if pr_params is not None:
            # use pr data structure
            out_dict = get_k_neighbor_within_ray(
                points=points_chunks[i],
                ray_origins=ray_origins_chunks[i],
                ray_directions=ray_directions_chunks[i],
                k=k,
                t_min=t_min,
                t_max=t_max,
                t_init=t_init_chunks[i] if t_init_chunks is not None else None,
                printout=printout,
                cached_info=cached_info,
                valid_mask=valid_mask_chunks[i] if valid_mask_chunks is not None else None,
                **pr_params,
            )
            find_neighbor_time = timer() - stime
            if printout:
                print(f'find_neighbor_time pr={find_neighbor_time:.4f} secs', flush=True)

        else:
            # use brute force method
            out_dict = get_k_neighbor_points(
                points=points_chunks[i],
                ray_origins=ray_origins_chunks[i],
                ray_directions=ray_directions_chunks[i],
                k=k,
                t_min=t_min,
                t_max=t_max,
                t_init=t_init_chunks[i] if t_init_chunks is not None else None,
                printout=printout,
            )
            find_neighbor_time = timer() - stime
            if printout:
                print(f'find_neighbor_time normal={find_neighbor_time:.4f} secs', flush=True)

        if not reuse_cache:
            cached_info = None
        else:
            cached_info = out_dict.get('cached_info', None)

        if 'cached_info' in out_dict:
            del out_dict['cached_info']
        out_dicts.append(out_dict)
        if printout:
            if torch.cuda.is_available():
                # torch.cuda.synchronize(device)
                print(
                    f'after {i}/{len(ray_origins_chunks)}-th get_k_neighbor_points: '
                    f'{torch.cuda.memory_allocated(device=device) / 2 ** 30} GB'
                )
            for key in out_dict:
                print(f"  {key}: {out_dict[key].shape} {np.prod(out_dict[key].shape) * 4 / 2 ** 30:.2f} GB")
            # torch.cuda.synchronize(device)

    # concatenate along chunk dimension
    out_dict = cat_dict(
        dict_list=out_dicts,
        dim_dict=chunk_dim,
    )

    # reshape b -> b_shape
    for key in out_dict:
        shape = list(out_dict[key].shape)
        out_dict[key] = torch.reshape(out_dict[key], list(b_size) + shape[1:])

    out_dict['cached_info'] = cached_info
    return out_dict


def get_k_neighbor_within_ray(
        points: torch.Tensor,  # ( b, n, 3)
        ray_origins: torch.Tensor,  # ( b, m, 3)
        ray_directions: torch.Tensor,  # ( b, m, 3)
        # nohit_token: torch.Tensor,
        k: int,
        ray_radius: float = -1.,
        grid_size: int = 100,
        grid_center: float = 0,
        grid_width: float = 2,
        t_min: float = 0.,
        t_max: float = 1.e10,
        t_init: torch.Tensor = None,
        # max_chunk_size: int = int(1e8),
        printout: bool = False,
        cached_info: T.Union[T.Dict[str, torch.Tensor], None] = None,
        valid_mask: torch.Tensor = None,  # (b, n, 1)
):
    """
    Args:
        points:
        ray_origins:
        ray_directions:
        k:
        ray_radius:
        grid_size:
        grid_center:
        grid_width:
        t_min:
        t_max:
        t_init:
        printout:
        cached_info:
            a dictionary containing the grid cell to point index so pr does not
            need to construct it again.
        valid_mask:
            (b, n, 1) whether to include a point in the neighbor search.

    Returns:

    """

    device = points.device
    batch_size, n_rays, _ = ray_origins.shape

    # no_hit_token = torch.ones(batch_size, 1, 3, device = device) * torch.inf

    stime = timer()

    if cached_info is None:
        gidx2pidx_bank = None
        gidx_start_idx = None
        refresh_cache = True
    else:
        gidx2pidx_bank = cached_info.get('gidx2pidx_bank', None)
        gidx_start_idx = cached_info.get('gidx_start_idx', None)
        refresh_cache = False

    with torch.no_grad():
        if t_init is not None:
            k = k * 2  # select 2k first

        out_dict = pr_utils.find_k_neighbor_points_of_rays(
            points=points.contiguous(),
            k=k,
            ray_origins=ray_origins.contiguous(),
            ray_directions=ray_directions.contiguous(),
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_center=grid_center,
            grid_width=grid_width,
            gidx2pidx_bank=gidx2pidx_bank,
            gidx_start_idx=gidx_start_idx,
            refresh_cache=refresh_cache,
            valid_mask=valid_mask,
        )
        all_idxs = out_dict['ray2pidx_heap']  # (b, m, k)
        neighbor_num = out_dict['ray_neighbor_num']  # (b, m)
        # ray2dist_heap = out_dict['ray2dist_heap']  # (b, m, k)  we not only need dist, we also need t
        gidx2pidx_bank = out_dict['gidx2pidx_bank']  # (b, n)
        gidx_start_idx = out_dict['gidx_start_idx']  # (b, max_n_cell+1)
        cached_info = dict(
            gidx2pidx_bank=gidx2pidx_bank,
            gidx_start_idx=gidx_start_idx,
        )

        # construct invalid mask
        neighbor_idx = torch.arange(
            k,
            device=neighbor_num.device,
        ).unsqueeze(0).expand(batch_size, n_rays, k)  # (b, m, k)
        invalid_mask = neighbor_idx >= neighbor_num.unsqueeze(-1)  # (b, m, k)

        # we assume index 0 is a point at inf
        all_idxs = all_idxs.masked_fill(invalid_mask, 0)  # (b, m, k)

        neighbor_points = torch.gather(
            input=points,  # (b, n, 3)
            dim=-2,
            index=all_idxs.reshape(batch_size, n_rays * k, 1).expand(-1, -1, 3)  # (b, m*k, 3)
        )  # (b, m*k, 3)

        dist_dict = compute_point_ray_distance(
            points=neighbor_points.reshape(batch_size * n_rays, k, 3),  # ( b*m, k, 3)
            ray_origins=ray_origins.reshape(-1, 1, 3),  # ( b*m, 1, 3)
            ray_directions=ray_directions.reshape(-1, 1, 3),  # ( b*m, 1, 3)
        )

        all_dists = dist_dict['dists'].squeeze(-2).reshape(batch_size, n_rays, k)  # (b, m, k)
        all_ts = dist_dict['ts'].squeeze(-2).reshape(batch_size, n_rays, k)  # (b, m, k)

        # set invalid neighbor points (point at inf) to have negative t and dist=inf
        all_dists = all_dists.masked_fill(invalid_mask, 1e12)
        all_ts = all_ts.masked_fill(invalid_mask, t_min - 1)  # will be ignored in rectify

        if t_init is not None:
            # select k points closer to t_init from the 2k points
            k = k // 2
            assert k > 0
            sorted_point_dist = torch.square(all_ts - t_init) + torch.square(all_dists)
            _, sorted_ts_idxs = torch.sort(sorted_point_dist, dim=-1)
            sorted_ts_idxs = sorted_ts_idxs[..., :k].clone()

            all_dists = torch.gather(
                input=all_dists,  # (*, m, 2*k)
                dim=-1,
                index=sorted_ts_idxs  # (*, m, k)
            )

            all_idxs = torch.gather(
                input=all_idxs,  # (*, m, 2*k)
                dim=-1,
                index=sorted_ts_idxs  # (*, m, k)
            )

            all_ts = torch.gather(
                input=all_ts,  # (*, m, 2*k)
                dim=-1,
                index=sorted_ts_idxs  # (*, m, k)
            )

    invalid_mask = torch.logical_or(all_ts < t_min, all_ts > t_max)  # (*, m, n)
    # notes: two kinds of invalid points:
    # (1) the background point, the position itself is now set to 1e12, see render.rasterize
    # (2) points lies in the opposite direction of the ray
    all_dists[invalid_mask] = torch.inf

    # note that all of these are not really "sorted"
    # the name are just to match the usage of get_k_neighbor_points
    return dict(
        sorted_dists=all_dists,  # (*, m, k)
        sorted_idxs=all_idxs,  # (*, m, k)
        sorted_ts=all_ts,  # (*, m, k) length on ray (can be negative)
        neighbor_num=neighbor_num,  # (*, m) number of neighbors of each ray
        cached_info=cached_info,
    )


def get_k_neighbor_points(
        points: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        k: int,
        t_min: float = 0.,
        t_max: float = 1.e10,
        t_init: torch.Tensor = None,
        printout: bool = False,
) -> T.Dict[str, T.Any]:
    """
    Given n points (xyz) and m rays, return the neighboring points to each ray.

    Args:
        points:
            (*, n, 3)
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)
        k:
            k nearest neighbors
        t_min:
            min t to consider
        t_max:
            max t to consider

    Returns:
        sorted_dists:
            (*, m, min(k, n))  the distance of the k nearest points to each ray (inf if not within t range)
        sorted_idxs:
            (*, m, min(k,n)) the index of points of the k nearest points
        sorted_ts:
            (*, m, min(k, n)) projection length of each point on ray from the ray_origins (can be negative)
    """
    device = points.device

    if printout:
        if torch.cuda.is_available():
            # torch.cuda.synchronize(device)
            print(f'  before compute ts: {torch.cuda.memory_allocated(device=device) / 2 ** 30} GB')
        #    torch.cuda.synchronize(device)

    dist_dict = compute_point_ray_distance(
        points=points,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
    )
    dists = dist_dict['dists']  # (*, m, n)  (ray, point)
    ts = dist_dict['ts']  # (*, m, n)

    if printout:
        if torch.cuda.is_available():
            # torch.cuda.synchronize(device)
            print(f'  after compute ts: {torch.cuda.memory_allocated(device=device) / 2 ** 30} GB')
            for key in dist_dict:
                print(f"    {key}: {dist_dict[key].shape} {np.prod(dist_dict[key].shape) * 4 / 2 ** 30:.2f} GB")
            # torch.cuda.synchronize(device)

    # map invalid ts's dist to inf
    invalid_mask = torch.logical_or(ts < t_min, ts > t_max)  # (*, m, n)
    # notes: two kinds of invalid points:
    # (1) the background point, the position itself is now set to 1e12, see render.rasterize
    # (2) points lies in the opposite direction of the ray
    dists[invalid_mask] = torch.inf

    # sort dists of the points for each ray
    # we can imagine sort() as doing two operations:
    # indices = argsort(x,dim)
    # y = gather(x,indices,dim)
    # and the argsort part is not differentiable
    # should give an option to do torch no_grad to make it clear
    # but not done yet
    # https://discuss.pytorch.org/t/differentiable-sorting-and-indices/89304
    sorted_dists, sorted_idxs = torch.sort(dists, dim=-1)  # (*, m, n), (*, m, n)

    if printout:
        if torch.cuda.is_available():
            # torch.cuda.synchronize(device)
            print(f'  before gather ts: {torch.cuda.memory_allocated(device=device) / 2 ** 30} GB')
            # torch.cuda.synchronize(device)

    # multiple passes
    # find 2k amount of neighbors first and keep only k nearest
    if t_init is not None:
        # keep only k nearest neighbors
        sorted_dists = sorted_dists[..., :2 * k].clone()  # (*, m, 2*k) the distance of the k nearest points to each ray
        sorted_idxs = sorted_idxs[..., :2 * k].clone()  # (*, m, 2*k) the index of k nearest points
        sorted_ts = torch.gather(
            input=ts,  # (*, m, n)
            dim=-1,
            index=sorted_idxs  # (*, m, 2*k)
        )  # (b, m, k)  neighbot_ts[b, m, i] = ts[b, m, neighbor_xyz_w_idxs[b, m, i]]

        # ray norm = 1, t difference = distance projected on ray
        sorted_point_dist = torch.square(sorted_ts - t_init) + torch.square(sorted_dists)
        _, sorted_ts_idxs = torch.sort(sorted_point_dist, dim=-1)
        sorted_ts_idxs = sorted_ts_idxs[..., :k].clone()

        sorted_dists = torch.gather(
            input=sorted_dists,  # (*, m, 2*k)
            dim=-1,
            index=sorted_ts_idxs  # (*, m, k)
        )

        sorted_idxs = torch.gather(
            input=sorted_idxs,  # (*, m, 2*k)
            dim=-1,
            index=sorted_ts_idxs  # (*, m, k)
        )

        sorted_ts = torch.gather(
            input=sorted_ts,  # (*, m, 2*k)
            dim=-1,
            index=sorted_ts_idxs  # (*, m, k)
        )
    else:
        # keep only k nearest neighbors
        sorted_dists = sorted_dists[..., :k].clone()  # (*, m, k) the distance of the k nearest points to each ray
        sorted_idxs = sorted_idxs[..., :k].clone()  # (*, m, k) the index of k nearest points

        sorted_ts = torch.gather(
            input=ts,  # (*, m, n)
            dim=-1,
            index=sorted_idxs  # (*, m, k)
        )  # (b, m, k)  neighbot_ts[b, m, i] = ts[b, m, neighbor_xyz_w_idxs[b, m, i]]

    if printout:
        if torch.cuda.is_available():
            # torch.cuda.synchronize(device)
            print(f'  after gather ts: {torch.cuda.memory_allocated(device=device) / 2 ** 30} GB')
            # torch.cuda.synchronize(device)

    return dict(
        sorted_dists=sorted_dists,  # (*, m, k)
        sorted_idxs=sorted_idxs,  # (*, m, k)
        sorted_ts=sorted_ts,  # (*, m, k) length on ray (can be negative)
    )


def rectify_points(
        points: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        translate: bool = False,
        randomize_translate: bool = False,
        ts: torch.Tensor = None,
        t_min: float = 0.,
        t_max: float = 1e6,
):
    """
    Given n points associated with each of the m rays (each row in points),
    rotate and translate the coordinate so that
    - ray direction becomes (0,0,1)
    - ray origin becomes (0,0,0)
    - if translate is True, the coordinate origin is chosen so that the t to the closest projection is 1

    Args:
        points:
            (*, m, n, 3)  xyz
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)
        translate:
            whether to translate the coord so that the closest point's projection on the
            ray has t = 0 for all t > 0.
        randomize_translate:
            only used when translate is true.
            If randomize_translate = true, tt will be one random distance between 0 and closest point
        ts:
            (*, m, n)  the projection length each points on the ray
            should be given only if translate is True.

    Returns:
        points_n:
            (*, m, n, 3)  the transformed points
        Rs_w2n:
            (*, m, 3, 3) the rotation matrix that transform the world coord to the rectified coord
        translation_w2n:
            (*, m, 3, 1) the translation vector that transform the world coord to the rectified coord
        tt:
            (*, m) the t that we subtract from the input ts
    """
    timing_info = dict()
    stime_total = timer()
    stime_frame = timer()
    y = torch.zeros_like(ray_directions)  # (*, m, 3)
    y[..., 1] = 1.  # try to use the current y-axis as the y-axis
    Rs_n2w = rigid_motion.construct_coord_frame(
        z=ray_directions,
        y=y,
    )  # (*, m, 3, 3)  the column in the 3*3 matrix is the axis coord with unit norm
    # Rs_n2w = torch.randn(*(ray_directions.shape[:-1]), 3, 3, device=ray_directions.device)
    # it can be thought of as the transform matrix from the new coord to the world coord
    timing_info['create_R_n2w_in_rectify'] = timer() - stime_frame

    stime_ts = timer()
    if translate:
        assert ts is not None
        ts = ts.clone()
        ts[torch.logical_or(
            ts < t_min,
            ts > t_max)] = torch.inf  # we do not care about negative ts (potential problem: all ts could be neg)
        tt, _ = ts.min(dim=-1, keepdim=True)  # (*, m, 1)
        tt[~torch.isfinite(tt)] = 0  # if all ts are neg, do not move the origin
        # in pr, invalid point is represented by background far-away point
        # do not shift if all points are background point as well
        # note that training process could occassionaly have a large hit error (should be OK since gradient is clipped)
        # maybe it is because far_thres 1e6 too large to thres out some points...?

        tt[tt > t_max] = 0
        if randomize_translate:
            # tt = tt * torch.rand(tt.shape, device=tt.device)
            tt = tt - torch.rand(tt.shape, device=tt.device) * 0.1
        else:
            pass
            # tt = tt - 0.05

        origins_w = ray_origins + tt * ray_directions  # (*, m, 3)
    else:
        tt_shape = list(ray_origins.shape)
        tt_shape[-1] = 1
        tt = torch.zeros(*tt_shape, device=ray_origins.device)  # (*, m, 1)
        origins_w = ray_origins  # (*, m, 3)
    timing_info['create_ts_in_rectify'] = timer() - stime_ts

    # create H_w2n (note the inversion)
    stime_H = timer()
    Rs_w2n = Rs_n2w.transpose(-1, -2)  # (*, m, 3, 3)
    translation_w2n = -1.0 * (Rs_w2n @ origins_w.unsqueeze(-1))  # (*, m, 3, 1)
    timing_info['create_H_in_rectify'] = timer() - stime_H

    # transform the points (m, n, 3):  (*, m, 1, 3, 3) @ (*, m, n, 3, 1) + (*, m, 1, 3, 1)
    stime_transform = timer()
    points_n = Rs_w2n.unsqueeze(-3) @ points.unsqueeze(-1) + translation_w2n.unsqueeze(-3)  # (*, m, n, 3, 1)
    timing_info['transform_in_rectify'] = timer() - stime_transform
    timing_info['total_in_rectify'] = timer() - stime_total

    return dict(
        points_n=points_n.squeeze(-1),  # (*, m, n, 3)
        Rs_w2n=Rs_w2n,  # (*, m, 3, 3)
        translation_w2n=translation_w2n,  # (*, m, 3, 1)
        tt=tt.squeeze(-1),  # (*, m) the t that we subtract from the input ts
        timing_info=timing_info,
    )


def compute_3d_xyz(
        z_map: torch.Tensor,
        intrinsic: torch.Tensor,
        H_c2w: torch.Tensor,
        subsample: int = 1,
        other_maps: T.List[torch.Tensor] = None,
):
    """
    Compute the xyz in the world coordinate using z_map and camera pose.

    Important note:
        The function uses a image coordinate system: x to right, y to "down", z to far.
        If the world coordinate is a different one (say x to right, y to "up", z to us),
        H_c2w need to include the image coordinate to world (ie. flip y and z),
        ex: H_actual * H_i2l

    Args:
        z_map:
            (*, h, w) the z coordinate of the point in the camera coordinate on the sensor,
            not along the corresponding camera ray.
        intrinsic:
            (*, 3, 3) camera intrinsic matrix
        H_c2w:
            (*, 4, 4) homegeneous matrix that convert camera coord to world coord.
            Note that the y axis should be inverted in the cam_poses.
        subsample: int
            index stride
        other_maps:
            a list of (*, h, w, d) to associated with each point

    Returns:
        points:
            (*, h//subsample, w//subsample, 3) xyz in world coordinates

    Notes:
        The function assumes the image coordinate origin is at the upper-left. So H_c2w should include
        the y-inverted transformation (e.g., H_c2w = H_c2w_y_flipped * H_flip_y).

    Notes2:
        If an element in z_map == torch.inf or torch.nan,
        the output xyz_w of the corresponding point will be torch.nan.
        Other points will be normal.

    """
    if other_maps is None:
        other_maps = []

    dtype = z_map.dtype
    device = z_map.device
    h, w = z_map.size(-2), z_map.size(-1)

    # generate u v w on the sensor coord
    u, v = torch.meshgrid(
        torch.arange(0, w, subsample, device=device),
        torch.arange(0, h, subsample, device=device),
        indexing='xy',

    )  # u: (h', w') for x,  v: (h', w') for y in the sensor coord
    # uv_shape = list(z_map.shape)
    # uv_shape[-2] = u.size(0)
    # uv_shape[-1] = u.size(1)
    # u = u.expand(uv_shape)  # (*, h', w')
    # v = v.expand(uv_shape)  # (*, h', w')

    z_map = z_map[..., v, u]  # (*, h', w')
    uvw = torch.stack(((u + 0.5) * z_map, (v + 0.5) * z_map, z_map), dim=-1).unsqueeze(-1)  # (*, h, w, 3, 1)
    inv_intrinsic = torch.linalg.inv(intrinsic).unsqueeze(-3).unsqueeze(-3)  # (*, 1, 1, 3, 3)
    xyz_c = inv_intrinsic @ uvw  # (*, h', w', 3, 1)  xyz in cam coord
    xyz_c_shape = list(xyz_c.shape)
    xyz_c_shape[-2] = 1
    xyz_c_shape[-1] = 1
    xyz1_c = torch.cat(
        (xyz_c, torch.ones(*xyz_c_shape, dtype=dtype, device=device)),
        dim=-2,
    )  # (*, h', w', 4, 1)
    H_c2w = H_c2w.unsqueeze(-3).unsqueeze(-3)  # (*, 1, 1, 4, 4)
    xyz1_w = H_c2w @ xyz1_c  # (*, h', w', 4, 1) xyz in world coord
    xyz_w = xyz1_w[..., :3, 0]  # (*, h', w', 3)

    # other_maps
    all_features = []
    for o_map in other_maps:
        if o_map is not None:
            out = o_map[..., v, u, :]  # (*, h', w', d)
        else:
            out = None
        all_features.append(out)

    return dict(
        xyz_w=xyz_w,
        other_maps=all_features,
    )


def compute_xyz_w_from_uv(
        uv_c: torch.Tensor,
        z_c: torch.Tensor,
        intrinsic: torch.Tensor,
        H_c2w: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the xyz_w in the world coordinate using the image coordinate and its z_c.

    Important note:
        The function assumes an image coordinate system: x to right, y to "down", z to far.
        If the world coordinate is a different one (say x to right, y to "up", z to us),
        H_c2w need to include the image coordinate to world (ie. flip y and z),
        ex: H_actual * H_i2l

    Args:
        uv_c:
            (*b_shape, *n_shape, 2) the image coordinate (in pixel), u to the right, v to down,
            (0, 0) is top left, (w, h) is bottom right
            In other words, u is column index, v is row index.
        z_c:
            (*b_shape, *n_shape, )  the z coordinate of the point in the camera coordinate on the sensor,
            not along the corresponding camera ray.
        intrinsic:
            (*b_shape, 3, 3) camera intrinsic matrix
        H_c2w:
            (*b_shape, 4, 4) homegeneous matrix that convert camera coord to world coord.
            Note that the y axis should be inverted in the cam_poses.

    Returns:
        (*b_shape, *n_shape, 3) xyz in world coordinates

    Notes:
        The function assumes the image coordinate origin is at the upper-left. So H_c2w should include
        the y-inverted transformation (e.g., H_c2w = H_c2w_y_flipped * H_flip_y).

    """

    *b_shape, _, _ = intrinsic.shape
    n_shape = z_c.shape[len(b_shape):]

    z_c = z_c.unsqueeze(-1)  # (*b_shape, *n_shape, 1)
    uvw = torch.cat(
        [
            uv_c * z_c,
            z_c
        ], dim=-1)  # (*b_shape, *n_shape, 3)

    uvw = uvw.reshape(*b_shape, *n_shape, 3, 1)  # (*b, *n, 3, 1)

    inv_intrinsic = torch.linalg.inv(intrinsic)  # (*b, 3, 3)
    inv_intrinsic = inv_intrinsic.reshape(*b_shape, *([1] * len(n_shape)), 3, 3)  # (*b, *n, 3, 3)

    xyz_c = inv_intrinsic @ uvw  # (*b, *n, 3, 1)  xyz in cam coord
    xyz_c_shape = list(xyz_c.shape)
    xyz_c_shape[-2] = 1
    xyz1_c = torch.cat(
        [
            xyz_c,  # (*b, *n, 3, 1)
            torch.ones(*xyz_c_shape, device=xyz_c.device, dtype=xyz_c.dtype),  # (*b, *n, 1, 1)
        ], dim=-2)  # (*b, *n, 4, 1)

    H_c2w = H_c2w.reshape(*b_shape, *([1] * len(n_shape)), 4, 4)  # (*b, *n, 4, 4)
    xyz1_w = H_c2w @ xyz1_c  # (*b, *n, 4, 1) xyz in world coord
    xyz_w = xyz1_w[..., :3, 0]  # (*b, *n, 3)

    return xyz_w


def pinhole_projection(
        xyz_w: torch.Tensor,
        intrinsics: torch.Tensor,
        H_c2w: torch.Tensor,
        dim_b: int = 0,
) -> torch.Tensor:
    """
    Compute the image coordinates of the 3D points in the world.

    Args:
        xyz_w:
            (*b_shape, *n_shape, 3) the points in world coordinate
        intrinsics:
            (*b_shape, *m_shape, 3, 3) the camera intrinsics
        H_c2w:
            (*b_shape, *m_shape, 4, 4) homegeneous matrix (camera -> world)
        dim_b:
            length of b_shape
    Returns:
        (*b_shape, *m_shape, *n_shape, 2) (col, row) on the images, can be outside of the image boundary
    """
    *bn_shape, _ = xyz_w.shape
    *bm_shape, _, _ = intrinsics.shape
    n_shape = bn_shape[dim_b:]
    m_shape = bm_shape[dim_b:]
    assert bn_shape[:dim_b] == bm_shape[:dim_b]
    b_shape = bn_shape[:dim_b]

    H_w2c = rigid_motion.inv_homogeneous_tensors(H_c2w)  # (*b, *m, 4, 4)
    H_w2c = H_w2c.reshape(*b_shape, *m_shape, *([1] * len(n_shape)), 4, 4)  # (*b, *m, *n, 4, 4)
    intrinsics = intrinsics.reshape(*b_shape, *m_shape, *([1] * len(n_shape)), 3, 3)  # (*b, *m, *n, 3, 3)
    xyz_w = torch.cat(
        [
            xyz_w,  # (*b, *n, 3)
            torch.ones(*b_shape, *n_shape, 1, device=xyz_w.device, dtype=xyz_w.dtype),
        ], dim=-1)  # (*b, *n, 4)
    xyz_w = xyz_w.reshape(*b_shape, *([1] * len(m_shape)), *n_shape, 4, 1)  # (*b, *m, *n, 4, 1)
    xyz_c = H_w2c @ xyz_w  # (*b, *m, *n, 4, 1)
    uvw_c = intrinsics @ xyz_c[..., :3, :]  # (*b, *m, *n, 3, 1)
    uv_c = uvw_c[..., :2, 0] / uvw_c[..., 2:3, 0]  # (*b, *m, *n, 2)
    return uv_c


def find_corresponding_uv(
        uv_c: torch.Tensor,
        z_map: torch.Tensor,
        intrinsics_from: torch.Tensor,
        H_c2w_from: torch.Tensor,
        intrinsics_to: torch.Tensor,
        H_c2w_to: torch.Tensor,
        dim_b: int = 0,
) -> torch.Tensor:
    """
    Compute the correspoding points in image coordinates of the pixels in a source image.

    Args:
        uv_c:
            (*b_shape, *n_shape, 2) the source pixels, on image grid, integer, within image boundary
            u is along the column axis, v is along the row axis.  [0, w], [0, h]
            Note that the pixel center is at [x.5, y.5].
        z_map:
            (*b_shape, h, w) the z coordinate of the point in the source camera coordinate on the sensor,
            not along the corresponding camera ray.
        intrinsics_from:
            (*b_shape, 3, 3) the source camera intrinsics
        H_c2w_from:
            (*b_shape, 4, 4) the homegeneous matrix (source camera -> world)
        intrinsics_to:
            (*b_shape, *m_shape, 3, 3) the target camera intrinsics
        H_c2w_to:
            (*b_shape, *m_shape, 4, 4) the homegeneous matrix (target camera -> world)
        dim_b:
            number of dimensions in b.  <= 1

    Returns:
        (*b_shape, *m_shape, *n_shape, 2) uv, on the images, can be outside of the image boundary.
        Note that the pixel center is at x.5, y.5.
        The projected uv from each of the *n_shape to each of the *m_shape
    """
    assert dim_b <= 1, f'only support dim_b = 0 or dim_b = 1'
    *bn_shape, _ = uv_c.shape
    b_shape = bn_shape[:dim_b]
    n_shape = bn_shape[dim_b:]
    h, w = z_map.size(-2), z_map.size(-1)
    b = np.prod(b_shape)
    n = np.prod(n_shape)

    # get z_c
    z_c = uv_sampling(
        uv=uv_c.reshape(b, n, 2),
        feature_map=z_map.reshape(b, h, w, 1),  # (b, h, w, 1)
        uv_normalized=False,
    )  # (b, n, dim=1)
    z_c = z_c.reshape(*b_shape, *n_shape)

    # project to world coord
    xyz_w = compute_xyz_w_from_uv(
        uv_c=uv_c,  # (*b_shape, *n_shape, 2)
        z_c=z_c,  # (*b_shape, *n_shape)
        intrinsic=intrinsics_from,  # (*b_shape, 3, 3)
        H_c2w=H_c2w_from,  # (*b_shape, 4, 4)
    )  # (*b_shape, *n_shape, 3)

    # map xyz_w to each camera
    uv_cs = pinhole_projection(
        xyz_w=xyz_w,  # (*b_shape, *n_shape, 3)
        intrinsics=intrinsics_to,  # (*b_shape, *m_shape, 3, 3)
        H_c2w=H_c2w_to,  # (*b_shape, *m_shape, 4, 4)
        dim_b=dim_b,
    )  # (*b_shape, *m_shape, *n_shape, 2)

    return uv_cs


def uv_sampling(
        uv: torch.Tensor,  # (b, *p, 2)
        feature_map: torch.Tensor,  # (b, h, w, dim)
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        uv_normalized: bool = True,
):
    """
    Sample the feature map at the uv values.

    Args:
        uv:
            (b, *p, 2)  values between [0, 1]
        feature_map:
            (b, h, w, dim), boundary corresponded to u=0, u=1, v=0, v=1.  (0,0) at top left, u to right, v to down
        pad_one_outsize:
            If true, values outside feature_map will be set as 1, else 0
        mode:
            mode used by grid_sample
        padding_mode:
            padding mode used by grid_sample. "zeros", "border", "reflection"
        uv_normalized:
            whether uv is normalized to [0, 1]. if None, uv is in the range of [0, w] [0, h]

    Returns:
        resampled_feature:
            (b, *p, dim)
    """

    if not uv_normalized:
        b, h, w, dim = feature_map.shape
        uv = uv.clone()
        uv[..., 0] = uv[..., 0] / w
        uv[..., 1] = uv[..., 1] / h

    # [0, 1] -> [-1, 1] used by grid_sampling
    uv = 2 * uv - 1  # (b, *p, 2)

    b, *p_shape, _2 = uv.shape
    assert _2 == 2
    uv = uv.reshape(b, 1, -1, 2)  # (b, 1, p, 2)

    # (b, h, w, dim) -> (b, dim, h, w)
    feature_map = feature_map.permute(0, 3, 1, 2)  # (b, dim, h, w)

    resampled_feature = torch.nn.functional.grid_sample(
        input=feature_map,  # (b, dim, h, w)
        grid=uv,  # (b, 1, p, 2)
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    # (b, dim, 1, p) -> (b, 1, p, dim)
    resampled_feature = resampled_feature.permute(0, 2, 3, 1)  # (b, 1, p, dim)
    resampled_feature = resampled_feature.reshape(b, *p_shape, resampled_feature.size(-1))  # (b, *p, dim)

    return resampled_feature  # (b, *p, dim)


def compute_3d_zdir_and_dps(
        z_map: torch.Tensor,
        intrinsic: torch.Tensor,
        H_c2w: torch.Tensor,
        subsample: int = 1,
):
    """
    Compute the zdir and dps in the world coordinate using z_map and camera pose.

    Args:
        z_map:
            (*, h, w) the z coordinate of the point in the camera coordinate on the sensor,
            not along the corresponding camera ray.
        intrinsic:
            (*, 3, 3) camera intrinsic matrix
        H_c2w:
            (*, 4, 4) homegeneous matrix that convert camera coord to world coord.
            Note that the y axis should be inverted in the cam_poses.
        subsample: int
            index stride
        other_maps:
            a list of (*, h, w, d) to associated with each point

    Returns:
        zdir_w:
            (*, h//subsample, w//subsample, 3) camera z direction in world coordinates
        dps_w:
            (*, h//subsample, w//subsample, 1) distance per sample in world coordinates
        dps_uw:
            (*, h//subsample, w//subsample, 3) distance per sample in u direction in world coordinates
        dps_vw:
            (*, h//subsample, w//subsample, 3) distance per sample in v direction in world coordinates
    """

    dtype = z_map.dtype
    device = z_map.device
    h, w = z_map.size(-2), z_map.size(-1)

    # generate u v w on the sensor coord
    u, v = torch.meshgrid(
        torch.arange(0, w, subsample, device=device),
        torch.arange(0, h, subsample, device=device),
        indexing='xy',
    )  # u: (h', w') for x,  v: (h', w') for y in the sensor coord

    z_map = z_map[..., v, u]  # (*, h', w')
    valid_z = z_map < 1e11  # default background z is set to 1e12
    inv_intrinsic = torch.linalg.inv(intrinsic).unsqueeze(-3).unsqueeze(-3)  # (*, 1, 1, 3, 3)

    # get distance per sample in u/v direction;
    dps_u = torch.stack((subsample * z_map, 0 * z_map), dim=-1)  # (*, h, w, 2) # distance per sample in u direction
    dps_v = torch.stack((0 * z_map, subsample * z_map), dim=-1)  # (*, h, w, 2) # distance per sample in v direction
    dps_uv = torch.stack((dps_u, dps_v), dim=-1)

    dps_uvc = inv_intrinsic[..., :2, :2] @ dps_uv  # (*, h', w', 2, 2) distance in cam cord

    dps_uvc_shape = list(dps_uvc.shape)
    dps_uvc_shape[-2] = 1
    # dps_uvc_shape[-1] = 2

    dps0_uvc = torch.cat(
        (dps_uvc, torch.zeros(*dps_uvc_shape, dtype=dtype, device=device)),
        dim=-2,
    )  # (*, h', w', 3, 2)
    dps_uvw = H_c2w[..., :3, :3].unsqueeze(-3).unsqueeze(-3) @ dps0_uvc  # (*, h', w', 3, 2)
    dps_uw = dps_uvw[..., 0]  # (*, h', w', 3)
    dps_vw = dps_uvw[..., 1]  # (*, h', w', 3)
    dps_w = torch.norm(dps_uw, dim=-1, keepdim=True)  # (*, h', w',1) # assume the same for u and v

    # if not hit, dps is set to 0
    dps_uw = dps_uw * valid_z.unsqueeze(-1)
    dps_vw = dps_vw * valid_z.unsqueeze(-1)
    dps_w = dps_w * valid_z.unsqueeze(-1)  # norm

    # get z direction for the camera
    one_shape = dps_uvc_shape
    one_shape[-1] = 1
    zero_shape = copy.deepcopy(one_shape)
    zero_shape[-2] = 2

    zdir_c = torch.cat(
        (
            torch.zeros(*zero_shape, dtype=dtype, device=device),
            torch.ones(*one_shape, dtype=dtype, device=device),
        ),
        dim=-2,
    )  # (*, h', w', 3, 1)
    zdir_w = H_c2w[..., :3, :3].unsqueeze(-3).unsqueeze(-3) @ zdir_c  # (*, h', w', 3, 1)
    zdir_w = zdir_w[..., 0]  # (*, h', w', 3)

    return dict(
        zdir_w=zdir_w,
        dps_w=dps_w,
        dps_uw=dps_uw,
        dps_vw=dps_vw,
    )


def plot_points_and_rays(
        points: torch.Tensor,  # (n, 3)
        ray_origins: torch.Tensor,  # (m, 3)
        ray_directions: torch.Tensor,  # (m, 3)
        ray_lengths: T.Union[torch.Tensor, float] = 10.,  # (m, 3) or float
        special_points: torch.Tensor = None,  # (p, 3)
        fig=None,
        point_size: float = 0.1,
        point_alpha: float = 0.1,
        special_point_size: float = 0.5,
        ray_color: T.List[float] = 'b',
        ray_linewidth: float = 0.1,
):
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        xs=points[:, 0],
        ys=points[:, 1],
        zs=points[:, 2],
        s=point_size,
        alpha=point_alpha,
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # plot a ray
    t = ray_lengths
    xs_from, xs_to = ray_origins[..., 0], ray_origins[..., 0] + t * ray_directions[..., 0]  # (n,)
    ys_from, ys_to = ray_origins[..., 1], ray_origins[..., 1] + t * ray_directions[..., 1]  # (n,)
    zs_from, zs_to = ray_origins[..., 2], ray_origins[..., 2] + t * ray_directions[..., 2]  # (n,)

    for i in range(len(xs_from)):
        ax.plot(
            xs=[xs_from[i], xs_to[i]],
            ys=[ys_from[i], ys_to[i]],
            zs=[zs_from[i], zs_to[i]],
            color=ray_color,
            linewidth=ray_linewidth,
        )
    # plot origin
    ax.scatter(
        xs=ray_origins[..., 0],
        ys=ray_origins[..., 1],
        zs=ray_origins[..., 2],
        s=point_size,
    )
    # plot intersection
    if special_points is not None:
        ax.scatter(
            xs=special_points[..., 0],
            ys=special_points[..., 1],
            zs=special_points[..., 2],
            s=special_point_size,
            c='r',
            marker='x',
        )

    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)

    return fig, ax


def generate_camera_rays(
        cam_poses: torch.Tensor,  # (m, 4, 4) target camera pose, cam to world
        intrinsics: torch.Tensor,  # (m, 3, 3)  intrinsic matrix of the camrea
        width_px: int,
        height_px: int,
        subsample: int = 1,  # only trace 1 ray every subsample sensor pixels
        offsets: T.Union[float, str, torch.Tensor] = 'center',
        device=torch.device('cpu'),
):
    """
    Generate camera rays (origin and direction) in the world coordinate.
    The function reproduces `o3d.t.geometry.RaycastingScene.create_rays_pinhole`
    when offset = ''center.

    Args:
        cam_poses:
            (m, 4, 4) homegeneous matrix that transforms the camera coord to world coord
            Note that to use the rays to render an image, you can have the y axis inverted in the cam_poses.
        intrinsics:
            (m, 3, 3) intrinsic matrix for each camera pose
        width_px:
            number of pixels on the sensor (before subsample)
        height_px:
            number of pixels on the sensor (before subsample)
        subsample:
            subsample the sensor (camera ray)
        offsets:
            float or (m, h, w) that will be added to the pixel location on the sensor
            If 0 or 'center', ray will be coming from the center of a pixel
            'rand': offset = [-0.5, 0.5)
        device:

    Returns:
        ray_origins_w:  (m, h', w', 3)
        ray_directions_w:  (m, h', w', 3)  normalized to unit norm

    Note:
        The function does not flip the y axis (which should be handled by the image coordinate)
        To use the rays to render an image, you can have the y axis inverted in the cam_poses.
    """

    m = cam_poses.size(0)

    # generate u v w on the sensor coord
    u, v = torch.meshgrid(
        torch.arange(0, width_px, subsample, dtype=cam_poses.dtype, device=device),
        torch.arange(0, height_px, subsample, dtype=cam_poses.dtype, device=device),
        indexing='xy',
    )  # u: (h', w') for x,  v: (h', w') for y in the sensor coord
    u = u + 0.5  # (h', w')
    v = v + 0.5  # (h', w')

    uv = torch.stack((u, v), dim=-1).to(dtype=cam_poses.dtype)  # (h', w', 2)
    uv_shape = uv.shape  # (h', w', 2)
    uv = uv.expand(m, *uv_shape)  # (m, h', w', 2)
    if isinstance(offsets, str):
        if offsets == 'center':
            pass
        elif offsets == 'rand':
            offsets = torch.rand_like(uv) - 0.5  # (m, h', w', 2)
            uv = uv + offsets
        else:
            raise NotImplementedError
    elif isinstance(offsets, torch.Tensor):
        # given the 2d uv offset (m, 2)
        uv = uv + offsets.to(dtype=uv.dtype, device=device).unsqueeze(1).unsqueeze(2)  # (m, 1, 1, 2)
    else:
        raise NotImplementedError

    return generate_camera_rays_from_uv(
        cam_poses=cam_poses,
        intrinsics=intrinsics,
        uv=uv,  # (m, h', w', 2)
        device=device,
    )


def generate_camera_rays_from_uv(
        cam_poses: torch.Tensor,  # (m, 4, 4) target camera pose, cam to world
        intrinsics: torch.Tensor,  # (m, 3, 3)  intrinsic matrix of the camrea
        uv: torch.Tensor,  # (m, *p, 2)
        device=torch.device('cpu'),
) -> T.Union[torch.Tensor, torch.Tensor]:
    """
    Generate camera rays (origin and direction) in the world coordinate given
    uv coordinate on the image. The uv coordinate on image is origin at top left,
    u to right, v to bottom.

    Args:
        cam_poses:
            (m, 4, 4) homegeneous matrix that transforms the camera coord to world coord
            Note that to use the rays to render an image, you can have the y axis inverted in the cam_poses.
        intrinsics:
            (m, 3, 3) intrinsic matrix for each camera pose
        width_px:
            number of pixels on the sensor (before subsample)
        height_px:
            number of pixels on the sensor (before subsample)
        uv:
            uv coordinate.  uv[..., 0]: u coordinate [0, w], uv[..., 1]: v coordinate [0, h],
        device:

    Returns:
        ray_origins_w:  (m, *p, 3)
        ray_directions_w:  (m, *p, 3)  normalized to unit norm

    Note:
        The function does not flip the y axis (which should be handled by the image coordinate)
        To use the rays to render an image, you can have the y axis inverted in the cam_poses.
    """

    m = cam_poses.size(0)
    _m, *p_shape, _2 = uv.shape
    assert m == _m
    assert _2 == 2

    uv1 = torch.cat(
        (
            uv,
            torch.ones(_m, *p_shape, 1, dtype=uv.dtype, device=uv.device)
        ), dim=-1).to(dtype=cam_poses.dtype, device=device)  # (m, *p, 3)

    # compute the inverse of the intrinsic matrices
    inv_intrinsics = torch.linalg.inv(intrinsics.to(device=device))  # (m, 3, 3)
    inv_intrinsics = inv_intrinsics.reshape(m, *([1] * len(p_shape)), 3, 3)  # (m, 1, 1, 3, 3)
    ray_directions_c = inv_intrinsics @ uv1.unsqueeze(-1)  # (m, *p, 3, 1)

    # cam coord -> world coord
    cam_poses = cam_poses.reshape(m, *([1] * len(p_shape)), 4, 4).to(device=device)  # (m, *p, 4, 4)
    ray_directions_w = cam_poses[..., :3, :3] @ ray_directions_c  # (m, *p, 3, 1), not normalized
    ray_origins_w = cam_poses[..., :3, 3].clone().expand(m, *p_shape, 3)  # (m, *p, 3)

    # normalize direction
    ray_directions_w = ray_directions_w.squeeze(-1)  # (m, *p, 3)
    ray_directions_w = ray_directions_w / torch.linalg.vector_norm(
        ray_directions_w, dim=-1, keepdims=True)  # (m, *p, 3)

    return ray_origins_w, ray_directions_w


def sample_regions_camera_rays_and_features(
        cam_poses: torch.Tensor,  # (m, 4, 4) target camera pose, cam to world
        intrinsics: torch.Tensor,  # (m, 3, 3)  intrinsic matrix of the camera
        sample_center: torch.Tensor,
        region_width_px: int,
        region_height_px: int,
        features: T.Dict[str, torch.Tensor] = None,
        device=torch.device('cpu'),
):
    """
    Used in PCD version of inverse rendering.

    Args:
        cam_poses:
        intrinsics:
        sample_center:
        region_width_px:
        region_height_px:
        features:
        device:

    Returns:

    """
    uv_c = pinhole_projection(
        xyz_w=sample_center,
        intrinsics=intrinsics,
        H_c2w=cam_poses,
    )
    uv_c = torch.squeeze(uv_c)  # (m, 2)

    uv_offset = uv_c - torch.Tensor([region_width_px / 2, region_height_px / 2]).to(device=device).unsqueeze(0)

    ray_origins_w, ray_directions_w = generate_camera_rays(
        cam_poses=cam_poses,
        intrinsics=intrinsics,
        width_px=region_width_px,
        height_px=region_height_px,
        subsample=1,
        offsets=uv_offset,
        device=device,
    )  # (b*m, h, w, 3), (b*m, h, w, 3) normalized

    u, v = torch.meshgrid(
        torch.arange(0, region_width_px, 1, device=device),
        torch.arange(0, region_height_px, 1, device=device),
        indexing='xy',
    )  # u: (h', w') for x,  v: (h', w') for y in the sensor coord
    u = u + 0.5
    v = v + 0.5

    if features is not None:
        resampled_features = dict()
        for key in features.keys():
            feature = features[key]
            assert len(feature.shape) >= 3, "dimension of feature should contain at least n, h, w "
            if len(feature.shape) == 3:
                feature = feature.unsqueeze(-1)

            # the part below should be replaced by another function resample_uv

            b, width_px, height_px, _ = feature.shape
            feature = feature.transpose(1, 3)

            grid = torch.stack([u, v], dim=2).unsqueeze(0) + uv_offset.unsqueeze(1).unsqueeze(2)  # (m, h, w, 2)
            grid = grid / torch.Tensor([width_px / 2, height_px / 2]).to(device=device) - 1  # normalize to [-1,1]

            # grid sample: in default is a bilinear interpolator
            # see https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            resampled_feature = torch.nn.functional.grid_sample(feature, grid)
            resampled_feature = resampled_feature.transpose(1, 3)
            resampled_features[key] = resampled_feature

    out_dict = dict(
        ray_origins=ray_origins_w,
        ray_directions=ray_directions_w,
    )

    if features is not None:
        out_dict['resampled_features'] = resampled_features

    return out_dict


def plot_multiple_images(
        imgs: T.Union[torch.Tensor, np.ndarray],
        dpi=150,
        mode='tile',  # 'horizontal', 'vertical', 'tile'
        fig=None,
        ax=None,
        colorbar=True,
        valrange=None,  # (min, max)
        ncols: int = 6,
        background_color: float = 0.,
):
    """Plot multiple images by concatenate them in space. """
    # imgs: (b, h, w, *)

    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs)

    mask = torch.logical_not(imgs.isfinite())
    imgs = imgs.masked_fill(mask, 0)

    imgs_list = torch.chunk(imgs, chunks=imgs.shape[0], dim=0)
    imgs_list = [img[0] for img in imgs_list]  # list of (h, w, *)

    if mode == 'horizontal':
        imgs = torch.cat(imgs_list, dim=1)
    elif mode == 'vertical':
        imgs = torch.cat(imgs_list, dim=0)
    elif mode == 'tile':
        assert len(imgs_list) > 0
        if imgs_list[0].ndim == 2:
            squeeze_last_dim = True
            imgs_list = [img.unsqueeze(-1) for img in imgs_list]  # list of (h, w, c)
        else:
            squeeze_last_dim = False
        imgs = render.tile_images(
            images=imgs_list,
            ncols=ncols,
            background_color=background_color,
        )
        if squeeze_last_dim:
            imgs = imgs.squeeze(-1)
    else:
        raise NotImplementedError

    if valrange is not None:
        imgs = (imgs - valrange[0]) / (valrange[1] - valrange[0])

    imgs = imgs.detach().cpu().numpy()

    # plot
    fig, axes = imagesc(
        arr=imgs,
        fig=fig,
        axes=ax,
        dpi=dpi,
        colorbar=colorbar,
    )
    plt.axis('scaled')
    return fig, axes


def fit_hyperplane(
        points: torch.Tensor,  # (*, n, d)
        centers: torch.Tensor = None,  # (*, d)
        weights: torch.Tensor = None,  # (*, n)
        th_eig_val: float = 1.e-3,
) -> T.Dict[str, T.Any]:
    """
    Fit a hyperplane for the n points such that the plane minimizes the point-to-plane distances.

    .. math::

        \min_{n,c}  \sum_i w_i (n^T * (p_i - c))^2,

    where :math:`n` is the normal of the hyperplane, :math:`c` is the anchor of the plane.

    Args:
        points:
            (*, n, d) the points in d-dimensinoal space
        centers:
            (*, d) the given anchors of the hyperplanes
        weights:
            (*, n) the weight to each points.  If None, all points have unit weights.
        # ray_origins:
        #     (*, d) the ray origins.  If given, we will also compute the intersection of the ray on the plane.
        # ray_directions:
        #     (*, d) the ray directions. If given, we will also compute the intersection of the ray on the plane.

    Returns:
        plane_normals:
            (*, d)  Note that the plane normal can points to one of the two directions
        centers:
            (*, d)
        ts:
            (*,) or None.  ts on the ray to the plane.

    """
    *p_shape, d = points.shape
    if weights is None:
        weights = torch.ones(*p_shape, 1, dtype=points.dtype, device=points.device)  # (*, n, 1)
    else:
        weights = weights.unsqueeze(-1)  # (*, n, 1)

    if centers is None:
        centers = (points * weights).sum(dim=-2) / weights.sum(dim=-2)  # (*, d)

    points_centered = points - centers.unsqueeze(-2)  # (*, n, d)
    PTP = (weights * points_centered).transpose(-1, -2) @ points_centered  # (*, d, d)

    # we will make sure PTP is at least rank 2 (at least two points not on the same line)
    with torch.no_grad():
        eig_vals, eig_vecs = torch.linalg.eigh(PTP)  # eig_vecs: (*, d, d),  eig_vals: (*, d) small to large
        valid_mask = (eig_vals[..., -2] > th_eig_val)  # (*,)
        valid_mask = torch.logical_and(
            valid_mask,
            (eig_vals[..., 1] - eig_vals[..., 0]) > th_eig_val,
        )  # (*,)

    # fill PTP with full rank matrix if invalid
    PTP = PTP.masked_fill(~valid_mask.unsqueeze(-1).unsqueeze(-1), 0) + \
          torch.randn_like(PTP).masked_fill(valid_mask.unsqueeze(-1).unsqueeze(-1), 0)

    # surface normal is the smallest eigen-vector of PTP
    eig_vals, eig_vecs = torch.linalg.eigh(PTP)  # eig_vecs: (*, d, d),  eig_vals: (*, d)
    plane_normals = eig_vecs[..., 0]  # (*, d)

    return dict(
        plane_normals=plane_normals,  # (*, d)
        centers=centers,  # (*, d)
        # eig_vals=eig_vals,  # (*, d) in ascending order
        valid_mask=valid_mask,  # (*,) bool.  True: valid to use
    )


def plane_ray_intersection(
        plane_centers: torch.Tensor,  # (*, d)
        plane_normals: torch.Tensor,  # (*, d)
        ray_origins: torch.Tensor,  # (*, d)
        ray_directions: torch.Tensor,  # (*, d)
) -> torch.Tensor:
    """
    Compute the intersection between plane_i and ray_i.

    Returns:
        ts: (*,)
    """

    nt_d = (ray_directions * plane_normals).sum(dim=-1)  # (*,)
    co = plane_centers - ray_origins  # (*, d)
    nt_co = (plane_normals * co).sum(dim=-1)  # (*,)
    mask = nt_d.abs() < 1.e-8  # (*,)
    ts = nt_co / nt_d  # (if nt_d == 0 -> t = inf)
    ts = ts.masked_fill(mask, torch.inf)
    return ts  # (*,)


def baseline_pcd_ray_intersection(
        points_w: np.ndarray,  # (n, 3)
        cam_poses: np.ndarray,  # (m, 4, 4) target camera pose, cam to world
        intrinsics: np.ndarray,  # (m, 3, 3)  intrinsic matrix of the camrea
        width_px: int,
        height_px: int,
        k: int,
        method: str = 'alpha',  # 'poisson', 'alpha', 'ball'
        poisson_depth: int = 9,
        alpha: float = 0.01,
        ball_radii: T.List[float] = (0.005, 0.01, 0.02, 0.04),
        points_rgb: np.ndarray = None,  # (n, 3)
):
    """
    Baseline point cloud-ray intersection.

    Algorithm:
    1. poisson surface reconstruction from point cloud to create a mesh
    2. ray tracing using the mesh to get surface normal

    Returns:
        mesh:
        est_ts_w:
            (m, h, w) distance on the ray direction to the intersection point
        est_surface_normals_w:
            (m, h, w, 3) surface normal at the intersection point, in the world coordinate
        est_hits:
            (m, h, w)  whether the ray intersect with a surface
    """

    # create pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_w)
    # if points_rgb is not None:
    #    pcd.rgbs = o3d.utility.Vector3dVector(points_rgb)

    # estimate normal at each vertex
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=k)

    # create mesh from pcd
    if method == 'poisson':
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    elif method == 'alpha':
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
    elif method == 'ball':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(ball_radii)
        )
    else:
        raise NotImplementedError

    # create ray tracing scene
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    extrinsic_matrices = [rigid_motion.RigidMotion.invert_homogeneous_matrix(cam_poses[i]) for i in
                          range(cam_poses.shape[0])]

    # ray trace to get surface normal (using ray tracing)
    all_rays = []  # (h, w, 6)  [ox, oy, oz, dx, dy, dz]
    all_surface_normals = []  # (h, w, 3)  [dx, dy, dz]
    all_ts = []  # (h, w)  inf if not hit
    for i in range(cam_poses.shape[0]):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsics[i],
            extrinsic_matrix=extrinsic_matrices[i],
            width_px=width_px,
            height_px=height_px,
        )  # (height_px, width_px, 6)  [ox, oy, oz, dx, dy, dz]  origin is the pinhole

        # cast the rays, get the intersections
        raycast_results = scene.cast_rays(rays)
        t_hits = raycast_results['t_hit']  # (height_px, width_px), inf if not hit the mesh
        hit_map = 1 - np.isinf(raycast_results['t_hit'].numpy())  # (h, w)  1 if hit a surface, 0 otherwise

        # note that primitive_normals is the normal of the triangle face
        # we can use uv map to interpolate vertex normal
        # interpolate surface normal using uv map
        if mesh.has_vertex_normals():
            surface_normals = render.interp_surface_normal_from_ray_tracing_results(
                mesh=mesh,
                raycast_results=raycast_results,
            )  # (height_px, width_px, 3)
        else:
            surface_normals = raycast_results['primitive_normals'].numpy()  # (height_px, width_px, 3)

        # if not hit a surface, set surface normal to (0, 0, 0)
        surface_normals = surface_normals * np.expand_dims(hit_map, axis=-1)  # (h, w, 3)

        # make sure surface normal points in the opposite direction of the ray
        rays = rays.numpy()
        est_normal_sign = np.sign(np.sum(surface_normals * rays[..., 3:], axis=-1, keepdims=True))  # (h, w, 1)
        surface_normals = surface_normals * (-1 * est_normal_sign)

        all_rays.append(rays)
        all_surface_normals.append(surface_normals)
        all_ts.append(t_hits.numpy())

    all_ts = np.stack(all_ts, axis=0)  # (n_target_img, h, w)
    all_surface_normals = np.stack(all_surface_normals, axis=0)  # (n_target_img, h, w, 3)
    all_ray_hits = np.isfinite(all_ts)  # (n_target_img, h, w)

    return dict(
        pcd=pcd,
        mesh=mesh,
        est_ts_w=all_ts,  # (m, h, w)
        est_surface_normals_w=all_surface_normals,  # (m, h, w ,3)
        est_hits=all_ray_hits,  # (m, h, w)
    )


def generate_camera_circle_path(
        num_poses: int,
        d_to_origin: float,
        r_circle: float,
        center_angles: T.Union[torch.Tensor, np.ndarray, T.List[float]],
        invert_yz: bool = True,
        alt_yaxis: bool = False,
) -> T.Union[torch.Tensor, np.ndarray]:
    """
    Generate a camera path that looks at the world origin
    Args:
        num_poses:
            number of camera poses sampled on the circle
        d_to_origin:
            distance to the origin
        r_circle:
            radius of the circle
        center_direction:
            (2,) theta (angle between x-axis), phi (angle between xy plane),
            the viewing direction of the center of the circle.  All in degree.
            The angles are given in the final coordinate (after yz is inverted)
        invert_yz:
            whether to invert the direction of y axis and z axis (since images y coord is flipped)
            This is to account for the difference in the image coordinate (x to right, y to down, z to far)
            and the world/opengl coordinate (x to right, y to up, z to us)
        alt_yaxis:
            an option to use an alternative definition of yaxis and makes a more stable circular path
    Returns:
        (num_poses, 4, 4) camera poses (that converts camera coord to world coords)
    """

    if isinstance(center_angles, np.ndarray):
        center_angles = torch.from_numpy(center_angles).float()
    elif isinstance(center_angles, (list, tuple)):
        center_angles = torch.tensor(center_angles).float()

    center_angles = center_angles.float()

    if invert_yz:
        # the coordinate is currently pre-yz-inverted
        # but center_angles are given after yz-inverted
        center_angles = -1 * center_angles

    # generate a circle on the xy plane (i.e., on the plane z = d_to_origin)
    thetas = torch.linspace(0, torch.pi * 2, num_poses) + torch.pi  # (n,)
    cam_positions_c = torch.stack(
        [
            torch.cos(thetas) * float(r_circle),
            torch.sin(thetas) * float(r_circle),
            torch.ones(num_poses) * float(d_to_origin),
        ], dim=1)  # (n, 3)

    # print(f'cam_positions_c.shape = {cam_positions_c.shape}')

    # rotate the camera positions
    v1 = torch.tensor([0, 0, 1], dtype=torch.float)
    v2 = torch.stack(
        [
            torch.cos(center_angles[1] * torch.pi / 180.) * torch.cos(center_angles[0] * torch.pi / 180.),
            torch.cos(center_angles[1] * torch.pi / 180.) * torch.sin(center_angles[0] * torch.pi / 180.),
            torch.sin(center_angles[1] * torch.pi / 180.),
        ], dim=0)

    # print(f'v1.shape = {v1.shape}')
    # print(f'v2.shape = {v2.shape}')

    R = rigid_motion.get_min_R(
        v1=v1,
        v2=v2,
    )  # (3,3)   v2 = R @ v1

    # print(f'R = {R}')

    cam_positions_w = (R.unsqueeze(0) @ cam_positions_c.unsqueeze(-1)).squeeze(-1)  # (n, 3)

    # create camera coordinate
    if not alt_yaxis:
        # all cameras look at the origin of the world -> -cam_positions_w
        ys = torch.zeros_like(cam_positions_w)
        ys[..., 1] = 1
    else:
        # the above implementation will make y-axis flip in a larger transform
        ys = torch.zeros_like(cam_positions_w)
        ys[..., 2] = 1
        ys = (R.unsqueeze(0) @ ys.unsqueeze(-1)).squeeze(-1)  # (n, 3)

    Rs_c2w = rigid_motion.construct_coord_frame(
        z=-1 * cam_positions_w,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    *b_shape, a, b = Rs_c2w.shape
    Hs_c2w = torch.zeros(*b_shape, 4, 4)
    Hs_c2w[..., :3, :3] = Rs_c2w
    Hs_c2w[..., :3, 3] = cam_positions_w
    Hs_c2w[..., 3, 3] = 1

    if invert_yz:
        H = torch.eye(4)
        H[1, 1] = -1.
        H[2, 2] = -1.
        Hs_c2w = H.unsqueeze(0) @ Hs_c2w

    return Hs_c2w  # (n, 4, 4)


def generate_camera_rect_path(
        num_poses: int,
        d_to_origin: float,
        x_length: float,
        y_length: float,
        center_angles: T.Union[torch.Tensor, np.ndarray, T.List[float]],
        x_center: float = 0,
        y_center: float = 0,
        invert_yz: bool = True,
        alt_yaxis: bool = False,
) -> T.Union[torch.Tensor, np.ndarray]:
    """
    Generate a rect camera path that looks at x_center, y_center
    Args:
        num_poses:
            number of camera poses sampled on the rect
        d_to_origin:
            distance to the origin
        x_length:
            length of x side
        y_length:
            length of y side
        x_center:
            center of x side
        y_center:
            center of y side
        center_direction:
            (2,) theta (angle between x-axis), phi (angle between xy plane),
            the viewing direction of the center of the circle.  All in degree.
            The angles are given in the final coordinate (after yz is inverted)
        invert_yz:
            whether to invert the direction of y axis and z axis (since images y coord is flipped)
            This is to account for the difference in the image coordinate (x to right, y to down, z to far)
            and the world/opengl coordinate (x to right, y to up, z to us)
        alt_yaxis:
            an option to use an alternative definition of yaxis and makes a more stable circular path
    Returns:
        (num_poses, 4, 4) camera poses (that converts camera coord to world coords)
    """

    if isinstance(center_angles, np.ndarray):
        center_angles = torch.from_numpy(center_angles).float()
    elif isinstance(center_angles, (list, tuple)):
        center_angles = torch.tensor(center_angles).float()

    center_angles = center_angles.float()

    if invert_yz:
        # the coordinate is currently pre-yz-inverted
        # but center_angles are given after yz-inverted
        center_angles = -1 * center_angles

    # generate a circle on the xy plane (i.e., on the plane z = d_to_origin)
    thetas = torch.linspace(0, torch.pi * 2, num_poses) + torch.pi  # (n,)

    step_size = 2 * (x_length + y_length) / num_poses

    # intitial poses: start from (0, y_length/2), keep increasing
    # bending the poses four times

    all_poses = np.stack(
        [np.arange(num_poses) * step_size,
         np.ones(num_poses) * y_length / 2],
        axis=-1
    )
    bend_ids = np.where(all_poses[:, 0] > (x_length / 2))[0]
    all_poses[bend_ids] = np.stack(
        [np.ones(len(bend_ids)) * x_length / 2,
         y_length / 2 + (-all_poses[bend_ids, 0] + x_length / 2)],
        axis=-1
    )
    bend_ids = np.where(all_poses[:, 1] < (-y_length / 2))[0]
    all_poses[bend_ids] = np.stack(
        [x_length / 2 + (all_poses[bend_ids, 1] + y_length / 2),
         -np.ones(len(bend_ids)) * y_length / 2],
        axis=-1
    )
    bend_ids = np.where(all_poses[:, 0] < -(x_length / 2))[0]
    all_poses[bend_ids] = np.stack(
        [-np.ones(len(bend_ids)) * x_length / 2,
         -y_length / 2 + (-all_poses[bend_ids, 0] - x_length / 2)],
        axis=-1
    )
    bend_ids = np.where(all_poses[:, 1] > (y_length / 2))[0]
    all_poses[bend_ids] = np.stack(
        [-x_length / 2 + (all_poses[bend_ids, 1] - y_length / 2),
         np.ones(len(bend_ids)) * y_length / 2],
        axis=-1
    )

    cam_positions_c = torch.stack(
        [
            torch.from_numpy(all_poses[:, 0] + x_center).to(dtype=torch.float),
            torch.from_numpy(all_poses[:, 1] + y_center).to(dtype=torch.float),
            torch.ones(num_poses) * float(d_to_origin),
        ], dim=1)  # (n, 3)

    cam_direction_c = torch.stack(
        [
            -torch.from_numpy(all_poses[:, 0]).to(dtype=torch.float),
            -torch.from_numpy(all_poses[:, 1]).to(dtype=torch.float),
            -torch.ones(num_poses) * float(d_to_origin),
        ], dim=1)  # (n, 3)

    # rotate the camera positions
    v1 = torch.tensor([0, 0, 1], dtype=torch.float)
    v2 = torch.stack(
        [
            torch.cos(center_angles[1] * torch.pi / 180.) * torch.cos(center_angles[0] * torch.pi / 180.),
            torch.cos(center_angles[1] * torch.pi / 180.) * torch.sin(center_angles[0] * torch.pi / 180.),
            torch.sin(center_angles[1] * torch.pi / 180.),
        ], dim=0)

    R = rigid_motion.get_min_R(
        v1=v1,
        v2=v2,
    )  # (3,3)   v2 = R @ v1

    cam_positions_w = (R.unsqueeze(0) @ cam_positions_c.unsqueeze(-1)).squeeze(-1)  # (n, 3)
    cam_direction_w = (R.unsqueeze(0) @ cam_direction_c.unsqueeze(-1)).squeeze(-1)  # (n, 3)

    # create camera coordinate
    if not alt_yaxis:
        # all cameras look at the origin of the world -> -cam_positions_w
        ys = torch.zeros_like(cam_positions_w)
        ys[..., 1] = 1
    else:
        # the above implementation will make y-axis flip in a larger transform
        ys = torch.zeros_like(cam_positions_w)
        ys[..., 2] = 1
        ys = (R.unsqueeze(0) @ ys.unsqueeze(-1)).squeeze(-1)  # (n, 3)

    Rs_c2w = rigid_motion.construct_coord_frame(
        # z=-1 * cam_positions_w,  # (n, 3)
        z=cam_direction_w,  # -1 * cam_positions_w,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    *b_shape, a, b = Rs_c2w.shape
    Hs_c2w = torch.zeros(*b_shape, 4, 4)
    Hs_c2w[..., :3, :3] = Rs_c2w
    Hs_c2w[..., :3, 3] = cam_positions_w
    Hs_c2w[..., 3, 3] = 1

    if invert_yz:
        H = torch.eye(4)
        H[1, 1] = -1.
        H[2, 2] = -1.
        Hs_c2w = H.unsqueeze(0) @ Hs_c2w

    return Hs_c2w  # (n, 4, 4)


def generate_camera_spiral_path(
        num_poses: int,
        num_circle: int,
        init_phi: float,
        center_angles: T.Union[torch.Tensor, np.ndarray, T.List[float]],
        r_max: float = 1,
        r_min: float = 1,
        r_freq: float = 1,
        invert_yz: bool = True,
) -> T.Union[torch.Tensor, np.ndarray]:
    """
    Generate a spiral camera path that looks at the world origin
    Args:
        num_poses:
            number of camera poses sampled on the spiral
        num_circle:
            number of circle the spiral made in xy plane
        init_phi:
            initial phi (angle between xy plane) of the path, the path will go from phi to -phi
        r_circle:
            radius of the spiral
        center_direction:
            (2,) theta (angle between x-axis), phi (angle between xy plane),
            the viewing direction of the center of the circle.  All in degree.
            The angles are given in the final coordinate (after yz is inverted)
        invert_yz:
            whether to invert the direction of y axis and z axis (since images y coord is flipped)
            This is to account for the difference in the image coordinate (x to right, y to down, z to far)
            and the world/opengl coordinate (x to right, y to up, z to us)

    Returns:
        (num_poses, 4, 4) camera poses (that converts camera coord to world coords)
    """

    if isinstance(center_angles, np.ndarray):
        center_angles = torch.from_numpy(center_angles).float()
    elif isinstance(center_angles, (list, tuple)):
        center_angles = torch.tensor(center_angles).float()

    center_angles = center_angles.float()

    if num_poses % 2 != 0:
        print('Warning: automatically change num_poses to be even')
        num_poses = num_poses + 1

    if invert_yz:
        # the coordinate is currently pre-yz-inverted
        # but center_angles are given after yz-inverted
        center_angles = -1 * center_angles

    # generate a circle on the xy plane (i.e., on the plane z = d_to_origin)
    thetas = torch.linspace(0, torch.pi * 2 * num_circle, num_poses) + torch.pi  # (n,)

    # uniformly sample along phi by cosine weighted sample
    # https://alexanderameye.github.io/notes/sampling-the-hemisphere/

    # calculate complementary phi: angle between z axis and camara position
    init_z = torch.cos(torch.pi / 2 - torch.tensor(init_phi))
    half_num_poses = int(num_poses / 2)
    comp_phi = torch.acos(torch.linspace(init_z, -init_z, half_num_poses))
    comp_phi = torch.concat([comp_phi, comp_phi[range(half_num_poses - 1, -1, -1)]], dim=0)
    phi = torch.pi / 2 - comp_phi

    r = (r_max - r_min) / 2 * torch.cos(thetas * r_freq) + (r_max + r_min) / 2

    cam_positions_c = torch.stack(
        [
            torch.cos(thetas) * torch.cos(phi) * r,
            torch.sin(thetas) * torch.cos(phi) * r,
            torch.sin(phi) * r,
        ], dim=1)  # (n, 3)

    # rotate the camera positions
    v1 = torch.tensor([0, 0, 1], dtype=torch.float)
    v2 = torch.stack(
        [
            torch.cos(center_angles[1] * torch.pi / 180.) * torch.cos(center_angles[0] * torch.pi / 180.),
            torch.cos(center_angles[1] * torch.pi / 180.) * torch.sin(center_angles[0] * torch.pi / 180.),
            torch.sin(center_angles[1] * torch.pi / 180.),
        ], dim=0)

    R = rigid_motion.get_min_R(
        v1=v1,
        v2=v2,
    )  # (3,3)   v2 = R @ v1

    cam_positions_w = (R.unsqueeze(0) @ cam_positions_c.unsqueeze(-1)).squeeze(-1)  # (n, 3)

    # create camera coordinate
    # all cameras look at the origin of the world -> -cam_positions_w
    ys = torch.stack(
        [
            -torch.cos(thetas) * torch.sin(phi),
            -torch.sin(thetas) * torch.sin(phi),
            torch.cos(phi)
        ], dim=1)

    ys = (R.unsqueeze(0) @ ys.unsqueeze(-1)).squeeze(-1)  # (n, 3)
    Rs_c2w = rigid_motion.construct_coord_frame(
        z=-1 * cam_positions_w,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    *b_shape, a, b = Rs_c2w.shape
    Hs_c2w = torch.zeros(*b_shape, 4, 4)
    Hs_c2w[..., :3, :3] = Rs_c2w
    Hs_c2w[..., :3, 3] = cam_positions_w
    Hs_c2w[..., 3, 3] = 1

    if invert_yz:
        H = torch.eye(4)
        H[1, 1] = -1.
        H[2, 2] = -1.
        Hs_c2w = H.unsqueeze(0) @ Hs_c2w

    return Hs_c2w  # (n, 4, 4)


def generate_camera_grids(
        num_x: int,
        num_y: int,
        cam_position_center,  # (1, 3)
        delta: float = 0.5,
) -> T.Union[torch.Tensor, np.ndarray]:
    if isinstance(cam_position_center, np.ndarray):
        cam_position_center = torch.from_numpy(cam_position_center).float()
    elif isinstance(cam_position_center, (list, tuple)):
        cam_position_center = torch.tensor(cam_position_center).float()

    cam_position_center = cam_position_center.float()

    ys = torch.zeros_like(cam_position_center)
    ys[..., 2] = 1
    Rs_c2w_grid = rigid_motion.construct_coord_frame(
        z=-1 * cam_position_center,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    x_sample = torch.arange(num_x) - (num_x - 1) / 2
    y_sample = torch.arange(num_y) - (num_y - 1) / 2
    grid_x, grid_y = torch.meshgrid(x_sample, y_sample)

    grid_id = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (num_x*num_y,2)
    cam_positions_w = grid_id @ Rs_c2w_grid[..., 0:2].t() * delta + cam_position_center.unsqueeze(0)

    ys = torch.zeros_like(cam_positions_w)
    ys[..., 2] = 1
    Rs_c2w = rigid_motion.construct_coord_frame(
        z=-1 * cam_positions_w,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    *b_shape, a, b = Rs_c2w.shape
    Hs_c2w = torch.zeros(*b_shape, 4, 4)
    Hs_c2w[..., :3, :3] = Rs_c2w
    Hs_c2w[..., :3, 3] = cam_positions_w
    Hs_c2w[..., 3, 3] = 1

    return Hs_c2w  # (n, 4, 4)


def generate_camera_polar_grids(
        num_theta: int,
        num_phi: int,
        r: int,
):
    """
    Sample grid on theta and phi.
    phi here: angle between z-axis and camera position

    Args:
        num_theta:
        num_phi: including two polar (0, pi)
        r:

    Returns:

    """

    theta_sample = torch.arange(num_theta + 1) / num_theta * torch.pi * 2  # [0,..., 2*pi]
    theta_sample = theta_sample[:-1]  # remove 2pi

    #  phi here: angle between z-axis and camera position
    phi_sample = torch.arange(num_phi) / (num_phi - 1) * torch.pi  # [0,..., pi]
    phi_sample = phi_sample[1:-1]  # remove polar

    theta, phi = torch.meshgrid(theta_sample, phi_sample)
    theta = theta.reshape(-1)
    phi = phi.reshape(-1)

    # position and angle of camera in grids
    grid_cam_positions = torch.stack(
        [
            torch.cos(theta) * torch.sin(phi) * r,
            torch.sin(theta) * torch.sin(phi) * r,
            torch.cos(phi) * r,
        ], dim=1)  # (n, 3)
    ys = torch.zeros_like(grid_cam_positions)
    ys[..., 2] = 1
    grid_cam_frames = rigid_motion.construct_coord_frame(
        z=-1 * grid_cam_positions,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    # position and angle of camera in polars
    polar_cam_positions = torch.cat(
        [
            torch.zeros(2, 2),
            torch.tensor([r, -r]).unsqueeze(-1)
        ], dim=1)
    ys = torch.zeros_like(polar_cam_positions)
    ys[..., 1] = 1
    polar_cam_frames = rigid_motion.construct_coord_frame(
        z=-1 * polar_cam_positions,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    # concat
    Rs_c2w = torch.cat([grid_cam_frames, polar_cam_frames], dim=0)
    cam_positions_w = torch.cat([grid_cam_positions, polar_cam_positions], dim=0)

    *b_shape, a, b = Rs_c2w.shape
    Hs_c2w = torch.zeros(*b_shape, 4, 4)
    Hs_c2w[..., :3, :3] = Rs_c2w
    Hs_c2w[..., :3, 3] = cam_positions_w
    Hs_c2w[..., 3, 3] = 1

    # find neighbor camera positions
    grid_num = num_theta * (num_phi - 2)
    id_grid = np.concatenate(
        [
            np.ones([num_theta, 1]) * grid_num,
            np.arange(num_theta * (num_phi - 2)).reshape([num_theta, num_phi - 2]),
            np.ones([num_theta, 1]) * (grid_num + 1)
        ], axis=1)
    id_grid = np.concatenate([id_grid, id_grid[[0], :]], axis=0)
    id_grid = id_grid.astype('int')

    # list of neighbor set
    neighbor_ids = [None] * (grid_num + 2)
    for self_id in range(len(neighbor_ids)):
        neighbor_ids[self_id] = set()

    for i in range(num_theta + 1):
        for j in range(num_phi):
            id_ij = id_grid[i, j]
            neighbor_ids[id_ij].add(id_grid[max(i - 1, 0), j])
            neighbor_ids[id_ij].add(id_grid[min(i + 1, num_theta), j])
            neighbor_ids[id_ij].add(id_grid[i, max(j - 1, 0)])
            neighbor_ids[id_ij].add(id_grid[i, min(j + 1, num_phi - 1)])

    # remove self from neighbors
    for self_id in range(len(neighbor_ids)):
        if self_id in neighbor_ids[self_id]:
            neighbor_ids[self_id].remove(self_id)

    return Hs_c2w, neighbor_ids  # (n, 4, 4)


def get_o3d_camera_frame(
        H_c2w: T.Union[torch.Tensor, np.ndarray],
        frame_size: float = 1.0,
) -> o3d.geometry.TriangleMesh:
    """Create a camera coordinate frame (as a mesh) in the world coordinate."""

    if isinstance(H_c2w, torch.Tensor):
        H_c2w = H_c2w.detach().cpu().numpy()
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    cam_frame.transform(H_c2w)

    return cam_frame


def draw_geometries(
        geometry_list: T.List[o3d.geometry.Geometry],
        window_name: str = '',
        width: int = 1920,
        height: int = 1080,
        left: int = 50,
        top: int = 50,
) -> T.Union[o3d.visualization.Visualizer, None]:
    """
    Mimic the behavior of :py:`o3d.visualization.draw_geometries` but free objects properly.

    Args:
        geometry_list:
            List of geometries to be visualized.
        window_name:
            The displayed title of the visualization window
        width:
            The width of the visualization window.
        height:
            The height of the visualization window.
        left:
            The left margin of the visualization window.
        top:
            The top margin of the visualization window.

    Returns:
        vis or None
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=width,
        height=height,
        left=left,
        top=top,
        window_name=window_name,
        visible=True,
    )
    # show back face to make sure ray-casting and rendering results are the same
    vis.get_render_option().mesh_show_back_face = True

    for mesh in geometry_list:
        vis.add_geometry(mesh)
    vis.run()
    vis = destroy_vis(vis)


def destroy_vis(vis: T.Union[o3d.visualization.Visualizer, None]):
    if vis is not None:
        vis.clear_geometries()
        del vis
        vis = None
    return vis


def clean_up_glb_write_gltf(
        filename_glb: str,
        overwrite_gltf: bool,
        exists_ok: bool,
        image_ext: str = '.png',
) -> str:
    """
    Clean up glb such that open3d can load the texture maps.
    Note that the function creates a gtlf file (with the same name)
    from the glb file. If the .gltf file exists and `overwrite_gltf`
    is False, it throws a runtime error.

    Args:
        filename_glb:
            the filename of the glb file.
        overwrite_gltf:
            whether to overwrite the gltf file in the same folder
            if existed.

    Returns:
        filename_gltf: str
    """

    name, ext = os.path.splitext(filename_glb)
    filename_gltf = f'{name}.gltf'

    if os.path.exists(filename_gltf):
        if overwrite_gltf:
            os.remove(filename_gltf)
        elif exists_ok:
            pass
        else:
            raise RuntimeError(f'glft file {filename_gltf} already exists')

    # convert glb to gltf
    if ext == '.glb' and not os.path.exists(filename_gltf):
        glb2gltf(filename_glb)

        # glb = GLTF2().load(filename_glb)
        # root_dir = os.path.dirname(filename_glb)
        # texture_dirname = os.path.splitext(filename_glb)[0]
        # os.makedirs(texture_dirname, exist_ok=True)
        #
        # for image_index, image in enumerate(glb.images):
        #     folder = os.path.join(texture_dirname, f'{image_index}')
        #     os.makedirs(folder, exist_ok=True)
        #     glb.export_image_to_file(image_index, root_dir, override=True)
        #     # image.uri = os.path.join(folder, f'{image_index}.png')
        #     image.uri = os.path.join(root_dir, f'{image_index}.jpg')
        #
        # glb.save_json(filename_gltf)

    assert os.path.exists(filename_gltf)

    # read gltf
    with open(filename_gltf, 'r') as f:
        mesh_dict = json.load(f)

    # add extension to images
    try:
        imgs = mesh_dict['images']
        for i in range(len(imgs)):
            if 'uri' in imgs[i] and len(os.path.splitext(imgs[i]['uri'])[1]) == 0:
                imgs[i]['uri'] = f'{imgs[i]["uri"]}{image_ext}'
            elif 'name' in imgs[i] and 'uri' not in imgs[i]:
                fn = imgs[i]['name']
                imgs[i]['uri'] = f'{fn}{image_ext}'

            # make sure images can be found
            if not os.path.exists(imgs[i]['uri']):
                texture_dirname = os.path.splitext(filename_gltf)[0]
                fn = os.path.join(texture_dirname, imgs[i]['uri'])
                if os.path.exists(fn):
                    imgs[i]['uri'] = fn
    except:
        pass

    # modify pbrMetallicRoughness in materials to use baseColorTexture
    try:
        materials = mesh_dict['materials']
        for i in range(len(materials)):
            emit_dict = materials[i]['emissiveTexture']
            materials[i]['pbrMetallicRoughness']['baseColorTexture'] = emit_dict
    except:
        pass

    with open(filename_gltf, 'w') as f:
        json.dump(mesh_dict, f, indent=2)

    return filename_gltf


def get_img_max_val(img: T.Union[np.ndarray, torch.Tensor]):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    dtype = img.dtype
    if dtype in {np.float32, np.float64, float}:
        max_val = 1.
    else:
        max_val = np.iinfo(dtype).max
    return max_val


def sample_patch(
        arr: torch.Tensor,  # (*b, c, h_in, w_in)
        patch_center: torch.Tensor,  # (*b, 2)   h, w
        patch_width_px: int,
        patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.,  # (*b,)
        patch_height_px: int = None,  # (*b,)
        patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,  # (*b,)
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        format: str = 'chw',  # 'hwc'
):
    """
    Sample from `arr` a patch centered at `center` with a different pixel pitch and number of pixels.

    Args:
        arr:
            (*b, c, h_in, w_in) or (*b, h_in, w_in, c), see `format`. the array to be sampled from.
        patch_center:
            (*b, 2) the center of each patch on arr. u: first dim [0, w_in], v: second dim [0, h_in]
        patch_width_px:
            number of pixels in the patch in width
        patch_width_pitch_scale:
            (*b,) the pitch of the patch (new_pitch / old_pitch)
        patch_height_px:
            if None, the same as `patch_width_px`
        patch_height_pitch_scale:
            if None, the same as `patch_width_pitch_scale`
        format:
            'chw':  arr is (b, c, h, w)
            'hwc':  arr is (b, h, w, c)

    Returns:
        (*b, c, patch_height_px, patch_width_px) or (*b, patch_height_px, patch_width_px, c)

    Note:
        coordinate system:
            The origin of the coordinate is at the top-left corner of `arr`.
            Each pixel in `arr` is 1 unit in width and height.
            The first dimension (u) is toward right and second dimension (v) is toward down.
            The first pixel center is `arr` is at (0.5, 0.5).
        This function should be compared with `uv_sampling`, which uses a different coordinate system.
    """

    if format == 'chw':
        *b_shape, c, h, w = arr.shape
        arr = arr.reshape(-1, c, h, w)  # (b, c, h, w)
    elif format == 'hwc':
        *b_shape, h, w, c = arr.shape
        arr = arr.reshape(-1, h, w, c).permute(0, 3, 1, 2)  # (b, c, h, w)
    else:
        raise NotImplementedError

    b = np.prod(b_shape)
    device = arr.device

    uv = generate_patch_uv(
        patch_center=patch_center,  # (*b, 2)
        patch_width_px=patch_width_px,
        patch_width_pitch_scale=patch_width_pitch_scale,
        patch_height_px=patch_height_px,
        patch_height_pitch_scale=patch_height_pitch_scale,
        device=device,
    )  # (*b, hp, wp, 2)
    uv = uv.reshape(b, uv.size(-3), uv.size(-2), uv.size(-1))  # (b, hp, wp, 2)

    # [0, w] -> [0, 2] -> [-1, 1]
    u = uv[..., 0] * (2 / w) - 1  # (b, hp, wp)
    v = uv[..., 1] * (2 / h) - 1  # (b, hp, wp)
    uv = torch.stack([u, v], dim=-1)  # (b, hp, wp, 2)  [0, w] [0, h]

    # grid_sample
    sampled_patch = torch.nn.functional.grid_sample(
        input=arr,  # (b, c, h, w)
        grid=uv,  # (b, hp, wp, 2)
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )  # (b, c, hp, wp)

    if format == 'chw':
        pass
    elif format == 'hwc':
        sampled_patch = sampled_patch.permute(0, 2, 3, 1)
    else:
        raise NotImplementedError

    sampled_patch = sampled_patch.reshape(*b_shape, sampled_patch.size(1), sampled_patch.size(2), sampled_patch.size(3))
    return sampled_patch


def generate_patch_uv(
        patch_center: torch.Tensor,  # (*b, 2)   h, w
        patch_width_px: int,
        patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.,  # (*b,)
        patch_height_px: int = None,  # (*b,)
        patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,  # (*b,)
        device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Generate uv coordinates ([0, w), [0, h)) of the patches centered at patch_center.

    Args:
        patch_center:
            (*b, 2) the center of each patch on arr. u: first dim [0, w_in], v: second dim [0, h_in]
        patch_width_px:
            number of pixels in the patch in width
        patch_width_pitch_scale:
            (*b,) the pitch of the patch (new_pitch / old_pitch)
        patch_height_px:
            if None, the same as `patch_width_px`
        patch_height_pitch_scale:
            if None, the same as `patch_width_pitch_scale`
        int_only:
            whether the center is always at an integer index

    Returns:
        (*b, patch_height_px, patch_width_px, 2) the u (first dimension) and v (second dimension)
        Note the returned uv can go out of bound.
    """

    *b_shape, _2 = patch_center.shape
    b = np.prod(b_shape)
    patch_center = patch_center.reshape(b, 2)  # (b, 2)
    if isinstance(patch_width_pitch_scale, (int, float)):
        patch_width_pitch_scale = torch.ones(b, dtype=torch.float, device=device) * patch_width_pitch_scale
    if isinstance(patch_width_pitch_scale, torch.Tensor):
        patch_width_pitch_scale = patch_width_pitch_scale.reshape(b).to(device=device)  # (b,)

    if patch_height_px is None:
        patch_height_px = patch_width_px

    if patch_height_pitch_scale is None:
        patch_height_pitch_scale = patch_width_pitch_scale
    if isinstance(patch_height_pitch_scale, (int, float)):
        patch_height_pitch_scale = torch.ones(b, dtype=torch.float, device=device) * patch_height_pitch_scale
    if isinstance(patch_height_pitch_scale, torch.Tensor):
        patch_height_pitch_scale = patch_height_pitch_scale.reshape(b).to(device=device)  # (b,)

    # generate the canonical grid for the patch
    patch_half_width_px = patch_width_px / 2
    patch_half_height_px = patch_height_px / 2

    u, v = torch.meshgrid(
        torch.arange(patch_width_px, dtype=torch.float, device=device),
        torch.arange(patch_height_px, dtype=torch.float, device=device),
        indexing='xy',
    )  # u: (hp, wp), [0, w-1],  v: (hp, wp) [0, h-1]  top-left (0,0)
    u = u + (0.5 - patch_half_width_px)
    v = v + (0.5 - patch_half_height_px)
    # u: (hp, wp), [-0.5, 0, 0.5],  v: (hp, wp)  [-0.5, 0, 0.5]

    # scale and recenter the canonical grid
    u = u.unsqueeze(0).expand(b, -1, -1) * patch_width_pitch_scale.reshape(b, 1, 1) \
        + patch_center[:, 0].reshape(b, 1, 1)  # [0, w]
    v = v.unsqueeze(0).expand(b, -1, -1) * patch_height_pitch_scale.reshape(b, 1, 1) \
        + patch_center[:, 1].reshape(b, 1, 1)  # [0, h]

    uv = torch.stack([u, v], dim=-1)  # (b, hp, wp, 2)  [0, w] [0, h]
    uv = uv.reshape(*b_shape, *(uv.shape[1:]))
    return uv


def sample_random_patch_uv(
        b_shape: T.Union[int, T.List[int]],
        width_px: int,
        height_px: int,
        patch_width_px: int,
        patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.,  # (*b,)
        patch_height_px: int = None,  # (*b,)
        patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,  # (*b,)
        int_only: bool = True,
        device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Samples random uv coordinates ([0, w), [0, h)) to create random patches.

    Args:
        b_shape:
            (*b,) determines the number of patches to sample
        width_px:
            width (in pixel) of the image to sample from, determines the range of u [0,w)
        height_px:
            height (in pixel) of the images to sample from, determines the range of v [0,h)
        patch_width_px:
            number of pixels in the patch in width
        patch_width_pitch_scale:
            (*b,) the pitch of the patch (new_pitch / old_pitch)
        patch_height_px:
            if None, the same as `patch_width_px`
        patch_height_pitch_scale:
            if None, the same as `patch_width_pitch_scale`
        int_only:
            whether the center is always at an integer index

    Returns:
        (*b, patch_height_px, patch_width_px, 2) the u (first dimension) and v (second dimension)

        Note that the patch can go out of bound
    """
    if isinstance(b_shape, int):
        b_shape = [b_shape]

    # randomly sample patch center
    patch_center = torch.rand(*b_shape, 2, device=device)  # (*b, 2)  [0,1)

    patch_center[..., 0] = patch_center[..., 0] * width_px  # (*b, 2)  [0,w) [0,h]
    patch_center[..., 1] = patch_center[..., 1] * height_px

    if int_only:
        # we need to snap to 0.5, 1.5, 2.5, which are the actual pixel center
        patch_center = torch.floor(patch_center) + 0.5

    uv = generate_patch_uv(
        patch_center=patch_center,  # (*b, 2)
        patch_width_px=patch_width_px,
        patch_width_pitch_scale=patch_width_pitch_scale,
        patch_height_px=patch_height_px,
        patch_height_pitch_scale=patch_height_pitch_scale,
        device=device,
    )
    return uv  # (*b, hp, wp, 2)
