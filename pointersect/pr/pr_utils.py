#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# The file implements an interface to call cpp/cuda pr implementations.

import os
import traceback
import typing as T
import warnings
from timeit import default_timer as timer

import torch
from torch._C import *  # otherwise not usable on mac: libc10.dylib' (no such file)
from torch.utils.cpp_extension import load

from . import naive

module_path = os.path.dirname(__file__)

pr_cpp_loaded = False
try:
    print('compiling pr_cpp..')
    pr_cpp = load(
        "pr_cpp",  # must be the same as the pybinf11 module
        sources=[
            os.path.join(module_path, "cpp/pr.cpp"),
        ],
        verbose=True,
    )
    pr_cpp_loaded = True
except:
    warnings.warn(f'pr_cpp compilation failed')
    print(traceback.format_exc())
    pr_cpp = None


pr_cuda_loaded = False
if torch.cuda.is_available():
    try:
        print('compiling pr_cuda..')
        pr_cuda = load(
            "pr_cuda",  # must be the same as the pybinf11 module
            sources=[
                os.path.join(module_path, "cuda/pr_cuda.cpp"),
                os.path.join(module_path, "cuda/pr_cuda_kernel.cu"),
            ],
            verbose=True,
            # is_python_module=False,
            # is_standalone=True,
        )
        print('finished compiling pr_cuda..', flush=True)
        pr_cuda_loaded = True
    except:
        warnings.warn(f'pr_cuda compilation failed')
        print(traceback.format_exc())
        pr_cuda = None


def find_neighbor_points_of_rays(
        points: torch.Tensor,  # (b, n, 3)
        ray_origins: torch.Tensor,  # (b, m, 3)
        ray_directions: torch.Tensor,  # (b, m, 3)
        ray_radius: T.Union[torch.Tensor, float],  # (b,)
        grid_size: T.Union[torch.Tensor, int],  # (b, 3)
        grid_center: T.Union[torch.Tensor, float, None] = None,  # (b, 3)
        grid_width: T.Union[torch.Tensor, float, None] = None,  # (b, 3)
) -> T.List[T.List[T.List[int]]]:
    """
    Find all the points within ray_radius of a ray.

    Args:
        points:
            (b, n, 3)  the xyz_w of points in world coord
        ray_origins:
            (b, m, 3)  the ray origin in the world coord
        ray_directions:
            (b, m, 3) the ray direction in the world coord
        ray_radius:
            (b,) the radius of each ray
        grid_size:
            (b, 3)  the number of grid points in xyz direction
        grid_center:
            (b, 3)  the center of the grid.
            If None: average of points
        grid_width:
            (b, 3)  the full width of the grid.
            If None: (max-min) from points

    Returns:
        list of list of list:  b -> m -> n_idx
        n_idx is the index of the points
    """

    batch_size = points.size(0)
    n_rays = ray_origins.size(-2)

    if isinstance(ray_radius, (float, int)):
        # ray radius should have only one dimension (batch)
        # ray_radius = torch.ones(batch_size, 3, dtype=points.dtype, device=points.device) * ray_radius
        ray_radius = torch.ones(batch_size, dtype=points.dtype, device=points.device) * ray_radius

    if isinstance(grid_size, int):
        grid_size = torch.ones(batch_size, 3, dtype=torch.long, device=points.device) * grid_size

    if grid_center is None:
        grid_center = torch.mean(points, dim=-2)  # (b, 3)
    elif isinstance(grid_center, (float, int)):
        grid_center = torch.ones(batch_size, 3, dtype=points.dtype, device=points.device) * grid_center

    if grid_width is None:
        grid_width = torch.max(points, dim=-2)[0] - torch.min(points, dim=-2)[0]  # (b, 3)
    elif isinstance(grid_width, (float, int)):
        grid_width = torch.ones(batch_size, 3, dtype=points.dtype, device=points.device) * grid_width

    t_min = 0.
    t_max = 1.e10

    printout = False
    stime = timer()

    if points.is_cuda and pr_cuda_loaded:
        all_ray2pidxs, all_ray_start_idxs, all_ray_end_idxs = pr_cuda.find_neighbor_points_of_rays(
            points,
            ray_origins,
            ray_directions,
            ray_radius,
            grid_size,
            grid_center,
            grid_width,
            t_min,
            t_max,
        )

        neighbor_time = timer()

        if printout:
            print(f'neighbor_time={neighbor_time - stime:.4f} secs', flush=True)

        all_ray2pidxs = all_ray2pidxs.detach().cpu()
        all_ray_start_idxs = all_ray_start_idxs.detach().cpu()
        all_ray_end_idxs = all_ray_end_idxs.detach().cpu()

        move_time = timer()
        if printout:
            print(f'move_time={move_time - neighbor_time:.4f} secs', flush=True)

        # n_idxs of a ray is found via ray2pidx[b][ray_start_idx[m]:ray_end_idx[m]]
        out = []
        for b in range(batch_size):
            ray_out = []
            ray2pidxs = all_ray2pidxs[b]
            ray_start_idxs = all_ray_start_idxs[b]
            ray_end_idxs = all_ray_end_idxs[b]

            for m in range(n_rays):
                nidxs = ray2pidxs[ray_start_idxs[m]:ray_end_idxs[m]]  # (n,)
                # nidxs = all_ray2pidxs[b, all_ray_start_idxs[b,m]:all_ray_end_idxs[b,m]]  # (n,)
                ray_out.append(nidxs)
            out.append(ray_out)

        process_time = timer()
        if printout:
            print(f'process_time={process_time - move_time:.4f} s ecs', flush=True)


    elif pr_cpp_loaded:
        out = pr_cpp.find_neighbor_points_of_rays(
            points,
            ray_origins,
            ray_directions,
            ray_radius,
            grid_size,
            grid_center,
            grid_width,
            t_min,
            t_max,
            "v2",
        )
    else:
        warnings.warn('using brute force method')
        out = naive.find_neighbor_points_of_rays_brute_force(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_center=grid_center,
            grid_width=grid_width,
        )

    return out


def find_k_neighbor_points_of_rays(
        points: torch.Tensor,  # (b, n, 3)
        k: int,
        ray_origins: torch.Tensor,  # (b, m, 3)
        ray_directions: torch.Tensor,  # (b, m, 3)
        ray_radius: T.Union[torch.Tensor, float],  # (b,)
        grid_size: T.Union[torch.Tensor, int],  # (b, 3)
        grid_center: T.Union[torch.Tensor, float, None] = None,  # (b, 3)
        grid_width: T.Union[torch.Tensor, float, None] = None,  # (b, 3)
        gidx2pidx_bank: T.Union[torch.Tensor, None] = None,  # (b, n)
        gidx_start_idx: T.Union[torch.Tensor, None] = None,  # (b, max_n_cell+1)
        refresh_cache: bool = True,
        valid_mask: T.Optional[torch.Tensor] = None,  # (b, n, 1) or (b, n)
        version: str = 'v4',
) -> T.Dict[str, torch.Tensor]:
    """
    Find all the points within ray_radius of a ray.

    Args:
        points:
            (b, n, 3)  the xyz_w of points in world coord
        ray_origins:
            (b, m, 3)  the ray origin in the world coord
        ray_directions:
            (b, m, 3) the ray direction in the world coord
        ray_radius:
            (b,) the radius of each ray
            if None: 0.1 * grid_width
        grid_size:
            (b, 3)  the number of grid points in xyz direction
            if None: 100
        grid_center:
            (b, 3)  the center of the grid.
            If None: average of points
        grid_width:
            (b, 3)  the full width of the grid.
            If None: (max-min) from points
        gidx2pidx_bank:
            (b, n) long, the cache from the cuda kernel
        gidx_start_idx:
            (b, max_n_cell+1) int32, the cache from the cuda kernel
        refresh_cache:
            if True, the cuda kernel will compute new gidx2pidx_bank and gidx_start_idx
            else it will reuse the caches.
        version:
            'v1': use the original implementation (grid_ray_intersection -> collect k points)
            'v2': traverse and collect concurrently (use less memory)
            'v3': use less memory + reuse grid_cell_caches

    Returns:
        list of list of list:  b -> m -> n_idx
        n_idx is the index of the points
    """
    # smemory = torch.cuda.memory_allocated(device=points.device)

    batch_size = points.size(0)
    n_points = points.size(-2)
    n_rays = ray_origins.size(-2)
    device = points.device

    if valid_mask is None:
        valid_mask = torch.ones(batch_size, n_points, dtype=torch.bool, device=device)
        valid_mask[:, 0] = 0  # assume index 0 is point at inf

    if valid_mask.ndim == 3:
        valid_mask = valid_mask.squeeze(-1)  # (b, n)
    valid_mask = valid_mask.to(dtype=torch.bool, device=device)

    # initialize return dict
    # all_sorted_dists = torch.ones(batch_size, n_rays, k, device = device) * torch.inf
    # all_sorted_idxs = torch.zeros(batch_size, n_rays, k, device = device, dtype=torch.long)
    # all_sorted_ts = torch.ones(batch_size, n_rays, k, device = device) * -1

    if grid_size is None:
        grid_size = 100

    if isinstance(grid_size, int):
        grid_size = torch.ones(batch_size, 3, dtype=torch.long, device=device) * grid_size

    if grid_center is None:
        if valid_mask is not None:
            grid_center = (
                                  torch.max(points.masked_fill(~valid_mask.unsqueeze(-1), -torch.inf), dim=-2)[0] +
                                  torch.min(points.masked_fill(~valid_mask.unsqueeze(-1), torch.inf), dim=-2)[0]
                          ) / 2  # (b, 3)
        else:
            grid_center = (
                                  torch.max(points.masked_fill(points > 1e6, -torch.inf), dim=-2)[0] +
                                  torch.min(points.masked_fill(points > 1e6, torch.inf), dim=-2)[0]
                          ) / 2  # (b, 3)
    elif isinstance(grid_center, (float, int)):
        grid_center = torch.ones(batch_size, 3, dtype=points.dtype, device=device) * grid_center

    if grid_width is None:
        if valid_mask is not None:
            grid_width = torch.max(points.masked_fill(~valid_mask.unsqueeze(-1), -torch.inf), dim=-2)[0] - \
                         torch.min(points.masked_fill(~valid_mask.unsqueeze(-1), torch.inf), dim=-2)[0]  # (b, 3)
        else:
            grid_width = torch.max(points.masked_fill(points > 1e6, -torch.inf), dim=-2)[0] - \
                         torch.min(points.masked_fill(points > 1e6, torch.inf), dim=-2)[0]  # (b, 3)
    elif isinstance(grid_width, (float, int)):
        grid_width = torch.ones(batch_size, 3, dtype=points.dtype, device=device) * grid_width

    if ray_radius is None or ray_radius < 0:
        ray_radius = 0.1 * grid_width.mean(dim=-1)  # (b,)

    if isinstance(ray_radius, (float, int)):
        # ray radius should have only one dimension (batch)
        # ray_radius = torch.ones(batch_size, 3, dtype=points.dtype, device=points.device) * ray_radius
        ray_radius = torch.ones(batch_size, dtype=points.dtype, device=device) * ray_radius

    grid_size = grid_size.to(device=device)
    grid_center = grid_center.to(device=device)
    grid_width = grid_width.to(device=device)
    ray_radius = ray_radius.to(device=device)

    t_min = 0.
    t_max = 1.e10

    printout = False
    stime = timer()

    out = []

    if points.is_cuda and pr_cuda_loaded:
        if version == 'v1':
            ray2pidx_heap, ray_neighbor_num = pr_cuda.find_k_neighbor_points_of_rays(
                points,
                ray_origins,
                ray_directions,
                ray_radius,
                grid_size,
                grid_center,
                grid_width,
                k,
                t_min,
                t_max,
            )
            # total_memory_ours = torch.cuda.memory_allocated(device=device) - smemory
            # print(f' ours use memory : {total_memory_ours  / 2 ** 30} GB')
            return dict(
                ray2pidx_heap=ray2pidx_heap,  # (b, m, k)
                ray_neighbor_num=ray_neighbor_num,  # (b, m)
            )
        elif version == 'v2':
            ray2pidx_heap, ray_neighbor_num = pr_cuda.find_k_neighbor_points_of_rays_v2(
                points,
                ray_origins,
                ray_directions,
                ray_radius,
                grid_size,
                grid_center,
                grid_width,
                k,
                t_min,
                t_max,
            )
            # total_memory_ours = torch.cuda.memory_allocated(device=device) - smemory
            # print(f' ours use memory : {total_memory_ours  / 2 ** 30} GB')
            return dict(
                ray2pidx_heap=ray2pidx_heap,  # (b, m, k)
                ray_neighbor_num=ray_neighbor_num,  # (b, m)
            )
        elif version == 'v3':
            if gidx2pidx_bank is None or gidx_start_idx is None or \
                    gidx2pidx_bank.numel() == 0 or gidx_start_idx.numel() == 0:
                gidx2pidx_bank = torch.empty(0, device=points.device, dtype=torch.long)
                gidx_start_idx = torch.empty(0, device=points.device, dtype=torch.int32)
                refresh_cache = True

            # # debug
            # print(f'refresh_cache = {refresh_cache}')
            # print(f'gidx2pidx_bank.shape = {gidx2pidx_bank.shape}')
            # print(f'gidx_start_idx.shape = {gidx_start_idx.shape}')

            (
                ray2pidx_heap,
                ray_neighbor_num,
                ray2dist_heap,
                gidx2pidx_bank,
                gidx_start_idx,
            ) = pr_cuda.find_k_neighbor_points_of_rays_v3(
                points,
                ray_origins,
                ray_directions,
                ray_radius,
                grid_size,
                grid_center,
                grid_width,
                gidx2pidx_bank,
                gidx_start_idx,
                k,
                t_min,
                t_max,
                refresh_cache,
            )
            # total_memory_ours = torch.cuda.memory_allocated(device=device) - smemory
            # print(f' ours use memory : {total_memory_ours  / 2 ** 30} GB')
            return dict(
                ray2pidx_heap=ray2pidx_heap,  # (b, m, k)
                ray_neighbor_num=ray_neighbor_num,  # (b, m)
                ray2dist_heap=ray2pidx_heap,  # (b, m, k)
                gidx2pidx_bank=gidx2pidx_bank,  # (b, n)
                gidx_start_idx=gidx_start_idx,  # (b, max_n_cell+1)
            )
        elif version == 'v4':
            if gidx2pidx_bank is None or gidx_start_idx is None or \
                    gidx2pidx_bank.numel() == 0 or gidx_start_idx.numel() == 0:
                gidx2pidx_bank = torch.empty(0, device=points.device, dtype=torch.long)
                gidx_start_idx = torch.empty(0, device=points.device, dtype=torch.int32)
                refresh_cache = True

            # if valid_mask is None:
            #     valid_mask = torch.ones(batch_size, n_points, dtype=torch.bool, device=device)
            #     valid_mask[:, 0] = 0  # assume index 0 is point at inf
            #
            # if valid_mask.ndim == 3:
            #     valid_mask = valid_mask.squeeze(-1)  # (b, n)
            # valid_mask = valid_mask.to(dtype=torch.bool, device=device)

            # # debug
            # print(f'refresh_cache = {refresh_cache}')
            # print(f'gidx2pidx_bank.shape = {gidx2pidx_bank.shape}')
            # print(f'gidx_start_idx.shape = {gidx_start_idx.shape}')

            (
                ray2pidx_heap,
                ray_neighbor_num,
                ray2dist_heap,
                gidx2pidx_bank,
                gidx_start_idx,
            ) = pr_cuda.find_k_neighbor_points_of_rays_v4(
                points,
                ray_origins,
                ray_directions,
                ray_radius,
                grid_size,
                grid_center,
                grid_width,
                gidx2pidx_bank,
                gidx_start_idx,
                valid_mask,
                k,
                t_min,
                t_max,
                refresh_cache,
            )
            # total_memory_ours = torch.cuda.memory_allocated(device=device) - smemory
            # print(f' ours use memory : {total_memory_ours  / 2 ** 30} GB')
            return dict(
                ray2pidx_heap=ray2pidx_heap,  # (b, m, k)
                ray_neighbor_num=ray_neighbor_num,  # (b, m)
                ray2dist_heap=ray2pidx_heap,  # (b, m, k)
                gidx2pidx_bank=gidx2pidx_bank,  # (b, n)
                gidx_start_idx=gidx_start_idx,  # (b, max_n_cell+1)
            )
        else:
            raise NotImplementedError

    else:
        return dict()
