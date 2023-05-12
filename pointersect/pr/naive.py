#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# This file implements an all-python version of point ray intersection algorithm.

import typing as T
from timeit import default_timer as timer

import torch

from plib import utils


def sub2ind(
        idx: torch.Tensor,  # (*, n, 3)
        size: torch.Tensor,  # (*, 3)
) -> torch.Tensor:
    """
    Given x y z index, change to the linear index.  (matlab's sub2ind)

    Args:
        idx:
            (*, n, 3) long
        size:
            (*, 3) long

    Returns:
        (*, n) long
    """
    # linear_idx = idx[..., 0] + \
    #              idx[..., 1] * size[..., 0:1] + \
    #              idx[..., 2] * (size[..., 0:1] * size[..., 1:2])  # (*, n)

    linear_idx = idx[..., 2] + \
                 idx[..., 1] * size[..., 2:3] + \
                 idx[..., 0] * (size[..., 1:2] * size[..., 2:3])  # (*, n)

    return linear_idx


def ind2sub(
        ind: torch.Tensor,  # (*, n)
        size: torch.Tensor,  # (*, 3)
) -> torch.Tensor:
    """
    Given linear index, return (i,j,k)

    Args:
        ind:
            # (*, n) long,  x + y * sx + z * sx * sy
            (*, n) long,  z + y * sz + x * sy * sz
        size:
            (*, 3) long

    Returns:
        (*, n, 3) long
    """

    # xy_size = size[..., 0:1] * size[..., 1:2]  # (*, 1)
    # zs = torch.div(ind, xy_size, rounding_mode='floor')  # (*, n)
    # ind = ind - zs * xy_size  # (*, n)
    # ys = torch.div(ind, size[..., 0:1], rounding_mode='floor')  # (*, n)
    # xs = ind - ys * size[..., 0:1]  # (*, n)

    yz_size = size[..., 1:2] * size[..., 2:3]  # (*, 1)
    xs = torch.div(ind, yz_size, rounding_mode='floor')  # (*, n)
    ind = ind - xs * yz_size  # (*, n)
    ys = torch.div(ind, size[..., 2:3], rounding_mode='floor')  # (*, n)
    zs = ind - ys * size[..., 2:3]  # (*, n)

    idx = torch.stack((xs, ys, zs), dim=-1)  # (*, n, 3)
    return idx


def get_grid_idx(
        points: torch.Tensor,  # (*, n, 3)
        grid_size: T.Union[torch.Tensor, int],  # (*, 3)
        center: T.Union[torch.Tensor, float] = 0.,  # (*, 3)
        grid_width: T.Union[torch.Tensor, float] = 1.,  # (*, 3)
        mode: str = 'ind',
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the grid index given xyz_w.
    Args:
        points:
            (*, n, 3)
        grid_size:
            (*, 3) long. number of grid cells in x y z.
        center:
            (*, 3)  center of the grid
        grid_width:
            (*, 3)  length (full width) of the grid in xyz
        include_all:
            whether to include any points outside the grid
            If True, all out-of-bound points will be assigned to -1 or (-1, -1, -1)
        mode:
            'subidx': return sub idx
            'ind': return linear index

    Returns:
        grid_idx:
            if mode == 'subidx':  (*, n, 3) long
            elif mode == 'ind':   (*, n) long
        valid_mask:
            (*, n) bool
    Algorithm:
        Let
            x_from = center_x - grid_length_x / 2
            x_to = center_x + grid_length_x / 2
            cell_width_x = grid_length / grid_size_x

        We divide x = [x_from, x_to] into grid_size cells, each cell is of width x_cell.

            x_idx = ((x - x_from) / cell_width_x).floor().clamp(0, grid_size_x-1)
            y_idx = ((y - y_from) / cell_width_y).floor().clamp(0, grid_size_y-1)
            z_idx = ((z - z_from) / cell_width_z).floor().clamp(0, grid_size_z-1)

            grid_idx = x_idx + y_dix * grid_size_x + z_idx * (grid_size_x * grid_size_y)
    """

    if isinstance(center, float):
        center = torch.tensor(center, dtype=points.dtype, device=points.device)
        center = center.view(*([1] * (points.ndim - 1))).expand(*([1] * (points.ndim - 2) + [3]))
    if isinstance(grid_size, (float, int)):
        grid_size = torch.tensor(grid_size, dtype=torch.long, device=points.device)
        grid_size = grid_size.view(*([1] * (points.ndim - 1))).expand(*([1] * (points.ndim - 2) + [3]))
    if isinstance(grid_width, (float, int)):
        grid_width = torch.tensor(grid_width, dtype=points.dtype, device=points.device)
        grid_width = grid_width.view(*([1] * (points.ndim - 1))).expand(*([1] * (points.ndim - 2) + [3]))

    grid_size = grid_size.to(dtype=torch.long, device=points.device)
    center = center.to(device=points.device)
    grid_width = grid_width.to(device=points.device)

    grid_from = (center - grid_width / 2).unsqueeze(-2)  # (*, 1, 3)
    grid_to = (center + grid_width / 2).unsqueeze(-2)  # (*, 1, 3)
    cell_width = (grid_width / grid_size).unsqueeze(-2)  # (*, 1, 3)
    p_idx = ((points - grid_from) / cell_width).floor().long()  # (*, n, 3), sub_idx on the grid

    # we will mark any out-of-bound points invalid
    valid_mask = torch.logical_and(
        points >= grid_from,
        points <= grid_to,
    ).all(dim=-1)  # (*, n),  bool

    if mode == 'ind':
        grid_idx = sub2ind(
            idx=p_idx,
            size=grid_size,
        )  # (*, n) long
    elif mode == 'subidx':
        grid_idx = p_idx  # (*, n, 3)
    else:
        raise NotImplementedError

    return grid_idx, valid_mask


def gather_points(
        grid_idxs: torch.Tensor,  # (b, n), long
        total_cells: torch.Tensor,  # (b,) long
        valid_mask: torch.Tensor = None,  # (b, n), bool
) -> T.List[T.List[T.List[int]]]:
    """
    gather the points belonging to each grid cell.

    Args:
        grid_idxs:
            grid index of each point
        total_cells:
            total grid cells
        valid_mask:
            whether to record the point

    Returns:
        list of list of list, (b, total_cells[bidx]) -> n_idx of points in the cell
    """
    batch_size, n_points = grid_idxs.shape
    grid_idxs = grid_idxs.long()
    total_cells = total_cells.long()

    all_cell2pidx = []
    for b in range(batch_size):
        cell2pidx: T.List[T.List[int]] = [[] for _ in range(total_cells[b])]
        for i in range(n_points):
            if valid_mask is None or valid_mask[b, i]:
                cell2pidx[grid_idxs[b, i]].append(i)
        all_cell2pidx.append(cell2pidx)
    return all_cell2pidx


def grid_ray_intersection(
        ray_origins: torch.Tensor,  # (b, m, 3)
        ray_directions: torch.Tensor,  # (b, m, 3)
        ray_radius: T.Union[torch.Tensor, float],  # (b, )
        grid_size: torch.Tensor,  # (b, 3)
        grid_center: torch.Tensor,  # (b, 3) long
        grid_width: torch.Tensor,  # (b, 3) long
):
    """
    Compute the intersection between grid cells and the ray.

    Args:
        ray_origins:
            (b, m, 3)
        ray_directions:
            (b, m, 3)
        ray_radius:
            (b,) or float
        grid_size:
            (b, 3) long, xyz
        grid_center:
            (b, 3), xyz
        grid_width:
            (b, 3), xyz

    Returns:
        list of list of list:  b -> ray_idx -> grid_idx (including -1, outside the grid)

    Algorithm:
        for each ray (px, py, pz, dx, dy, dz)
            if dx.abs() < 1e-8:
                # use y=c plane
            else:
                # use x=c plane

            - find plane-ray intersection point on each grid plane (xi, yi, zi)
            - for each (xi, yi, zi)
                x_from = discretize(xi - radius)
                x_to = discretize(xi + radius)
                (same for y and z)

                find grid idx for all of combinations, ie, get all grid cells
                surrounding the point.
    """

    batch_size, n_rays, _ = ray_origins.shape
    device = ray_origins.device

    if isinstance(ray_radius, float):
        ray_radius = torch.ones(batch_size, device=device) * ray_radius  # (b,)

    grid_from = (grid_center - grid_width / 2)  # (b, 3)
    cell_width = (grid_width / grid_size)  # (b, 3)
    total_cells = torch.prod(grid_size, dim=-1)  # (b,)

    # determine whether to intersect with x=c plane, y=c plane, or z=c plane.
    # It is important to select a direction so that the intersection point on two nearby planes
    # do not exceed one grid in one of the rest of the two directions.
    # Fortunately, it can be shown that it will always happen when we select wisely.

    # Let inv_tx = |dx / wx|  (1 / time to travel one x grid)
    #     inv_ty = |dy / wy|  (1 / time to travel one y grid)
    #     inv_tz = |dz / wz|  (1 / time to travel one z grid)
    # If inv_tx is the largest of (inv_tx, inv_ty, inv_tz) -> we will intersect with x=c planes.
    # vise versa for inv_ty and inv_tz.
    # When we select inv_tx, it ensures we move in the x direction in 1 grid, y and z in <= 1 grid.

    inv_t = (ray_directions / cell_width.unsqueeze(-2)).abs()  # (b, m, 3)
    idx_to_use = torch.argmax(inv_t, dim=-1)  # (b, m)  (0, 1, 2)

    all_grid_idxs = []
    for b in range(batch_size):

        # build a meshgrid for local grid idx.  This will determine the neighboring grids to gather.
        # the local grid width = 2 * local_grid_radius
        local_grid_radius = (torch.ceil(ray_radius[b, None] / cell_width[b] + 1)).long()  # (3,) long,  xyz
        grid_idx_from = -1 * local_grid_radius  # (3,) xyz
        grid_idx_to = local_grid_radius  # (3,) xyz
        xs = torch.arange(grid_idx_from[0], grid_idx_to[0] + 1, dtype=torch.long, device=device)
        ys = torch.arange(grid_idx_from[1], grid_idx_to[1] + 1, dtype=torch.long, device=device)
        zs = torch.arange(grid_idx_from[2], grid_idx_to[2] + 1, dtype=torch.long, device=device)
        Xs, Ys, Zs = torch.meshgrid(xs, ys, zs, indexing='ij')
        local_grid_subidx = torch.cat(
            (
                Xs.reshape(-1, 1),
                Ys.reshape(-1, 1),
                Zs.reshape(-1, 1),
            ), dim=-1)  # (n_local_grid, 3)  xyz in grid

        # since the local grid has a certain width, we need to intersect with more planes
        xplane_idxs = torch.arange(
            -1 * local_grid_radius[0],
            grid_size[b, 0] + local_grid_radius[0],
            dtype=ray_origins.dtype,
            device=device,
        )  # (num_xplanes+2r,),  -r, -r+1, ..., -1, | 0, 1, ..., n_xplanes-1 |, n_xplanes, ..., n_xplanes+r-1
        yplane_idxs = torch.arange(
            -1 * local_grid_radius[1],
            grid_size[b, 1] + local_grid_radius[1],
            dtype=ray_origins.dtype,
            device=device,
        )  # (num_yplanes+2r,)
        zplane_idxs = torch.arange(
            -1 * local_grid_radius[2],
            grid_size[b, 2] + local_grid_radius[2],
            dtype=ray_origins.dtype,
            device=device,
        )  # (num_zplanes+2r,)
        xplane_cs = grid_from[b, 0] + (xplane_idxs + 0.5) * cell_width[b, 0]  # (num_xplanes+2r,)
        yplane_cs = grid_from[b, 1] + (yplane_idxs + 0.5) * cell_width[b, 1]  # (num_yplanes+2r,)
        zplane_cs = grid_from[b, 2] + (zplane_idxs + 0.5) * cell_width[b, 2]  # (num_zplanes+2r,)
        xplane_ts = (xplane_cs.unsqueeze(0) - ray_origins[b, :, 0:1]) / ray_directions[b, :, 0:1]  # (m, num_xplanes+2r)
        yplane_ts = (yplane_cs.unsqueeze(0) - ray_origins[b, :, 1:2]) / ray_directions[b, :, 1:2]  # (m, num_yplanes+2r)
        zplane_ts = (zplane_cs.unsqueeze(0) - ray_origins[b, :, 2:3]) / ray_directions[b, :, 2:3]  # (m, num_zplanes+2r)

        # compute ray-plane intersection
        grid_idxs = []
        for m in range(n_rays):

            if idx_to_use[b, m] == 0:
                ts = xplane_ts[m]  # (num_xplanes+2rx, )
            elif idx_to_use[b, m] == 1:
                ts = yplane_ts[m]  # (num_yplanes+2ry, )
            elif idx_to_use[b, m] == 2:
                ts = zplane_ts[m]  # (num_zplanes+2rz, )
            else:
                raise RuntimeError(f"{idx_to_use[b, m]}")

            # for each intersection point, gather all neighboring cells
            ps = ray_origins[b, m].unsqueeze(0) + ts.unsqueeze(-1) * ray_directions[b, m].unsqueeze(
                0)  # (num_planes+2r, 3)
            ps_grid_ind, _ = get_grid_idx(
                points=ps,  # (n_plane+4r, 3)
                grid_size=grid_size[b],  # (3,)
                center=grid_center[b],  # (3,)
                grid_width=grid_width[b],  # (3,)
                # include_outside=True,  # need to be true to handle boundary
                mode='subidx',
            )  # (n_plane+2r, 3),  ps_grid_ind can < 0 or >= grid_size

            # get all neighboring cells (some will have invalid grid_idx)
            gidxs = ps_grid_ind.unsqueeze(1) + local_grid_subidx.unsqueeze(0)  # (n_plane+2r, n_local_grid, 3) subidx

            # handle invalid gidxs
            valid_mask = torch.logical_and(
                gidxs >= 0,
                gidxs < grid_size[b].reshape(1, 1, 3),
            ).all(dim=-1)  # (n_plane+2r, n_local_grid)
            gidxs = gidxs[valid_mask]

            gidxs = sub2ind(
                idx=gidxs.view(-1, 3),  # (n_plane*n_local_grid, 3)
                size=grid_size[b],  # (3,)
            )  # (n_cells,)
            gidxs = gidxs.unique().detach().cpu().tolist()  # (n_cells,)
            grid_idxs.append(gidxs)

        all_grid_idxs.append(grid_idxs)
    return all_grid_idxs


def find_neighbor_points_of_rays(
        points: torch.Tensor,  # (b, n, 3)
        ray_origins: torch.Tensor,  # (b, m, 3)
        ray_directions: torch.Tensor,  # (b, m, 3)
        ray_radius: T.Union[torch.Tensor, float],  # (b,)
        grid_size: T.Union[torch.Tensor, int],  # (b, 3)
        grid_center: T.Union[torch.Tensor, float, None] = 0.,  # (b, 3)
        grid_width: T.Union[torch.Tensor, float, None] = 1.,  # (b, 3)
        print_out: bool = True,
) -> T.List[T.List[T.List[int]]]:
    """
    Find the points within `radius` of a ray, ie, the vertical distance from the point to ray <= radius.

    Returns:
        list of list of list:  b -> m -> n_idx

    Note:
        Our algorithm is very simple. In order to be parallelized on gpu easily, we want to every thread to have
        as few branching conditions (if/else) as possible.

        We will:
        - parallelize on ray
        - do the same thing for all rays.
        - ignore all points outside the grid boundary

    Algorithm:

        1. determine a grid
        2. discretize the xyz of points -> calculate grid indices of each point
        3. for each grid cell, gather points belonging to the cell
        4. for each ray (px, py, pz, dx, dy, dz)
                if dy.abs() < 1e-8:
                    # use x=c plane
                else:
                    # use y=c plane

                - find plane-ray intersection point on each grid plane (xi, yi, zi)
                - for each (xi, yi, zi)
                    x_from = discretize(xi - radius)
                    x_to = discretize(xi + radius)
                    (same for y and z)

                    find grid idx for all of combinations, ie, get all grid cells
                    surrounding the point.

        5. given grid idxs for each ray, gather point idxs


    """

    batch_size, n_rays, _ = ray_origins.shape

    # determine a grid
    if grid_center is None:
        grid_center = points.mean(dim=-2)  # (b, 3)

    if grid_width is None:
        grid_width = points.max(dim=-2) - points.min(dim=-1)  # (b, 3)

    if isinstance(grid_center, float):
        grid_center = torch.tensor(grid_center, dtype=points.dtype, device=points.device)
        grid_center = grid_center.view(1, 1).expand(batch_size, 3)  # (b, 3)
    if isinstance(grid_width, (float, int)):
        grid_width = torch.tensor(grid_width, dtype=points.dtype, device=points.device)
        grid_width = grid_width.view(1, 1).expand(batch_size, 3)  # (b, 3)
    if isinstance(grid_size, (float, int)):
        grid_size = torch.tensor(grid_size, dtype=torch.long, device=points.device)
        grid_size = grid_size.view(1, 1).expand(batch_size, 3)  # (b, 3)
    grid_size = grid_size.long()  # (b, 3)
    total_cells = torch.prod(grid_size, dim=-1)  # (b,)

    # get grid idx of each point
    stime = timer()
    total_stime = timer()
    grid_idxs, valid_mask = get_grid_idx(
        points=points,  # (b, n, 3)
        grid_size=grid_size,
        center=grid_center,
        grid_width=grid_width,
        # include_outside=include_outside,
        mode='ind',
    )  # (b, n)  [0, total_cells-1]
    time_grid_idx = timer() - stime

    # gather points belonging to each cell
    stime = timer()
    all_cell2pidx = gather_points(
        grid_idxs=grid_idxs,  # (b, n)
        total_cells=total_cells,  # (b,)
        valid_mask=valid_mask,  # (b, n)
    )  # b -> total_cells[b] -> point_idxs
    time_gather_cell = timer() - stime

    # gather the cells intersected by the ray
    stime = timer()
    all_ray2gidxs = grid_ray_intersection(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        ray_radius=ray_radius,
        grid_size=grid_size,
        grid_center=grid_center,
        grid_width=grid_width,
        # include_outside=include_outside,
    )  # b -> ray_idx -> grid_idx (including -1)
    time_intersect = timer() - stime

    # gather points in the passing cells
    stime = timer()
    all_ray2pidxs = []  # b -> ray_idx -> point_idx
    for b in range(batch_size):
        ray2gidxs = all_ray2gidxs[b]
        ray2pidxs = []
        for m in range(n_rays):
            gidxs = ray2gidxs[m]
            pidxs = []
            for gidx in gidxs:
                assert gidx >= 0 and gidx < total_cells[b], f'{gidx}, {total_cells[b]}'
                pidxs += all_cell2pidx[b][gidx]

            # refine the points (remove point-ray distance > ray_radius)
            if len(pidxs) > 0:
                dist_dict = utils.compute_point_ray_distance(
                    points=points[b, pidxs],  # (n_point, 3)
                    ray_origins=ray_origins[b, m:m + 1],  # (1, 3)
                    ray_directions=ray_directions[b, m:m + 1],  # (1,3)
                )
                dists = dist_dict['dists'].squeeze(0)  # (n_points,)
                ts = dist_dict['ts'].squeeze(0)  # (n_points,)
                pidxs = torch.tensor(pidxs)
                mask = torch.logical_and(
                    dists <= ray_radius[b],
                    ts >= 0,
                )
                pidxs = pidxs[mask].tolist()

            ray2pidxs.append(pidxs)
        all_ray2pidxs.append(ray2pidxs)
    time_gather_result = timer() - stime
    time_total = timer() - total_stime

    if print_out:
        print(f"total uses {time_total:.1f} secs")
        print(f"    griding uses {time_grid_idx:.1f} secs = {time_grid_idx / time_total * 100.:.2f}%")
        print(f"    celling uses {time_gather_cell:.1f} secs = {time_gather_cell / time_total * 100.:.2f}%")
        print(f"    intersect uses {time_intersect:.1f} secs = {time_intersect / time_total * 100.:.2f}%")
        print(f"    gather uses {time_gather_result:.1f} secs = {time_gather_result / time_total * 100.:.2f}%")
        print("")
    return all_ray2pidxs


def find_neighbor_points_of_rays_brute_force(
        points: torch.Tensor,  # (b, n, 3)
        ray_origins: torch.Tensor,  # (b, m, 3)
        ray_directions: torch.Tensor,  # (b, m, 3)
        ray_radius: T.Union[torch.Tensor, float],  # (b,)
        grid_size: T.Union[torch.Tensor, int],  # (b, 3)
        grid_center: T.Union[torch.Tensor, float, None] = 0.,  # (b, 3)
        grid_width: T.Union[torch.Tensor, float, None] = 1.,  # (b, 3)
) -> T.List[T.List[T.List[int]]]:
    batch_size, n_rays, _ = ray_origins.shape

    # determine a grid
    if grid_center is None:
        grid_center = points.mean(dim=-2)  # (b, 3)

    if grid_width is None:
        grid_width = points.max(dim=-2) - points.min(dim=-1)  # (b, 3)

    if isinstance(grid_center, float):
        grid_center = torch.tensor(grid_center, dtype=points.dtype, device=points.device)
        grid_center = grid_center.view(1, 1).expand(batch_size, 3)  # (b, 3)
    if isinstance(grid_width, (float, int)):
        grid_width = torch.tensor(grid_width, dtype=points.dtype, device=points.device)
        grid_width = grid_width.view(1, 1).expand(batch_size, 3)  # (b, 3)
    if isinstance(grid_size, (float, int)):
        grid_size = torch.tensor(grid_size, dtype=torch.long, device=points.device)
        grid_size = grid_size.view(1, 1).expand(batch_size, 3)  # (b, 3)
    grid_size = grid_size.long()  # (b, 3)
    total_cells = torch.prod(grid_size, dim=-1)  # (b,)

    # handle outside points
    grid_from = grid_center - grid_width / 2  # (b, 3)
    grid_to = grid_center + grid_width / 2  # (b, 3)
    valid_mask = torch.logical_and(
        points >= grid_from.unsqueeze(1),
        points <= grid_to.unsqueeze(1),
    ).all(dim=-1).detach().cpu().tolist()  # (b, n),  bool

    dist_dict = utils.compute_point_ray_distance_in_chunks(
        points=points,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
    )
    dists = dist_dict['dists']  # (b, m, n)
    ts = dist_dict['ts']  # (b, m, n)

    # note that the part below are the actual bottleneck
    all_ray2pidxs = []  # b -> ray_idx -> point_idx
    for b in range(batch_size):
        ray2pidxs = []
        for m in range(n_rays):
            mask = torch.logical_and(
                dists[b, m] <= ray_radius[b],
                ts[b, m] >= 0,
            )
            pidxs = torch.nonzero(
                mask,
                as_tuple=False,
            )[:, 0].detach().cpu().tolist()

            pidxs = [p for p in pidxs if valid_mask[b][p]]

            ray2pidxs.append(pidxs)
        all_ray2pidxs.append(ray2pidxs)

    return all_ray2pidxs


def find_k_neighbor_points_of_rays_brute_force(
        points: torch.Tensor,  # (b, n, 3)
        k: int,
        ray_origins: torch.Tensor,  # (b, m, 3)
        ray_directions: torch.Tensor,  # (b, m, 3)
        ray_radius: T.Union[torch.Tensor, float],  # (b,)
        grid_size: T.Union[torch.Tensor, int],  # (b, 3)
        grid_center: T.Union[torch.Tensor, float, None] = 0.,  # (b, 3)
        grid_width: T.Union[torch.Tensor, float, None] = 1.,  # (b, 3)
) -> T.List[T.List[T.List[int]]]:
    """
    Modified version of find_neighbor_points_of_rays_brute_force
    return k neighbors within ray_radius
    report the valid numbers of neighbors in neighbor_num

    Args:
        points:
        k:
        ray_origins:
        ray_directions:
        ray_radius:
        grid_size:
        grid_center:
        grid_width:

    Returns:

    """

    smemory = torch.cuda.memory_allocated(device=points.device)

    batch_size, n_rays, _ = ray_origins.shape

    # determine a grid
    if grid_center is None:
        grid_center = points.mean(dim=-2)  # (b, 3)

    if grid_width is None:
        grid_width = points.max(dim=-2) - points.min(dim=-1)  # (b, 3)

    if isinstance(grid_center, float):
        grid_center = torch.tensor(grid_center, dtype=points.dtype, device=points.device)
        grid_center = grid_center.view(1, 1).expand(batch_size, 3)  # (b, 3)
    if isinstance(grid_width, (float, int)):
        grid_width = torch.tensor(grid_width, dtype=points.dtype, device=points.device)
        grid_width = grid_width.view(1, 1).expand(batch_size, 3)  # (b, 3)
    if isinstance(grid_size, (float, int)):
        grid_size = torch.tensor(grid_size, dtype=torch.long, device=points.device)
        grid_size = grid_size.view(1, 1).expand(batch_size, 3)  # (b, 3)
    grid_size = grid_size.long()  # (b, 3)
    total_cells = torch.prod(grid_size, dim=-1)  # (b,)

    # handle outside points
    grid_from = grid_center - grid_width / 2  # (b, 3)
    grid_to = grid_center + grid_width / 2  # (b, 3)
    valid_points = torch.logical_and(
        points >= grid_from.unsqueeze(1),
        points <= grid_to.unsqueeze(1),
    ).all(dim=-1)  # (b, n),  bool

    dist_dict = utils.compute_point_ray_distance_in_chunks(
        points=points,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
    )
    dists = dist_dict['dists']  # (b, m, n)
    ts = dist_dict['ts']  # (b, m, n)

    t_min = 0.
    t_max = 1.e10

    invalid_mask = torch.logical_or(ts < t_min, ts > t_max)  # (*, m, n)
    invalid_mask = torch.logical_or(invalid_mask, torch.logical_not(valid_points).unsqueeze(-2))  # (b, m, n)
    invalid_mask = torch.logical_or(invalid_mask, dists > ray_radius)  # (*, m, n)

    dists[invalid_mask] = torch.inf

    # sort dists of the points for each ray
    sorted_dists, sorted_idxs = torch.sort(dists, dim=-1)  # (*, m, n), (*, m, n)

    # keep only k nearest neighbors
    sorted_dists = sorted_dists[..., :k].clone()  # (*, m, k) the distance of the k nearest points to each ray
    sorted_idxs = sorted_idxs[..., :k].clone()  # (*, m, k) the index of k nearest points

    sorted_ts = torch.gather(
        input=ts,  # (*, m, n)
        dim=-1,
        index=sorted_idxs  # (*, m, k)
    )  # (b, m, k)  neighbot_ts[b, m, i] = ts[b, m, neighbor_xyz_w_idxs[b, m, i]]

    neighbor_num = torch.sum(torch.isfinite(sorted_dists), dim=-1)

    total_memory_gt = torch.cuda.memory_allocated(device=points.device) - smemory
    print(f' gt use memory : {total_memory_gt / 2 ** 30} GB')

    return sorted_idxs, neighbor_num
