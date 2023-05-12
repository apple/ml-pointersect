#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import typing as T
from timeit import default_timer as timer

import torch
from torch import profiler

from plib import utils


def find_neighbors_and_rectify(
        points_w: torch.Tensor,  # (b, n, 3)
        ray_origins_w: torch.Tensor,  # (b, m, 3)
        ray_directions_w: torch.Tensor,  # (b, m, 3)
        k: int,
        t_min: float = 1.0e-8,
        t_max: float = 1.0e10,
        t_init: torch.Tensor = None,
        other_maps: T.List[torch.Tensor] = None,
        rotate_other_maps: T.List[bool] = None,
        translate_other_maps: T.List[bool] = None,
        max_chunk_size: int = int(1e8),
        pr_params: T.Dict[str, T.Any] = None,
        randomize_translate: bool = False,
        printout: bool = False,
        cached_info: T.Union[T.Dict[str, torch.Tensor], None] = None,
        valid_mask: torch.Tensor = None,  # (b, n, 1)
        enable_timing: bool = False,
) -> T.Dict[str, T.Any]:
    """
    Find the k nearest points by each ray and rectify the coordinate system as if the rays are
    in the direction of (0,0,1) and originated from (0,0,0).
    additional feature inputs may include camera view direction or local frame vectors
    and these vectors need to be rectified as well (apply rotation only)

    Args:
        points_w:  (b, n, 3)
        ray_origins_w:
        ray_directions_w:
        k:
        t_min:
        t_max:
        other_maps:
            a list of (b, n, d) to associated with each point_w
        rotate_other_maps:
            a list of bool to indicate whether the feature needs to be rotated,
            i.e, multiplied by Rs_w2n
        translate_other_maps:
            a list of bool to indicate whether the feature needs to be translated
            i.e, added by translation_w2n
        cached_info:
            a dictionary containing the grid cell to point index so pr does not
            need to construct it again.
        valid_mask:
            (b, n, 1) whether the point (including point at inf) should be considered
            in the neighbor search.
    """

    if max_chunk_size < 0:
        max_chunk_size = int(1e13)

    if other_maps is None:
        other_maps = []

    b, n, _ = points_w.shape
    m = ray_origins_w.size(-2)
    timing_info = dict()

    # gather neighboring points using their xyz_w  (can be sped up by octree)
    if enable_timing:
        torch.cuda.synchronize()
    stime_total = timer()
    stime_pr = timer()
    # stime_pr_cuda = torch.cuda.Event(enable_timing=enable_timing)
    with profiler.record_function('pr'):
        with torch.no_grad():
            neighbor_info = utils.get_k_neighbor_points_in_chunks(
                points=points_w,  # (b, n, 3)
                ray_origins=ray_origins_w,  # (b, m, 3)
                ray_directions=ray_directions_w,  # (b, m, 3)
                k=k,
                t_min=t_min,
                t_max=t_max,
                t_init=t_init,
                max_chunk_size=max_chunk_size,
                pr_params=pr_params,
                printout=printout,
                cached_info=cached_info,
                valid_mask=valid_mask,
            )
    if enable_timing:
        torch.cuda.synchronize()
    # etime_pr_cuda = torch.cuda.Event(enable_timing=enable_timing)
    total_time_pr = timer() - stime_pr
    timing_info['pr'] = total_time_pr

    # use the sorted_idxs to get the points (separately for each b, we want to select k index from xyz_w for each ray)
    total_misc = 0
    stime_misc = timer()
    neighbor_xyz_w_idxs = neighbor_info['sorted_idxs']  # (b, m, k)
    neighbor_ts = neighbor_info['sorted_ts']  # (b, m, k)   # some t may be at t_min - 1
    valid_mask = torch.logical_and(neighbor_ts >= t_min, neighbor_ts < t_max)  # (b, m, k)
    cached_info = neighbor_info.get('cached_info', None)
    # ts = neighbor_info['ts']  # (b, m, n)
    i_shape = list(neighbor_xyz_w_idxs.shape)  # (b, m, k)
    total_misc += (timer() - stime_misc)

    total_gather = 0
    stime_gather = timer()
    neighbor_points = torch.gather(
        input=points_w.unsqueeze(-3).expand(b, m, n, 3),  # (b, m, n, 3)
        dim=-2,
        index=neighbor_xyz_w_idxs.unsqueeze(-1).expand(*(i_shape + [3])),  # (b, m, k, 3)
    )  # (b, m, k, 3)  neighbor_points[b, m, i] = xyz_w[b, m, neighbor_xyz_w_idxs[b, m, i]]
    total_gather += (timer() - stime_gather)

    # rectify the neighboring points so that ray origin is 0, ray direction is (0,0,1), t_min = 0
    if enable_timing:
        torch.cuda.synchronize()
    stime_rectify = timer()
    with profiler.record_function('rectify'):
        rectify_out_dict = utils.rectify_points(
            points=neighbor_points,  # (b, m, k, 3)
            ray_origins=ray_origins_w,  # (b, m, 3)
            ray_directions=ray_directions_w,  # (b, m, 3)
            translate=True,
            ts=neighbor_ts,  # (b, m, k)
            randomize_translate=randomize_translate,
            t_min=t_min,
            t_max=t_max,
        )
    if enable_timing:
        torch.cuda.synchronize()
    timing_info['rectify'] = timer() - stime_rectify
    timing_info.update(rectify_out_dict['timing_info'])

    rectified_points = rectify_out_dict['points_n']  # (b, m, k, 3)  in the transformed coord
    ts_shift = rectify_out_dict['tt']  # (b, m)  # (ts_new = ts - ts_shift)
    Rs_w2n = rectify_out_dict['Rs_w2n']  # (b, m, 3, 3)  from world coord to the transformed coord
    translation_w2n = rectify_out_dict['translation_w2n']  # (b, m, 3, 1)  from the world coord to the transformed coord

    # gather other_map
    stime_gather = timer()
    gathered_other_maps = []
    if other_maps is not None:
        if rotate_other_maps is None:
            rotate_other_maps = [False] * len(other_maps)
        if translate_other_maps is None:
            translate_other_maps = [False] * len(other_maps)

    assert (len(other_maps) == len(rotate_other_maps))
    assert (len(other_maps) == len(translate_other_maps))

    for i, other_map in enumerate(other_maps):
        d = other_map.size(-1)
        gathered_other_map = torch.gather(
            input=other_map.unsqueeze(-3).expand(b, m, n, d),  # (b, m, n, d)
            dim=-2,
            index=neighbor_xyz_w_idxs.unsqueeze(-1).expand(*(i_shape + [d])),  # (b, m, k, d)
        )  # (b, m, k, d)  neighbor_points[b, m, i] = xyz_w[b, m, neighbor_xyz_w_idxs[b, m, i]]

        # rectify features if needed
        if rotate_other_maps[i]:
            gathered_other_map = (Rs_w2n.unsqueeze(-3) @ gathered_other_map.unsqueeze(-1)).squeeze(-1)  # (b, m, k, 3)
        if translate_other_maps[i]:
            gathered_other_map = (gathered_other_map.unsqueeze(-1) + translation_w2n.unsqueeze(-3)).squeeze(
                -1)  # (b, m, k, 3)
        gathered_other_maps.append(gathered_other_map)

    # timing_info['assert_in_find_neighbors_and_rectify'] = total_assert
    timing_info['misc_in_find_neighbors_and_rectify'] = total_misc
    total_gather += (timer() - stime_gather)
    timing_info['gather_in_find_neighbors_and_rectify'] = total_gather
    total_time = timer() - stime_total
    timing_info['total_find_neighbors_and_rectify'] = total_time

    out_dict = dict(
        neighbor_info=neighbor_info,
        # rectify_out_dict=rectify_out_dict,  # not used
        points_n=rectified_points,  # (b, m, k, 3)  in the transformed coord
        ts_shift=ts_shift,  # (b, m)  # (ts_n = ts - ts_shift)
        Rs_w2n=Rs_w2n,  # (b, m, 3, 3)  from world coord to the transformed coord
        translation_w2n=translation_w2n,  # (b, m, 3, 1)  from the world coord to the transformed coord
        neighbor_other_maps=gathered_other_maps,  # list of (b, m, k, d) associated with each point
        cached_info=cached_info,  # dict containing cached info for pr
        valid_mask=valid_mask,  # (b, m, k) bool
        timing_info=timing_info,
    )

    if pr_params is not None:
        out_dict['neighbor_num'] = neighbor_info['neighbor_num']  # (b, m)

    return out_dict


def positional_encoding(
        tensor: torch.Tensor,
        num_encoding_functions: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor:
            (*, dim_in) Input tensor to be positionally encoded.
        num_encoding_functions:
            Number of encoding functions used to compute a positional encoding (default: 6).
        include_input:
            Whether to include the input in the positional encoding (default: True).
        log_sampling:
            whether to sample the sinusoid frequencies in log scale.

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.  (*, dim_out)

    .. math::
        dim_{out} = d_{in} * include_input + num_encoding_functions * 2
    """
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_embedding_function(
        num_encoding_functions: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
):
    r"""
    Returns a lambda function that internally calls positional_encoding.

    Args:
        num_encoding_functions:
            Number of encoding functions used to compute a positional encoding (default: 6).
        include_input:
            Whether to include the input in the positional encoding (default: True).
        log_sampling:
            whether to sample the sinusoid frequencies in log scale.

    Returns:
        A lambda function that convert input to positional encoding.
        the output dimension is (*, dim_out), where
        :math:`dim_{out} = d_{in} * include_input + num_encoding_functions * 2`
    """
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )
