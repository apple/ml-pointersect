#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import sys
sys.path.append("cdslib")

import unittest
import numpy as np
import torch
import pr_cpp
import typing as T
from pointersect.pr import naive
from timeit import default_timer as timer
import random


class TestIndexing(unittest.TestCase):
    def test_sub2ind(self, b=10, n=20, max_size=20):
        size = torch.randint(max_size-1, size=[b, 3]) + 1 # (b, 3)
        total_cells = torch.prod(size, dim=-1)  # (b,)
        subidx = torch.rand(b, n, 3).floor().long()

        # python
        stime = timer()
        ind_gt = naive.sub2ind(subidx, size)
        time_python = timer() - stime

        # cpp
        stime = timer()
        ind_cpp = pr_cpp.sub2ind(subidx, size)
        time_cpp = timer() - stime

        print('sub2ind')
        print(f'python: {time_python * 1000.:.3f} ms')
        print(f'cpp: {time_cpp * 1000.:.3f} ms ({time_python/time_cpp:.2f} speed up)')

        assert torch.allclose(ind_gt, ind_cpp)


    def test_ind2sub(self, b=10, n=20, max_size=20):
        size = torch.randint(max_size-1, size=[b, 3]) + 1  # (b, 3)
        total_cells = torch.prod(size, dim=-1)  # (b,)
        ind = (torch.rand(b, n) * total_cells.unsqueeze(-1)).floor().long()

        # python
        stime = timer()
        out_gt = naive.ind2sub(ind, size)
        time_python = timer() - stime

        # cpp
        stime = timer()
        out_cpp = pr_cpp.ind2sub(ind, size)
        time_cpp = timer() - stime

        print('ind2sub')
        print(f'python: {time_python * 1000.:.3f} ms')
        print(f'cpp: {time_cpp * 1000.:.3f} ms ({time_python/time_cpp:.2f} speed up)')

        assert torch.allclose(out_gt, out_cpp)


class TestGatherPoints(unittest.TestCase):
    def test(self, b=10, n=20, max_size=20):
        size = torch.randint(max_size-1, size=[b, 3]) + 1 # (b, 3)
        total_cells = torch.prod(size, dim=-1)  # (b,)
        grid_idxs = (torch.rand(b, n) * total_cells.unsqueeze(1)).floor().long()
        valid_mask = torch.randn(b, n) > 0.5

        stime = timer()
        all_cell2pidx_gt = naive.gather_points(
            grid_idxs=grid_idxs,
            total_cells=total_cells,
            valid_mask=valid_mask,
        )
        time_python = timer() - stime

        stime = timer()
        _all_cell2pidx = pr_cpp.gather_points(
            grid_idxs,
            total_cells,
            valid_mask,
        )
        time_cpp = timer() - stime

        # convert list of idxmap to list of list of list
        _all_cell2pidx = [c.get_table() for c in _all_cell2pidx]
        all_cell2pidx = []
        for b in range(len(_all_cell2pidx)):
            cell2pidx = [[] for i in range(total_cells[b])]
            for gidx in _all_cell2pidx[b]:
                cell2pidx[gidx] = list(_all_cell2pidx[b][gidx])
            all_cell2pidx.append(cell2pidx)

        print('gather points:')
        print(f'python: {time_python * 1000.:.3f} ms')
        print(f'cpp: {time_cpp * 1000.:.3f} ms ({time_python / time_cpp:.2f} speed up)')

        assert len(all_cell2pidx_gt) == len(all_cell2pidx)
        for b in range(len(all_cell2pidx_gt)):
            cell2pidx_gt = all_cell2pidx_gt[b]
            cell2pidx = all_cell2pidx[b]
            assert len(cell2pidx_gt) == len(cell2pidx)
            for gidx in range(len(cell2pidx_gt)):
                pidxs_gt = np.array(sorted(list(cell2pidx_gt[gidx])))
                pidxs = np.array(sorted(list(cell2pidx[gidx])))
                assert np.allclose(pidxs_gt, pidxs)



class TestGetGridIdx(unittest.TestCase):

    def test(self, b=10, n=20, max_size=20):
        points = torch.randn(b, n, 3)
        grid_size = (torch.rand(b, 3) * 9 + 1).floor().long()
        grid_center = torch.randn(b, 3)
        grid_width = torch.rand(b, 3) * 2 + 0.1

        stime = timer()
        grid_idxs_gt, valid_mask_gt = naive.get_grid_idx(
            points=points,
            grid_size=grid_size,
            center=grid_center,
            grid_width=grid_width,
            mode='ind',
        )
        time_python = timer() - stime

        stime = timer()
        grid_idxs, valid_mask = pr_cpp.get_grid_idx(
            points,
            grid_size,
            grid_center,
            grid_width,
            'ind',
        )
        time_cpp = timer() - stime

        print('get_grid_idx:')
        print(f'python: {time_python * 1000.:.3f} ms')
        print(f'cpp: {time_cpp * 1000.:.3f} ms ({time_python / time_cpp:.2f} speed up)')

        assert torch.allclose(grid_idxs_gt, grid_idxs)
        assert torch.allclose(valid_mask_gt, valid_mask)



class TestGridRayIntersection(unittest.TestCase):

    def test(self, b=1, m=100, max_size=20):

        ray_origins = torch.randn(b, m, 3)  # (b, m, 3)
        ray_directions = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)  # (b, m, 3)
        ray_radius = torch.rand(b,)*2 + 1e-3  # (b,)
        grid_size = (torch.rand(b, 3) * (max_size-1) + 1).floor().long()  # (b, 3)
        total_cells = torch.prod(grid_size, dim=-1)
        grid_center = torch.randn(b, 3)
        grid_width = torch.rand(b, 3) * 20 + 10

        stime = timer()
        all_grid_idxs_gt = naive.grid_ray_intersection(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_center=grid_center,
            grid_width=grid_width,
        )
        time_python = timer() - stime

        stime = timer()
        _all_grid_idxs = pr_cpp.grid_ray_intersection(
            ray_origins,
            ray_directions,
            ray_radius,
            grid_size,
            grid_center,
            grid_width,
        )
        time_cpp = timer() - stime

        # convert list of idxmap to list of list of list
        _all_grid_idxs = [c.get_table() for c in _all_grid_idxs]
        all_grid_idxs = []
        for b in range(len(_all_grid_idxs)):
            grid_idxs = [[] for i in range(m)]
            for midx in _all_grid_idxs[b]:
                grid_idxs[midx] = list(_all_grid_idxs[b][midx])
            all_grid_idxs.append(grid_idxs)

        print('grid_ray_intersection:')
        print(f'python: {time_python * 1000.:.3f} ms')
        print(f'cpp: {time_cpp * 1000.:.3f} ms ({time_python / time_cpp:.2f} speed up)')

        assert len(all_grid_idxs_gt) == len(all_grid_idxs)
        for b in range(len(all_grid_idxs_gt)):
            grid_idxs_gt = all_grid_idxs_gt[b]
            grid_idxs = all_grid_idxs[b]
            assert len(grid_idxs_gt) == len(grid_idxs)
            for midx in range(len(grid_idxs_gt)):
                gidxs_gt = np.array(sorted(list(grid_idxs_gt[midx])))
                gidxs = np.array(sorted(list(grid_idxs[midx])))
                assert np.allclose(gidxs_gt, gidxs)


# class TestGridRayIntersectionV2(unittest.TestCase):
#
#     def test(self, b=1, m=100, max_size=20):
#
#         ray_origins = torch.randn(b, m, 3)  # (b, m, 3)
#         ray_directions = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)  # (b, m, 3)
#         ray_radius = torch.rand(b, ) * 2 + 1e-3  # (b,)
#         grid_size = (torch.rand(b, 3) * (max_size - 1) + 1).floor().long()  # (b, 3)
#         total_cells = torch.prod(grid_size, dim=-1)
#         grid_center = torch.randn(b, 3)
#         grid_width = torch.rand(b, 3) * 20 + 10
#
#         stime = timer()
#         all_grid_idxs_gt = naive.grid_ray_intersection(
#             ray_origins=ray_origins,
#             ray_directions=ray_directions,
#             ray_radius=ray_radius,
#             grid_size=grid_size,
#             grid_center=grid_center,
#             grid_width=grid_width,
#         )
#         time_python = timer() - stime
#
#         stime = timer()
#         _all_grid_idxs = pr_cpp.grid_ray_intersection_v2(
#             ray_origins,
#             ray_directions,
#             ray_radius,
#             grid_size,
#             grid_center,
#             grid_width,
#         )
#         time_cpp = timer() - stime
#
#         # convert list of idxmap to list of list of list
#         _all_grid_idxs = [c.get_table() for c in _all_grid_idxs]
#         all_grid_idxs = []
#         for b in range(len(_all_grid_idxs)):
#             grid_idxs = [[] for i in range(m)]
#             for midx in _all_grid_idxs[b]:
#                 grid_idxs[midx] = list(_all_grid_idxs[b][midx])
#             all_grid_idxs.append(grid_idxs)
#
#         print('grid_ray_intersection_v2:')
#         print(f'python: {time_python * 1000.:.3f} ms')
#         print(f'cpp: {time_cpp * 1000.:.3f} ms ({time_python / time_cpp:.2f} speed up)')
#
#         assert len(all_grid_idxs_gt) == len(all_grid_idxs)
#         for b in range(len(all_grid_idxs_gt)):
#             grid_idxs_gt = all_grid_idxs_gt[b]
#             grid_idxs = all_grid_idxs[b]
#             assert len(grid_idxs_gt) == len(grid_idxs)
#             for midx in range(len(grid_idxs_gt)):
#                 gidxs_gt = np.array(sorted(list(grid_idxs_gt[midx])))
#                 gidxs = np.array(sorted(list(grid_idxs[midx])))
#                 assert np.allclose(gidxs_gt, gidxs)


class TestFindNeighborPointsOfRays(unittest.TestCase):
    def _test(
            self,
            points: torch.Tensor,  # (b, n, 3)
            ray_origins: torch.Tensor,  # (b, m, 3)
            ray_directions: torch.Tensor,  # (b, m, 3)
            ray_radius: T.Union[torch.Tensor, float],  # (b,)
            grid_size: T.Union[torch.Tensor, int],  # (b, 3)
            grid_center: T.Union[torch.Tensor, float, None] = 0.,  # (b, 3)
            grid_width: T.Union[torch.Tensor, float, None] = 1.,  # (b, 3)
    ):
        batch_size, n_rays, _ = ray_origins.shape

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


        # stime = timer()
        # all_ray2pidxs_python = naive.find_neighbor_points_of_rays(
        #     points=points,
        #     ray_origins=ray_origins,
        #     ray_directions=ray_directions,
        #     ray_radius=ray_radius,
        #     grid_size=grid_size,
        #     grid_center=grid_center,
        #     grid_width=grid_width,
        #     # include_outside=include_outside,
        # )
        # total_time_python = timer() - stime

        stime = timer()
        all_ray2pidxs_gt = naive.find_neighbor_points_of_rays_brute_force(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_center=grid_center,
            grid_width=grid_width,
            # include_outside=include_outside,
        )
        total_time_brute_force = timer() - stime
        # print(f'python/gt = {total_time_python:1f}/{total_time_brute_force:1f} = {total_time_python/total_time_brute_force*100:.3f}%')

        # stime = timer()
        # all_ray2pidxs = pr_cpp.find_neighbor_points_of_rays(
        #     points,
        #     ray_origins,
        #     ray_directions,
        #     ray_radius,
        #     grid_size,
        #     grid_center,
        #     grid_width,
        #     0,
        #     1e10,
        #     "v1",
        # )
        # total_time_cpp = timer() - stime
        # print(f'cpp_v1/gt = {total_time_cpp:1f}/{total_time_brute_force:1f} = {total_time_cpp / total_time_brute_force * 100:.3f}%')
        # print(f'cpp_v1/python = {total_time_cpp:1f}/{total_time_python:1f} = {total_time_cpp / total_time_python * 100:.3f}%')

        stime = timer()
        all_ray2pidxs_v2 = pr_cpp.find_neighbor_points_of_rays(
            points,
            ray_origins,
            ray_directions,
            ray_radius,
            grid_size,
            grid_center,
            grid_width,
            0,
            1e10,
            "v2",
        )
        total_time_cpp = timer() - stime
        print(f'cpp_v2/gt = {total_time_cpp:1f}/{total_time_brute_force:1f} = {total_time_cpp / total_time_brute_force * 100:.3f}%')
        # print(f'cpp_v2/python = {total_time_cpp:1f}/{total_time_python:1f} = {total_time_cpp / total_time_python * 100:.3f}%')


        # # check v1
        # assert len(all_ray2pidxs) == batch_size
        # assert len(all_ray2pidxs_gt) == batch_size
        #
        # for b in range(batch_size):
        #     ray2pidxs = all_ray2pidxs[b]
        #     ray2pidxs_gt = all_ray2pidxs_gt[b]
        #     assert len(ray2pidxs) == n_rays
        #     assert len(ray2pidxs_gt) == n_rays
        #
        #     for m in range(n_rays):
        #         pidxs = ray2pidxs[m]
        #         pidxs_gt = ray2pidxs_gt[m]
        #         pidxs = np.sort(np.array(pidxs))
        #         pidxs_gt = np.sort(np.array(pidxs_gt))
        #         assert len(pidxs) == len(pidxs_gt), f"[{b},{m}], {pidxs}, {pidxs_gt}"
        #         assert np.allclose(pidxs, pidxs_gt), f"[{b},{m}], {pidxs}, {pidxs_gt}"


        # check v2
        assert len(all_ray2pidxs_v2) == batch_size
        assert len(all_ray2pidxs_gt) == batch_size

        for b in range(batch_size):
            ray2pidxs = all_ray2pidxs_v2[b]
            ray2pidxs_gt = all_ray2pidxs_gt[b]
            assert len(ray2pidxs) == n_rays
            assert len(ray2pidxs_gt) == n_rays

            for m in range(n_rays):
                pidxs = ray2pidxs[m]
                pidxs_gt = ray2pidxs_gt[m]
                pidxs = np.sort(np.array(pidxs))
                pidxs_gt = np.sort(np.array(pidxs_gt))
                assert len(pidxs) == len(pidxs_gt), f"[{b},{m}], {pidxs}, {pidxs_gt}"
                assert np.allclose(pidxs, pidxs_gt), f"[{b},{m}], {pidxs}, {pidxs_gt}"


    def test1(
            self,
            b=1,
            m=1,
            n=10000,
            max_grid_width=20.,
            max_ray_radius=1.,
            max_grid_size=100,
            seed=1,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        print(f"=======================================")
        print(f"m = {m}, n = {n}, max_gw={max_grid_width}, max_radius={max_ray_radius}, max_gs={max_grid_size}")

        points = torch.rand(b, n, 3) * max_grid_width - max_grid_width / 2
        ray_origins = torch.randn(b, m, 3)
        ray_directions = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)
        ray_radius = torch.rand(b) * (max_ray_radius - 0.00001) + 0.00001
        grid_size = (torch.rand(b, 3) * (max_grid_size - 1) + 1).long()

        self._test(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
        )

    def test_many(self):

        ns = [1000, 10000, 100000, 1000000, 10000000, 100000000]
        ms = [1] * len(ns)

        for m, n in zip(ms, ns):
            self.test1(
                m=m,
                n=n,
                max_ray_radius=2.,
                max_grid_width=20.,
                max_grid_size=100,
                seed=None,
            )


if __name__ == '__main__':
    unittest.main()
