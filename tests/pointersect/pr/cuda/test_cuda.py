#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import sys
sys.path.append("cdslib")

import unittest
import numpy as np
import torch

import typing as T
from pointersect.pr import naive
from pointersect.pr.pr_utils import pr_cpp, pr_cuda
from timeit import default_timer as timer
import random
from plib import utils



class TestGetGridIdx(unittest.TestCase):

    def _test(self, b=10, n=20, max_size=20, mode='ind'):
        if not torch.cuda.is_available():
            return

        points = torch.randn(b, n, 3)
        grid_size = (torch.rand(b, 3) * (max_size-1) + 1).floor().long()
        grid_center = torch.randn(b, 3)
        grid_width = torch.rand(b, 3) * 2 + 0.1

        stime = timer()
        grid_idxs_gt, valid_mask_gt = naive.get_grid_idx(
            points=points,
            grid_size=grid_size,
            center=grid_center,
            grid_width=grid_width,
            mode=mode,
        )
        time_python = timer() - stime

        stime = timer()
        grid_idxs, valid_mask = pr_cpp.get_grid_idx(
            points,
            grid_size,
            grid_center,
            grid_width,
            mode,
        )
        time_cpp = timer() - stime

        points = points.cuda()
        grid_size = grid_size.cuda()
        grid_center = grid_center.cuda()
        grid_width = grid_width.cuda()

        stime = timer()
        grid_idxs, valid_mask, cell_counts = pr_cuda.get_grid_idx(
            points,
            grid_size,
            grid_center,
            grid_width,
            mode,
        )
        time_cuda = timer() - stime
        grid_idxs = grid_idxs.cpu()
        valid_mask = valid_mask.cpu()
        cell_counts = cell_counts.cpu()

        print('get_grid_idx:')
        print(f'python: {time_python * 1000.:.3f} ms')
        print(f'cpp: {time_cpp * 1000.:.3f} ms ({time_python / time_cpp:.2f} speed up)')
        print(f'cuda: {time_cuda * 1000.:.3f} ms ({time_python / time_cuda:.2f} speed up)')

        assert torch.allclose(grid_idxs_gt, grid_idxs)
        assert torch.allclose(valid_mask_gt, valid_mask)


        # check cell count
        batch_size = points.size(0)
        max_total_cells = grid_size.prod(dim=-1).max()
        cell_counts_gt = torch.zeros(batch_size, max_total_cells, dtype=torch.int)
        if mode == 'subidx':
            grid_idxs_ind = naive.sub2ind(grid_idxs_gt, grid_size.cpu())  # (b, n)
        else:
            grid_idxs_ind = grid_idxs_gt
        for b in range(grid_idxs_ind.size(0)):
            for i in range(grid_idxs_ind.size(1)):
                if valid_mask_gt[b][i]:
                    cell_counts_gt[b][grid_idxs_ind[b][i]] += 1
        assert torch.allclose(cell_counts, cell_counts_gt)

    def test_ind(self):
        self._test(
            b=100,
            n=2000,
            mode='ind',
        )

    def test_subidx(self, repeat=1):
        for r in range(repeat):
            self._test(
                b=100,
                n=20000,
                mode='subidx',
            )


class TestGatherPoints(unittest.TestCase):
    def test(self, b=10, n=100, max_size=20):
        if not torch.cuda.is_available():
            return

        size = torch.randint(max_size-1, size=[b, 3]) + 1  # (b, 3)
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

        # compute cell_counts
        max_num_cells = total_cells.max()
        cell_counts = torch.zeros(b, max_num_cells, dtype=torch.int)
        for bidx in range(b):
            for pidx in range(n):
                if valid_mask[bidx, pidx]:
                    cell_counts[bidx, grid_idxs[bidx,pidx]] += 1

        grid_idxs = grid_idxs.cuda()
        total_cells = total_cells.cuda()
        valid_mask = valid_mask.cuda()
        cell_counts = cell_counts.cuda()

        stime = timer()
        all_cell2pidx = pr_cuda.gather_points_v1(
            grid_idxs,
            total_cells,
            valid_mask,
            cell_counts,
        )
        time_cuda_v1 = timer() - stime

        stime = timer()
        membank, cell_start_idxs = pr_cuda.gather_points_v2(
            grid_idxs,
            total_cells,
            valid_mask,
            cell_counts,
        )  # membank: (b, n),  cell_start_idxs: (b, n_cell+1)
        time_cuda_v2 = timer() - stime

        # contruct all_cell2pidx from membank
        all_cell2pidx_v2 = []
        for ib in range(b):
            cell2pidx = []
            for ic in range(total_cells[ib]):
                pidxs = membank[ib, cell_start_idxs[ib, ic]:cell_start_idxs[ib, ic+1]]
                cell2pidx.append(pidxs)
            all_cell2pidx_v2.append(cell2pidx)


        print('gather points:')
        print(f'python: {time_python * 1000.:.3f} ms')
        print(f'cpp: {time_cpp * 1000.:.3f} ms ({time_python / time_cpp:.2f} speed up)')
        print(f'cuda_v1: {time_cuda_v1 * 1000.:.3f} ms ({time_python / time_cuda_v1:.2f} speed up)')
        print(f'cuda_v2: {time_cuda_v2 * 1000.:.3f} ms ({time_python / time_cuda_v2:.2f} speed up)')

        # check v1
        assert len(all_cell2pidx_gt) == len(all_cell2pidx)
        for b in range(len(all_cell2pidx_gt)):
            cell2pidx_gt = all_cell2pidx_gt[b]
            cell2pidx = all_cell2pidx[b]
            assert len(cell2pidx_gt) == len(cell2pidx)
            for gidx in range(len(cell2pidx_gt)):
                pidxs_gt = np.array(sorted(list(cell2pidx_gt[gidx])))
                pidxs = np.array(sorted(cell2pidx[gidx].detach().cpu().tolist()))
                assert np.allclose(pidxs_gt, pidxs)

        # check v2
        assert len(all_cell2pidx_gt) == len(all_cell2pidx_v2)
        for b in range(len(all_cell2pidx_gt)):
            cell2pidx_gt = all_cell2pidx_gt[b]
            cell2pidx = all_cell2pidx_v2[b]
            assert len(cell2pidx_gt) == len(cell2pidx)
            for gidx in range(len(cell2pidx_gt)):
                pidxs_gt = np.array(sorted(list(cell2pidx_gt[gidx])))
                pidxs = np.array(sorted(cell2pidx[gidx].detach().cpu().tolist()))
                assert np.allclose(pidxs_gt, pidxs)

    def test2(self):
        self.test(
            b=10,
            n=52000,
            max_size=10,
        )


class TestGridRayIntersection(unittest.TestCase):

    def test(self, b=10, m=1000, max_size=40, seed=0):

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        ray_origins = torch.randn(b, m, 3)  # (b, m, 3)
        ray_directions = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)  # (b, m, 3)
        ray_radius = torch.rand(b,)*0.8 + 1e-3  # (b,)
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
        _all_grid_idxs_cpp = pr_cpp.grid_ray_intersection_v2(
            ray_origins,
            ray_directions,
            ray_radius,
            grid_size,
            grid_center,
            grid_width,
        )
        time_cpp = timer() - stime

        # convert list of idxmap to list of list of list
        _all_grid_idxs_cpp = [c.get_table() for c in _all_grid_idxs_cpp]
        all_grid_idxs_cpp = []
        for b in range(len(_all_grid_idxs_cpp)):
            grid_idxs = [[] for i in range(m)]
            for midx in _all_grid_idxs_cpp[b]:
                grid_idxs[midx] = list(_all_grid_idxs_cpp[b][midx])
            all_grid_idxs_cpp.append(grid_idxs)

        ray_origins = ray_origins.cuda()
        ray_directions = ray_directions.cuda()
        ray_radius = ray_radius.cuda()
        grid_size = grid_size.cuda()
        grid_center = grid_center.cuda()
        grid_width = grid_width.cuda()

        stime = timer()
        _all_grid_idxs, n_all_grid_idx = pr_cuda.grid_ray_intersection(
            ray_origins,
            ray_directions,
            ray_radius,
            grid_size,
            grid_center,
            grid_width,
        )
        time_cuda = timer() - stime

        # convert list of idxmap to list of list of list
        all_grid_idxs = []
        for b in range(_all_grid_idxs.size(0)):
            grid_idxs = [_all_grid_idxs[b, i, :n_all_grid_idx[b,i]].detach().cpu().tolist() for i in range(_all_grid_idxs.size(1))]
            all_grid_idxs.append(grid_idxs)

        print('grid_ray_intersection:')
        print(f'python: {time_python * 1000.:.3f} ms')
        print(f'cpp: {time_cpp * 1000.:.3f} ms ({time_python / time_cpp:.2f} speed up)')
        print(f'cuda: {time_cuda * 1000.:.3f} ms ({time_python / time_cuda:.2f} speed up)')

        assert len(all_grid_idxs_cpp) == len(all_grid_idxs)
        for b in range(len(all_grid_idxs_cpp)):
            grid_idxs_gt = all_grid_idxs_cpp[b]
            grid_idxs = all_grid_idxs[b]
            assert len(grid_idxs_gt) == len(grid_idxs)
            for midx in range(len(grid_idxs_gt)):
                gidxs_gt = np.array(sorted(list(grid_idxs_gt[midx])))
                gidxs = np.array(sorted(list(grid_idxs[midx])))
                if not np.allclose(gidxs_gt, gidxs):
                    print('gidxs_gt:')
                    print(gidxs_gt)
                    print('gidxs:')
                    print(gidxs)
                    subidx_gt = naive.ind2sub(torch.from_numpy(gidxs_gt), grid_size.cpu())
                    subidx = naive.ind2sub(torch.from_numpy(gidxs), grid_size.cpu())
                    ii = np.where(gidxs_gt != gidxs)
                    diff_subidx_gt = subidx_gt[0][ii]
                    diff_subidx = subidx[0][ii]
                    diff_both = torch.cat((diff_subidx_gt, diff_subidx), dim=-1)
                    a = np.stack([gidxs_gt, gidxs, gidxs_gt-gidxs], axis=0)
                    assert False



class TestCollectPointsOnRay(unittest.TestCase):

    def test(self, b=1, n=100, m=2, max_size=40, seed=0):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)


        points = torch.randn(b, n, 3)
        ray_origins = torch.randn(b, m, 3)  # (b, m, 3)
        ray_directions = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)  # (b, m, 3)
        ray_radius = torch.rand(b,)*0.8 + 1e-3  # (b,)
        grid_size = (torch.rand(b, 3) * (max_size-1) + 1).floor().long()  # (b, 3)
        total_cells = torch.prod(grid_size, dim=-1)
        grid_center = torch.randn(b, 3)
        grid_width = torch.rand(b, 3) * 20 + 10

        points = points.cuda()
        ray_origins = ray_origins.cuda()
        ray_directions = ray_directions.cuda()
        ray_radius = ray_radius.cuda()
        grid_size = grid_size.cuda()
        grid_center = grid_center.cuda()
        grid_width = grid_width.cuda()
        total_cells = total_cells.cuda()

        stime = timer()
        grid_idxs, valid_mask, cell_counts = pr_cuda.get_grid_idx(
            points,
            grid_size,
            grid_center,
            grid_width,
            'ind',
        )
        time_cuda = timer() - stime
        print('CollectPointsOnRay::get_grid_idx:')
        print(f'cuda: {time_cuda * 1000.:.3f} ms ')

        stime = timer()
        gidx2pidx_bank, gidx_start_idx = pr_cuda.gather_points_v2(
            grid_idxs,
            total_cells,
            valid_mask,
            cell_counts,
        )  # membank: (b, n),  cell_start_idxs: (b, n_cell+1)
        time_cuda = timer() - stime
        print('CollectPointsOnRay::gather_points_v2:')
        print(f'cuda: {time_cuda * 1000.:.3f} ms ')

        stime = timer()
        ray2gidx, n_ray2gidx = pr_cuda.grid_ray_intersection(
            ray_origins,
            ray_directions,
            ray_radius,
            grid_size,
            grid_center,
            grid_width,
        )
        time_cuda = timer() - stime
        print('CollectPointsOnRay::grid_ray_intersection:')
        print(f'cuda: {time_cuda * 1000.:.3f} ms ')

        stime = timer()
        all_ray2pidxs, all_ray_start_idxs, all_ray_end_idxs = pr_cuda.collect_points_on_ray(
            ray2gidx,
            n_ray2gidx,
            gidx2pidx_bank,
            gidx_start_idx.int(),
            points,
            ray_origins,
            ray_directions,
            ray_radius,
            0.,
            1e12,
        )
        time_cuda = timer() - stime
        print('CollectPointsOnRay::collect_points_on_ray:')
        print(f'cuda: {time_cuda * 1000.:.3f} ms ')


        # construct ground truth
        all_ray2pidxs = all_ray2pidxs.cpu()
        all_ray_start_idxs = all_ray_start_idxs.cpu()
        all_ray_end_idxs = all_ray_end_idxs.cpu()
        ray2gidx = ray2gidx.cpu()  # (b, m, M)
        n_ray2gidx = n_ray2gidx.cpu()
        gidx2pidx_bank = gidx2pidx_bank.cpu()
        gidx_start_idx = gidx_start_idx.cpu()
        grid_idxs = grid_idxs.cpu()
        valid_mask = valid_mask.cpu()
        cell_counts = cell_counts.cpu()
        ray_radius = ray_radius.cpu()

        all_ray2pidxs_gt = []
        for ib in range(b):
            ray2pidxs = []
            r2g = ray2gidx[ib]  # (m, n_cells)
            n_r2g = n_ray2gidx[ib]  # (m,)
            for im in range(m):
                gidxs = r2g[im, :n_r2g[im]]  # (c,)
                all_pidxs = []
                for gidx in gidxs:
                    pidxs = gidx2pidx_bank[ib, gidx_start_idx[ib, gidx]:gidx_start_idx[ib, gidx+1]]
                    for pidx in pidxs:
                        p = points[ib, pidx:pidx+1]  # (1, 3)
                        out_dict = utils.compute_point_ray_distance(
                            points=p,
                            ray_origins=ray_origins[ib, im:im+1],
                            ray_directions=ray_directions[ib, im:im+1],
                        )
                        if out_dict['dists'][0, 0] > ray_radius[ib] \
                                or out_dict['ts'][0, 0] < 0 \
                                or out_dict['ts'][0, 0] > 1e12:  #  or not valid_mask[ib, pidx]:
                            pass
                        else:
                            all_pidxs.append(pidx)
                ray2pidxs.append(all_pidxs)
            all_ray2pidxs_gt.append(ray2pidxs)

        # construct list -> list -> list
        tmp_all_ray2pidxs = []
        for ib in range(all_ray2pidxs.size(0)):
            ray2pidxs = []
            for im in range(all_ray_start_idxs.size(1)):
                pidxs = all_ray2pidxs[ib, all_ray_start_idxs[ib, im]:all_ray_end_idxs[ib, im]]
                ray2pidxs.append(pidxs)
            tmp_all_ray2pidxs.append(ray2pidxs)
        all_ray2pidxs = tmp_all_ray2pidxs

        # compare with gt
        assert len(all_ray2pidxs) == b, f'{len(all_ray2pidxs)}, {b}'
        assert len(all_ray2pidxs_gt) == b, f'{len(all_ray2pidxs_gt)}, {b}'

        for ib in range(b):
            ray2pidxs = all_ray2pidxs[ib]
            ray2pidxs_gt = all_ray2pidxs_gt[ib]
            assert len(ray2pidxs) == m, f'{len(ray2pidxs)}, {m}'
            assert len(ray2pidxs_gt) == m, f'{len(ray2pidxs_gt)}, {m}'

            for im in range(m):
                pidxs = ray2pidxs[im]
                pidxs_gt = ray2pidxs_gt[im]
                pidxs = np.sort(np.array(pidxs))
                pidxs_gt = np.sort(np.array(pidxs_gt))
                assert len(pidxs) == len(pidxs_gt), f"[{ib},{im}], {pidxs}, {pidxs_gt}"
                assert np.allclose(pidxs, pidxs_gt), f"[{ib},{im}], {pidxs}, {pidxs_gt}"




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
        n_points = points.size(1)

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

        points = points.cuda()
        ray_origins = ray_origins.cuda()
        ray_directions = ray_directions.cuda()
        ray_radius = ray_radius.cuda()
        grid_size = grid_size.cuda()
        grid_center = grid_center.cuda()
        grid_width = grid_width.cuda()

        if n_rays * n_points < 5e9:
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
        else:
            total_time_brute_force = np.inf
            all_ray2pidxs_gt = None

        points = points.cpu()
        ray_origins = ray_origins.cpu()
        ray_directions = ray_directions.cpu()
        ray_radius = ray_radius.cpu()
        grid_size = grid_size.cpu()
        grid_center = grid_center.cpu()
        grid_width = grid_width.cpu()

        stime = timer()
        all_ray2pidxs_cpp = pr_cpp.find_neighbor_points_of_rays(
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

        points = points.cuda()
        ray_origins = ray_origins.cuda()
        ray_directions = ray_directions.cuda()
        ray_radius = ray_radius.cuda()
        grid_size = grid_size.cuda()
        grid_center = grid_center.cuda()
        grid_width = grid_width.cuda()

        stime = timer()
        all_ray2pidxs, all_ray_start_idxs, all_ray_end_idxs = pr_cuda.find_neighbor_points_of_rays(
            points,
            ray_origins,
            ray_directions,
            ray_radius,
            grid_size,
            grid_center,
            grid_width,
            0,
            1e10,
        )
        total_time_cuda = timer() - stime
        print(f'cuda/gt = {total_time_cuda:1f}/{total_time_brute_force:1f} = {total_time_cuda / total_time_brute_force * 100:.3f}%')
        print(f'cuda speed up: {total_time_brute_force/total_time_cuda:.2f}x over gt,  {total_time_cpp/total_time_cuda:.2f}x over cpp')

        all_ray2pidxs = all_ray2pidxs.detach().cpu()
        all_ray_start_idxs = all_ray_start_idxs.detach().cpu()
        all_ray_end_idxs = all_ray_end_idxs.detach().cpu()

        # check v2
        if all_ray2pidxs_gt is not None:
            assert all_ray2pidxs.size(0) == batch_size
            assert all_ray_start_idxs.size(0) == batch_size
            assert all_ray_end_idxs.size(0) == batch_size
            assert len(all_ray2pidxs_gt) == batch_size

            for b in range(batch_size):
                ray2pidxs = all_ray2pidxs[b]
                ray2pidxs_gt = all_ray2pidxs_gt[b]
                ray_start_idxs = all_ray_start_idxs[b]
                ray_end_idxs = all_ray_end_idxs[b]
                assert len(ray_start_idxs) == n_rays
                assert len(ray2pidxs_gt) == n_rays

                for m in range(n_rays):
                    pidxs = ray2pidxs[ray_start_idxs[m]:ray_end_idxs[m]]
                    pidxs_gt = ray2pidxs_gt[m]
                    pidxs = np.sort(np.array(pidxs))
                    pidxs_gt = np.sort(np.array(pidxs_gt))
                    if len(pidxs) != len(pidxs_gt):
                        print('here')

                    assert len(pidxs) == len(pidxs_gt), f"[{b},{m}], {pidxs}, {pidxs_gt}"
                    assert np.allclose(pidxs, pidxs_gt), f"[{b},{m}], {pidxs}, {pidxs_gt}"


    def test1(
            self,
            b=1,
            m=1,
            n=10000,
            max_grid_width=20.,
            max_ray_radius=1.,
            max_grid_size=20,
            seed=1,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        print(f"=======================================")
        print(f"m = {m}, n = {n}, max_gw={max_grid_width}, max_radius={max_ray_radius}, max_gs={max_grid_size}")

        points = (torch.rand(b, n, 3) - 0.5) * 2 *  max_grid_width
        ray_origins = torch.randn(b, m, 3)
        ray_directions = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)
        ray_radius = torch.rand(b) * (max_ray_radius - 0.00001) + 0.00001
        grid_size = (torch.rand(b, 3) * (max_grid_size - 1) + 1).long()
        grid_width = torch.ones(b, 3) * max_grid_width

        self._test(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_width=grid_width,
        )

    def test_ns(self):

        ns = [1000, 10000, 100000, 1000000, 10000000]
        ms = [1] * len(ns)

        for m, n in zip(ms, ns):
            self.test1(
                m=m,
                n=n,
                max_ray_radius=2.,
                max_grid_width=20.,
                max_grid_size=20,
                seed=m,
            )

    def test_ms(self):

        ms = [1000, 10000, 100000]
        ns = [100] * len(ms)

        for m, n in zip(ms, ns):
            self.test1(
                m=m,
                n=n,
                max_ray_radius=2.,
                max_grid_width=20.,
                max_grid_size=20,
                seed=m,
            )

    def test_mns(self):

        ns = [10000, 10000, 100000, 1000000]
        ms = [10000, 10000, 100000, 1000000]

        for m, n in zip(ms, ns):
            self.test1(
                m=m,
                n=n,
                max_ray_radius=6.,
                max_grid_width=20.,
                max_grid_size=20,
                seed=1,#m,
            )


class TestMaxHeap(unittest.TestCase):
    def test(self, b=1, m=1000, n=1000, seed=0):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        values = torch.rand(b, m, n).cuda()
        k = 40
        print(values.device)
        min_k_values = pr_cuda.keep_min_k_values(
            values,
            k
        )
        min_k_values_sorted = min_k_values.sort(dim=-1)[0]
        min_k_values_sorted_gt = values.sort(dim=-1)[0][..., :k]
        assert (min_k_values_sorted == min_k_values_sorted_gt).all()


if __name__ == '__main__':
    unittest.main()
    # TestGridRayIntersection().test(b=10, m=10000, max_size=40, seed=1) // this one fails
    # TestGridRayIntersection().test()
    # TestCollectPointsOnRay().test()
    # TestFindNeighborPointsOfRays().test_mns()
    # TestFindNeighborPointsOfRays().test_ns()
    # TestMaxHeap().test()
