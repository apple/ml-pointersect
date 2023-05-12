import sys
sys.path.append('cdslib')
import unittest
from pointersect.pr import pr_utils, naive
import torch
import typing as T
import numpy as np
import random
from timeit import default_timer as timer
from plib import utils
import open3d as o3d

class TestFindNeighborPoints(unittest.TestCase):

    def _test(
            self,
            type: str,  # 'cpp' or 'cuda'
            points: torch.Tensor,  # (b, n, 3)
            ray_origins: torch.Tensor,  # (b, m, 3)
            ray_directions: torch.Tensor,  # (b, m, 3)
            ray_radius: T.Union[torch.Tensor, float],  # (b,)
            grid_size: T.Union[torch.Tensor, int],  # (b, 3)
            grid_center: T.Union[torch.Tensor, float, None] = 0.,  # (b, 3)
            grid_width: T.Union[torch.Tensor, float, None] = 1.,  # (b, 3)
            k: int = 40,
    ):
        if type == 'cpp':
            device = torch.device('cpu')
            assert pr_utils.pr_cpp_loaded, f'pr_cpp is not loaded'
        elif type == 'cuda':
            if not torch.cuda.is_available():
                print('no available gpu, pass')
                return
            device = torch.device('cuda')
            assert pr_utils.pr_cuda_loaded, f'pr_cuda is not loaded'
        else:
            raise NotImplementedError


        batch_size, n_rays, _ = ray_origins.shape

        points = points.to(device=device)
        ray_origins = ray_origins.to(device=device)
        ray_directions = ray_directions.to(device=device)
        ray_radius = ray_radius.to(device=device) if isinstance(ray_radius, torch.Tensor) else ray_radius
        grid_size = grid_size.to(device=device) if isinstance(grid_size, torch.Tensor) else grid_size
        grid_center = grid_center.to(device=device) if isinstance(grid_center, torch.Tensor) else grid_center
        grid_width = grid_width.to(device=device) if isinstance(grid_width, torch.Tensor) else grid_width

        stime = timer()
        all_ray2pidxs_gt = naive.find_neighbor_points_of_rays_brute_force(
        #all_ray2pidxs_gt = naive.find_neighbor_points_of_rays(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_center=grid_center,
            grid_width=grid_width,
        )
        total_time_gt = timer() - stime

        stime = timer()
        all_ray2pidxs = pr_utils.find_neighbor_points_of_rays(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_center=grid_center,
            grid_width=grid_width,
        )
        total_time = timer() - stime
        print(f'{type}/gt = {total_time:1f}/{total_time_gt:1f} = {total_time / total_time_gt * 100:.3f}%')

        # check v2
        assert len(all_ray2pidxs) == batch_size
        assert len(all_ray2pidxs_gt) == batch_size

        for b in range(batch_size):
            ray2pidxs = all_ray2pidxs[b]
            ray2pidxs_gt = all_ray2pidxs_gt[b]
            assert len(ray2pidxs) == n_rays
            assert len(ray2pidxs_gt) == n_rays

            for m in range(n_rays):
                pidxs = ray2pidxs[m]
                pidxs_gt = ray2pidxs_gt[m]
                pidxs = np.sort(np.array(pidxs))
                pidxs_gt = np.sort(np.array(pidxs_gt))

                # for debugging a particular case, not used anymore
                # if m == 4:
                #     bug_ray_o = ray_origins[[0],[4]]
                #     bug_ray_d = ray_directions[[0],[4]]
                #     bug_point = points[[0],[1186]]
                #     dist_dict = utils.compute_point_ray_distance_in_chunks(
                #         points=bug_point,
                #         ray_origins=bug_ray_o,
                #         ray_directions=bug_ray_d,
                #     )
                #     print(1)

                assert len(pidxs) == len(pidxs_gt), f"[{b},{m}], {pidxs}, {pidxs_gt}"
                assert np.allclose(pidxs, pidxs_gt), f"[{b},{m}], {pidxs}, {pidxs_gt}"

        del all_ray2pidxs_gt
        del all_ray2pidxs

        # check getting k neighbor points only
        if type == 'cpp':
            print('not implement finding k neighbors in cpp')
            return

        print('before brute force')
        torch.cuda.empty_cache()
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')

        # k = 40
        stime = timer()
        all_ray2kpidx_gt, neighbor_num_gt = naive.find_k_neighbor_points_of_rays_brute_force(
            points=points,
            k=k,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_center=grid_center,
            grid_width=grid_width,
        )
        total_time_gt = timer() - stime

        # release memory to avoid affecting performance
        all_ray2kpidx_gt = all_ray2kpidx_gt.detach().cpu()
        neighbor_num_gt = neighbor_num_gt.detach().cpu()

        print('after brute force (released results)')
        torch.cuda.empty_cache()
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')

        # v1
        stime = timer()
        out_dict = pr_utils.find_k_neighbor_points_of_rays(
            points=points,
            k=k,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_center=grid_center,
            grid_width=grid_width,
            version='v1',
        )
        all_ray2kpidx_heap = out_dict['ray2pidx_heap']
        neighbor_num = out_dict['ray_neighbor_num']
        total_time = timer() - stime
        print(f'{type}_k_v1/gt = {total_time:1f}/{total_time_gt:1f} = {total_time / total_time_gt * 100:.3f}%')

        all_ray2kpidx_heap =  all_ray2kpidx_heap.detach().cpu()
        neighbor_num = neighbor_num.detach().cpu()

        assert( (neighbor_num  == neighbor_num_gt).all())

        neighbor_idx = torch.arange(k, device=neighbor_num.device).unsqueeze(0).repeat(batch_size, n_rays, 1)
        invalid_mask = neighbor_idx >= neighbor_num.unsqueeze(-1)

        all_ray2kpidx_heap[invalid_mask] = int(1e10)
        all_ray2kpidx_heap, _ = torch.sort(all_ray2kpidx_heap, dim=-1)  # (*, m, n), (*, m, n)
        all_ray2kpidx_gt[invalid_mask] = int(1e10)
        all_ray2kpidx_gt, _ = torch.sort(all_ray2kpidx_gt, dim=-1)  # (*, m, n), (*, m, n)
        # if points contains repeat ones, assertion could fail
        assert( (all_ray2kpidx_heap == all_ray2kpidx_gt).all())

        #diff_id =  torch.where((all_ray2kpidx_heap[0] != all_ray2kpidx_gt[0]).any(dim=-1))[0]
        #torch.cat([all_ray2kpidx_heap, all_ray2kpidx_gt])[:,diff_id,:].transpose(0,1)

        print('after v1 (released results)')
        torch.cuda.empty_cache()
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')

        # v2 implementation
        stime = timer()
        out_dict = pr_utils.find_k_neighbor_points_of_rays(
            points=points,
            k=k,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_center=grid_center,
            grid_width=grid_width,
            version='v2',
        )
        total_time = timer() - stime
        print(f'{type}_k_v2/gt = {total_time:1f}/{total_time_gt:1f} = {total_time / total_time_gt * 100:.3f}%')
        all_ray2kpidx_heap = out_dict['ray2pidx_heap']
        neighbor_num = out_dict['ray_neighbor_num']

        all_ray2kpidx_heap = all_ray2kpidx_heap.detach().cpu()
        neighbor_num = neighbor_num.detach().cpu()

        assert ((neighbor_num == neighbor_num_gt).all())

        neighbor_idx = torch.arange(k, device=neighbor_num.device).unsqueeze(0).repeat(batch_size, n_rays, 1)
        invalid_mask = neighbor_idx >= neighbor_num.unsqueeze(-1)

        all_ray2kpidx_heap[invalid_mask] = int(1e10)
        all_ray2kpidx_heap, _ = torch.sort(all_ray2kpidx_heap, dim=-1)  # (*, m, n), (*, m, n)
        all_ray2kpidx_gt[invalid_mask] = int(1e10)
        all_ray2kpidx_gt, _ = torch.sort(all_ray2kpidx_gt, dim=-1)  # (*, m, n), (*, m, n)
        # if points contains repeat ones, assertion could fail
        assert ((all_ray2kpidx_heap == all_ray2kpidx_gt).all())

        print('after v2 (released results)')
        torch.cuda.empty_cache()
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')

    def test_cpp(
            self,
            b=1,
            m=1,
            n=100000,
            max_grid_width=20.,
            max_ray_radius=1.,
            max_grid_size=100,
            seed=1,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        points = torch.rand(b, n, 3) * max_grid_width - max_grid_width / 2
        ray_origins = torch.randn(b, m, 3)
        ray_directions = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)
        ray_radius = torch.rand(b) * (max_ray_radius - 0.00001) + 0.00001
        grid_size = (torch.rand(b, 3) * (max_grid_size - 1) + 1).long()

        self._test(
            type='cpp',
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
        )

    # problem: when m=n=10000, some result will not match
    # max_grid_width=20.,
    # max_ray_radius=6.,
    # max_grid_size=20,
    # seed=1,
    # one happens in batch 0, ray 4, naive version include an addition id 1186
    def test_cuda(
            self,
            b=1,
            m=20000,
            n=20000,
            max_grid_width=20.,
            max_ray_radius=1.,
            max_grid_size=20,
            seed=0,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        points = (torch.rand(b, n, 3) - 0.5) * 2 * (max_grid_width * 0.95)
        ray_origins = torch.randn(b, m, 3)
        ray_directions = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)
        ray_radius = torch.rand(b) * (max_ray_radius - 0.01) + 0.01
        grid_size = (torch.rand(b, 3) * (max_grid_size - 1) + 1).long()
        grid_width = torch.ones(b, 3) * max_grid_width

        self._test(
            type='cuda',
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_width=grid_width,
        )


if __name__ == '__main__':
    unittest.main()
