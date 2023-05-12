import unittest
from pointersect.pr import naive
import torch
import typing as T
import numpy as np
import random
from timeit import default_timer as timer

class TestGetGridIdx(unittest.TestCase):
    def _test(
            self,
            b: int,
            n: int,
            grid_size,
            center,
            grid_width,
    ):
        points = torch.rand(b, n, 3)
        gidx, valid_mask = naive.get_grid_idx(
            points=points,
            grid_size=grid_size,
            center=center,
            grid_width=grid_width,
        )
        if isinstance(grid_size, int):
            total_cells = grid_size * grid_size * grid_size
        else:
            total_cells = torch.prod(grid_size, dim=-1, keepdim=True)  # (*, 1)
        torch.prod(gidx)
        assert torch.logical_or(gidx >= 0, torch.logical_not(valid_mask)).all()
        assert torch.logical_or(gidx < total_cells, torch.logical_not(valid_mask)).all()


    def test1(self):
        self._test(
            b=3,
            n=10,
            grid_size=5,
            center=0.,
            grid_width=1.,
        )

    def test2(self):
        b = 3
        self._test(
            b=b,
            n=10,
            grid_size=(torch.rand(b, 3) * 10).long() + 1,
            center=torch.randn(b, 3),
            grid_width=torch.rand(b, 3),
        )



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
            # include_outside: bool = True,
    ):
        batch_size, n_rays, _ = ray_origins.shape

        stime = timer()
        all_ray2pidxs = naive.find_neighbor_points_of_rays(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_center=grid_center,
            grid_width=grid_width,
            # include_outside=include_outside,
        )
        total_time = timer() - stime

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
        total_time_gt = timer() - stime
        print(f'new/gt = {total_time:1f}/{total_time_gt:1f} = {total_time/total_time_gt*100:.3f}%')

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
                assert len(pidxs) == len(pidxs_gt), f"[{b},{m}], {pidxs}, {pidxs_gt}"
                assert np.allclose(pidxs, pidxs_gt), f"[{b},{m}], {pidxs}, {pidxs_gt}"

    def test(
            self,
            b=10,
            m=7,
            n=20,
            max_grid_size=5,
            seed=1,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        points = torch.randn(b, n, 3)
        ray_origins = torch.randn(b, m, 3)
        ray_directions = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)
        ray_radius = torch.rand(b) + 0.01
        grid_size = (torch.rand(b, 3) * (max_grid_size - 1) + 1).long()

        self._test(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            # include_outside=include_outside,
        )

    def test2(self):
        max_b = 100
        max_n = 50
        max_m = 50
        max_grid_size=30
        for i in range(10):
            print(f'{i}')
            seed = i
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            self.test(
                b=np.random.randint(max_b) + 1,
                n=np.random.randint(max_n) + 1,
                m=np.random.randint(max_m) + 1,
                max_grid_size=np.random.randint(max_grid_size) + 1,
                seed=seed,
            )


if __name__ == '__main__':
    unittest.main()
