# The script tests how many rays and points can pr handles

import unittest
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
from pointersect.models.pointersect import SimplePointersect

class TestLimit(unittest.TestCase):
    def _test(
            self,
            b: int = 1,
            n: int = 10000,  # number of points
            m: int = 20000,  # number of rays
            k: int = 40,  # number of neighboring points
            ray_radius: float = 0.1,
            grid_size: int = 100,
            grid_width: float = 1.,
    ):
        print('very beginning')
        torch.cuda.empty_cache()
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0)/1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0)/1.e6} MB')

        points = (torch.rand(b, n, 3) - 0.5) * 2 * grid_width
        ray_origins = torch.randn(b, m, 3)
        ray_directions = torch.nn.functional.normalize(torch.randn(b, m, 3), dim=-1)
        ray_radius = torch.ones(b) * ray_radius
        grid_size = torch.ones(b, 3, dtype=torch.long) * grid_size
        grid_width = torch.ones(b, 3) * grid_width


        if not torch.cuda.is_available():
            return

        # test gpu
        device = torch.device('cuda')
        points = points.to(device=device)
        ray_origins = ray_origins.to(device=device)
        ray_directions = ray_directions.to(device=device)
        ray_radius = ray_radius.to(device=device) if isinstance(ray_radius, torch.Tensor) else ray_radius
        grid_size = grid_size.to(device=device) if isinstance(grid_size, torch.Tensor) else grid_size
        grid_width = grid_width.to(device=device) if isinstance(grid_width, torch.Tensor) else grid_width

        print('send to gpu')
        torch.cuda.empty_cache()
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')

        stime = timer()
        all_ray2pidxs = pr_utils.find_k_neighbor_points_of_rays(
            points=points,
            k=k,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            ray_radius=ray_radius,
            grid_size=grid_size,
            grid_width=grid_width,
            version='v2',
        )
        time_pr_k_cuda1 = timer() - stime
        print(f'pr_k_cuda_1: {time_pr_k_cuda1:.3f} secs')

        torch.cuda.empty_cache()
        print('after find_k_neighbor_points_of_rays')
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')


    def test(self):

        # ns = [1e4, 1e5, 1e6, 1e7]
        # ms = [1e4, 1e5, 1e6, 1e7]

        ns = [1e7]
        ms = [1e7]

        for n in ns:
            for m in ms:
                print(f'n = {n}, m = {m} -------------------------')
                self._test(
                    b=1,
                    n=int(n),  # number of points
                    m=int(m),  # number of rays
                    k=40,  # number of neighboring points
                    ray_radius=0.1,
                    grid_size=100,
                    grid_width=1.,
                )
                print('')

# results
"""
n = 10000.0, m = 10000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 2.097152 MB
cuda memory allocated: 0.362496 MB
beginning
GPU 0 memory: used=846 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=866 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=866 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1162 MB, total=40536.2 MB
pr_k_cuda_1: 0.013 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 23.068672 MB
cuda memory allocated: 3.64288 MB

n = 10000.0, m = 100000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 23.068672 MB
cuda memory allocated: 2.522112 MB
beginning
GPU 0 memory: used=1162 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1162 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1182 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1214 MB, total=40536.2 MB
pr_k_cuda_1: 0.003 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 56.623104 MB
cuda memory allocated: 35.322368 MB

n = 10000.0, m = 1000000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 27.262976 MB
cuda memory allocated: 25.28768 MB
beginning
GPU 0 memory: used=1166 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1186 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1186 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1686 MB, total=40536.2 MB
pr_k_cuda_1: 0.003 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 369.098752 MB
cuda memory allocated: 354.151936 MB

n = 10000.0, m = 10000000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 245.366784 MB
cuda memory allocated: 240.121856 MB
beginning
GPU 0 memory: used=1374 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1394 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1394 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=6362 MB, total=40536.2 MB
pr_k_cuda_1: 0.008 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 3527.409664 MB
cuda memory allocated: 3520.375808 MB

n = 100000.0, m = 10000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 23.068672 MB
cuda memory allocated: 1.442304 MB
beginning
GPU 0 memory: used=1162 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1162 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1182 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1184 MB, total=40536.2 MB
pr_k_cuda_1: 0.003 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 25.165824 MB
cuda memory allocated: 4.722688 MB

n = 100000.0, m = 100000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 23.068672 MB
cuda memory allocated: 3.60192 MB
beginning
GPU 0 memory: used=1162 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1162 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1182 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1216 MB, total=40536.2 MB
pr_k_cuda_1: 0.004 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 58.720256 MB
cuda memory allocated: 36.402176 MB

n = 100000.0, m = 1000000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 35.651584 MB
cuda memory allocated: 25.784832 MB
beginning
GPU 0 memory: used=1174 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1174 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1194 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1694 MB, total=40536.2 MB
pr_k_cuda_1: 0.004 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 377.48736 MB
cuda memory allocated: 354.649088 MB

n = 100000.0, m = 10000000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 266.338304 MB
cuda memory allocated: 241.201664 MB
beginning
GPU 0 memory: used=1394 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1394 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1414 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=6382 MB, total=40536.2 MB
pr_k_cuda_1: 0.014 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 3548.381184 MB
cuda memory allocated: 3521.455616 MB

n = 1000000.0, m = 10000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 14.680064 MB
cuda memory allocated: 12.825088 MB
beginning
GPU 0 memory: used=1154 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1174 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1194 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1194 MB, total=40536.2 MB
pr_k_cuda_1: 0.004 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 35.651584 MB
cuda memory allocated: 16.105472 MB

n = 1000000.0, m = 100000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 35.651584 MB
cuda memory allocated: 14.984704 MB
beginning
GPU 0 memory: used=1174 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1174 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1214 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1248 MB, total=40536.2 MB
pr_k_cuda_1: 0.009 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 71.303168 MB
cuda memory allocated: 47.78496 MB

n = 1000000.0, m = 1000000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 39.845888 MB
cuda memory allocated: 37.750272 MB
beginning
GPU 0 memory: used=1178 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1198 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1218 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1718 MB, total=40536.2 MB
pr_k_cuda_1: 0.007 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 381.681664 MB
cuda memory allocated: 366.614528 MB

n = 1000000.0, m = 10000000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 257.949696 MB
cuda memory allocated: 252.584448 MB
beginning
GPU 0 memory: used=1386 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1406 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1426 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=6394 MB, total=40536.2 MB
pr_k_cuda_1: 0.019 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 3539.992576 MB
cuda memory allocated: 3532.8384 MB

n = 10000000.0, m = 10000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 123.731968 MB
cuda memory allocated: 120.242176 MB
beginning
GPU 0 memory: used=1258 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1356 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1454 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1454 MB, total=40536.2 MB
pr_k_cuda_1: 0.004 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 144.703488 MB
cuda memory allocated: 123.52256 MB

n = 10000000.0, m = 100000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 144.703488 MB
cuda memory allocated: 122.83648 MB
beginning
GPU 0 memory: used=1278 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1356 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1454 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1486 MB, total=40536.2 MB
pr_k_cuda_1: 0.002 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 178.25792 MB
cuda memory allocated: 155.636736 MB

n = 10000000.0, m = 1000000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 148.897792 MB
cuda memory allocated: 145.16736 MB
beginning
GPU 0 memory: used=1282 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1380 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1478 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=1970 MB, total=40536.2 MB
pr_k_cuda_1: 0.009 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 490.733568 MB
cuda memory allocated: 474.031616 MB

n = 10000000.0, m = 10000000.0 -------------------------
very beginning
cuda memory reserved: 0.0 MB
cuda memory allocated: 0.0 MB
send to gpu
cuda memory reserved: 367.0016 MB
cuda memory allocated: 360.001536 MB
beginning
GPU 0 memory: used=1490 MB, total=40536.2 MB
after get_grid_idx_cuda
GPU 0 memory: used=1588 MB, total=40536.2 MB
after gather_points_cuda_v2
GPU 0 memory: used=1686 MB, total=40536.2 MB
after collect_k_points_on_ray_cuda
GPU 0 memory: used=6654 MB, total=40536.2 MB
pr_k_cuda_1: 0.019 secs
after find_k_neighbor_points_of_rays
cuda memory reserved: 3649.04448 MB
cuda memory allocated: 3640.255488 MB

.
----------------------------------------------------------------------
Ran 1 test in 122.189s

OK
"""


if __name__ == '__main__':
    unittest.main()
