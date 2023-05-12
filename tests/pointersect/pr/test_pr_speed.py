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

class TestSpeed(unittest.TestCase):
    def test(
            self,
            b: int = 1,
            n: int = 10000,  # number of points
            m: int = 80000,  # number of rays
            k: int = 40,  # number of neighboring points
            ray_radius: float = 0.1,
            grid_size: int = 100,
            grid_width: float = 1.,
            # pointersect
            num_layers: int = 4,

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

        # # test cpu
        # device = torch.device('cpu')
        # points = points.to(device=device)
        # ray_origins = ray_origins.to(device=device)
        # ray_directions = ray_directions.to(device=device)
        # ray_radius = ray_radius.to(device=device) if isinstance(ray_radius, torch.Tensor) else ray_radius
        # grid_size = grid_size.to(device=device) if isinstance(grid_size, torch.Tensor) else grid_size
        # grid_width = grid_width.to(device=device) if isinstance(grid_width, torch.Tensor) else grid_width
        #
        # stime = timer()
        # all_ray2pidxs = pr_utils.find_neighbor_points_of_rays(
        #     points=points,
        #     ray_origins=ray_origins,
        #     ray_directions=ray_directions,
        #     ray_radius=ray_radius,
        #     grid_size=grid_size,
        #     grid_width=grid_width,
        # )
        # time_pr_old = timer() - stime
        # print(f'pr_old_cpu: {time_pr_old:.3f} secs')


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

        # stime = timer()
        # all_ray2pidxs = pr_utils.find_neighbor_points_of_rays(
        #     points=points,
        #     ray_origins=ray_origins,
        #     ray_directions=ray_directions,
        #     ray_radius=ray_radius,
        #     grid_size=grid_size,
        #     grid_width=grid_width,
        # )
        # time_pr_cuda1 = timer() - stime
        # print(f'pr_old_cuda1: {time_pr_cuda1:.3f} secs')
        #
        # stime = timer()
        # all_ray2pidxs = pr_utils.find_neighbor_points_of_rays(
        #     points=points,
        #     ray_origins=ray_origins,
        #     ray_directions=ray_directions,
        #     ray_radius=ray_radius,
        #     grid_size=grid_size,
        #     grid_width=grid_width,
        # )
        # time_pr_cuda2 = timer() - stime
        # print(f'pr_old_cuda2: {time_pr_cuda2:.3f} secs')

        torch.cuda.empty_cache()

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
        print('after first run of find_k_neighbor_points_of_rays')
        torch.cuda.empty_cache()
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')

        print(f'ray_origins.shape = {ray_origins.shape}')
        print(f'ray_origins = {ray_origins}')

        print(f'points.shape = {points.shape}')
        print(f'points = {points}')


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
        time_pr_k_cuda2 = timer() - stime
        print(f'pr_k_cuda_2: {time_pr_k_cuda2:.3f} secs')

        torch.cuda.empty_cache()
        print('after second run of find_k_neighbor_points_of_rays')
        torch.cuda.empty_cache()
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')

        stime = timer()
        pr_point_idxs = all_ray2pidxs['ray2pidx_heap']
        neighbor_num = all_ray2pidxs['ray_neighbor_num']
        i_shape = list(pr_point_idxs.shape)
        # pr_point_idxs: (b, m, k)= 1 * 20000 * 40 * 8 = 6400000 byte = 6.4 MB
        # neighbor_num: (b, m) = 1 * 20000 * 8 = 0.16 MB

        pr_points = torch.gather(
            input=points.unsqueeze(-3).expand(b, m, n, 3),  # (b, m, n, 3)
            dim=-2,
            index=pr_point_idxs.unsqueeze(-1).expand(*(i_shape + [3])),  # (b, m, k, 3)
        )  # (b, m, k, 3)  neighbor_points[b, m, i] = xyz_w[b, m, neighbor_xyz_w_idxs[b, m, i]]
        time_gather = timer() - stime
        print(f'gather: {time_gather:.3f} secs')

        # print(all_ray2pidxs)
        # print(all_ray2pidxs[0].shape)
        # print(all_ray2pidxs[1].shape)

        # run pointersect
        net = SimplePointersect(
            learn_dist=False,
            num_layers=num_layers,
            dim_feature=512,
            num_heads=4,
            positional_encoding_num_functions=0,
            positional_encoding_include_input=True,
            positional_encoding_log_sampling=True,
            nonlinearity='silu',
            dim_mlp=512,
            encoding_type='pos',  # currently only support classic positional encoding
            dropout=0.1,
            direction_param='norm_vec',  # 'theta_phi'
            estimate_surface_normal_weights=False,
            estimate_image_rendering_weights=True,
            use_layer_norm=False,
            dim_point_feature=0,  # additional feature description of points (other than xyz)
            use_rgb_as_input=True,
            use_dist_as_input=False,  # if true, use |x|,|y|,|z| and sqrt(x^2+y^2) in ray space as input
            use_zdir_as_input=False,  # if true, use camera viewing direction (2 vector, 3 dim) as input
            use_dps_as_input=False,   # if true, use local frame width (1 value, 1 dim) as input
            use_dpsuv_as_input=False, # if true, use local frame (2 vectors, 6 dim) as input
            use_pr=True, # if true, learn a token to replace invalid input
        )

        net.eval()
        net.to(device=device)

        torch.set_grad_enabled(False)
        # neighbor_num = torch.ones(b, m, dtype=torch.long, device=device) * k

        torch.cuda.empty_cache()
        print('before first run of pointersect')
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')

        stime = timer()
        net(
            points=pr_points,
            additional_features=None,
            neighbor_num=neighbor_num,
            printout=False,
            max_chunk_size=-1,
        )
        time_pointersect1 = timer() - stime
        print(f'pointersect1: {time_pointersect1:.3f} secs')

        torch.cuda.empty_cache()
        print('after first run of pointersect')
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')

        stime = timer()
        net(
            points=pr_points,
            additional_features=None,
            neighbor_num=neighbor_num,
            printout=False,
            max_chunk_size=40000,
        )
        time_pointersect2 = timer() - stime
        print(f'pointersect2: {time_pointersect2:.3f} secs')

        torch.cuda.empty_cache()
        print('after second run of pointersect')
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')

        stime = timer()
        net(
            points=pr_points,
            additional_features=None,
            neighbor_num=neighbor_num,
            printout=False,
            max_chunk_size=40000,
        )
        time_pointersect3 = timer() - stime
        print(f'pointersect3: {time_pointersect3:.3f} secs')

        torch.cuda.empty_cache()
        print('after third run of pointersect')
        print(f'cuda memory reserved: {torch.cuda.memory_reserved(0) / 1.e6} MB')
        print(f'cuda memory allocated: {torch.cuda.memory_allocated(0) / 1.e6} MB')


"""
Results:
compiling pr_cpp..
compiling pr_cuda..
pr_old_cuda1: 0.167 secs
pr_old_cuda2: 0.153 secs
device: cuda:0
pr_k_cuda_1: 0.042 secs
device: cuda:0
pr_k_cuda_2: 0.040 secs
gather: 0.000 secs
invalid learned token norm is 21.9189510345459
pointersect1: 1.442 secs
invalid learned token norm is 21.9189510345459
pointersect2: 0.454 secs
invalid learned token norm is 21.9189510345459
pointersect3: 0.454 secs
"""

if __name__ == '__main__':
    unittest.main()
