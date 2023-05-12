import unittest
from plib import utils, rigid_motion
import torch
import numpy as np
from timeit import default_timer as timer
import open3d as o3d


class Test_AABB_Intersection(unittest.TestCase):
    def test_1(self):
        N = 10000
        stime = timer()
        for _ in range(N):
            out_dict = utils.ray_aabb_intersection(
                ray_origin=torch.zeros(3),
                ray_direction=torch.tensor([0,0,1.]),
                bbox_min_bounds=torch.tensor([0,0,2.]),
                bbox_max_bounds=torch.tensor([10, 10, 10.]),
            )
        etime = timer()
        print(f'v1 takes {(etime - stime) / N} secs per test')
        assert out_dict['is_intersected']
        assert np.isclose(out_dict['t_near'].detach().cpu().numpy(), 2.)
        assert np.isclose(out_dict['t_far'].detach().cpu().numpy(), 10.)


    def test_2(self):
        N = 10000
        stime = timer()
        for _ in range(N):
            out_dict = utils.ray_aabb_intersection_2(
                ray_origin=torch.zeros(3),
                ray_direction=torch.tensor([0,0,1.]),
                bbox_min_bounds=torch.tensor([0,0,2.]),
                bbox_max_bounds=torch.tensor([10, 10, 10.]),
            )
        etime = timer()
        print(f'v2 takes {(etime - stime) / N} secs per test')
        assert out_dict['is_intersected']
        assert np.isclose(out_dict['t_near'].detach().cpu().numpy(), 2.)
        assert np.isclose(out_dict['t_far'].detach().cpu().numpy(), 10.)

    def test_3(self):
        N = 10000
        stime = timer()
        for _ in range(N):
            out_dict = utils.ray_aabb_intersection(
                ray_origin=torch.zeros(3),
                ray_direction=torch.tensor([0,0,1.]),
                bbox_min_bounds=torch.tensor([1, 1, 2.]),
                bbox_max_bounds=torch.tensor([10, 10, 10.]),
            )
        etime = timer()
        print(f'v1 takes {(etime - stime) / N} secs per test')
        assert out_dict['is_intersected'] == False



class Test_Point_Ray_Distance(unittest.TestCase):
    def test_1(self):
        N = 5
        points = torch.randn(N, 3)
        ray_origins = torch.zeros(1, 3)
        ray_directions = torch.zeros(1, 3)
        ray_directions[0, 2] = 1.

        out_dict = utils.compute_point_ray_distance(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )
        dists = out_dict['dists']
        projections = out_dict['projections']
        ts = out_dict['ts']

        assert dists.shape == (1, N)
        assert projections.shape == (1, N, 3)
        assert ts.shape == (1, N)
        assert torch.allclose(dists, torch.linalg.norm(points[None, :, :2], dim=-1))
        assert torch.allclose(ts.sign(), points[:, 2:3].t().sign())

        projs = torch.zeros(N, 3)
        projs[..., 2] = points[:, 2]
        projs = projs.unsqueeze(0)
        assert torch.allclose(projections, projs)

    def test_chunk(self):
        b_shape = [3, 5]
        n = 10
        m = 7
        points = torch.randn(*b_shape, n, 3)
        ray_origins = torch.randn(*b_shape, m, 3)
        ray_directions = torch.zeros(*b_shape, m, 3)
        ray_directions[..., 2] = 1.

        # standard (no chunking)
        out_dict_gt = utils.compute_point_ray_distance(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )

        # with chunking
        mn = m * n
        for max_chunk_size in [int(1e9), mn//2, mn+1]:
            out_dict = utils.compute_point_ray_distance_in_chunks(
                points=points,
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                max_chunk_size=max_chunk_size,
            )
            for key in out_dict_gt:
                assert torch.allclose(out_dict_gt[key], out_dict[key]), f'{max_chunk_size}'




class Test_Generate_Cam_Rays(unittest.TestCase):

    def test_1(self):
        m = 1
        width_px, height_px = 100, 50
        f = 40.
        cam_poses = rigid_motion.generate_random_camera_poses(
            n=m,
            max_angle=180.,
            min_r=0.5,
            max_r=1.5,
        )
        cam_poses = utils.to_tensor(cam_poses, dtype=torch.float)
        cam_poses = torch.stack(cam_poses, dim=0)  # (m, 4, 4)

        intrinsics = torch.tensor([
            [f, 0., width_px * 0.5],
            [0., f, height_px * 0.5],
            [0, 0, 1],
        ]).expand(m ,3, 3)  # (m, 3, 3)

        ray_origins_w, ray_directions_w = utils.generate_camera_rays(
            cam_poses=cam_poses,
            intrinsics=intrinsics,
            width_px=width_px,
            height_px=height_px,
            subsample=1,
        )

        # o3d ray
        H_w2c = torch.linalg.inv(cam_poses)  # (m, 4, 4)
        o3d_ray_origins = []
        o3d_ray_directions = []
        for i in range(m):
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                intrinsic_matrix=intrinsics[i].numpy(),
                extrinsic_matrix=H_w2c[i].numpy(),
                width_px=width_px,
                height_px=height_px,
            )  # (target_height_px, target_width_px, 6)  [ox, oy, oz, dx, dy, dz]  origin is the pinhole
            rays = torch.from_numpy(rays.numpy())  # (h, w, 6)
            ray_o = rays[:, :, :3]
            ray_d = rays[:, :, 3:]

            o3d_ray_origins.append(ray_o)
            o3d_ray_directions.append(ray_d)
        o3d_ray_origins = torch.stack(o3d_ray_origins, dim=0)  # (m, h, w, 3)
        o3d_ray_directions = torch.stack(o3d_ray_directions, dim=0)  # (m, h, w, 3)
        o3d_ray_directions = o3d_ray_directions / torch.linalg.vector_norm(o3d_ray_directions, dim=-1, keepdims=True)


        assert torch.allclose(o3d_ray_origins, ray_origins_w)
        assert torch.allclose(o3d_ray_directions, ray_directions_w, rtol=1e-4, atol=1e-4)

        diff = torch.linalg.vector_norm(o3d_ray_directions - ray_directions_w, dim=-1)




class Test_KNN(unittest.TestCase):

    def test_chunk(self):
        b_shape = [3, 5]
        n = 10
        m = 7
        k=6
        points = torch.randn(*b_shape, n, 3)
        ray_origins = torch.randn(*b_shape, m, 3)
        ray_directions = torch.zeros(*b_shape, m, 3)
        ray_directions[..., 2] = 1.

        # standard (no chunking)
        out_dict_gt = utils.get_k_neighbor_points(
            points=points,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            k=k,
        )

        # with chunking
        mn = m * n
        for max_chunk_size in [int(1e9), mn//2, mn+1]:
            out_dict = utils.get_k_neighbor_points_in_chunks(
                points=points,
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                k=k,
                max_chunk_size=max_chunk_size,
            )
            for key in out_dict_gt:
                assert torch.allclose(out_dict_gt[key], out_dict[key]), f'{max_chunk_size}'



class Test_Compute_XYZ_W_From_UV(unittest.TestCase):

    def test(self):
        width_px, height_px = 300, 200
        num_cam_poses = 5
        z_map = torch.rand(num_cam_poses, height_px, width_px) * 10 + 0.1  # (m, h, w)

        f = 40.
        cam_poses = rigid_motion.generate_random_camera_poses(
            n=num_cam_poses,
            max_angle=180.,
            min_r=0.5,
            max_r=1.5,
        )
        cam_poses = utils.to_tensor(cam_poses, dtype=torch.float)
        cam_poses = torch.stack(cam_poses, dim=0)  # (m, 4, 4)

        intrinsics = torch.tensor([
            [f, 0., width_px * 0.5],
            [0., f, height_px * 0.5],
            [0, 0, 1],
        ]).expand(num_cam_poses, 3, 3)  # (m, 3, 3)

        # gt
        out_dict = utils.compute_3d_xyz(
            z_map=z_map,
            intrinsic=intrinsics,
            H_c2w=cam_poses,
        )
        xyz_gt = out_dict['xyz_w']  # (m, h, w, 3)
        # print(xyz_gt.shape)

        # new
        u, v = torch.meshgrid(
            torch.arange(0, width_px),
            torch.arange(0, height_px),
            indexing='xy',
        )
        u = u + 0.5
        v = v + 0.5
        # print(f'u.shape = {u.shape}')
        # print(f'v.shape = {v.shape}')
        # print(f'z_map.shape = {v.shape}')

        uv_c = torch.stack([u, v], dim=-1)  # (h, w, 2)
        # z_c = z_map[:, uv_c[..., 1], uv_c[..., 0]]  # (n, h, w)
        z_c = z_map


        uv_c = uv_c.expand(num_cam_poses, -1, -1, -1)  # (n, h, w, 2)
        # print(f'uv_c.shape = {uv_c.shape}')
        # print(f'z_c.shape = {z_c.shape}')

        xyz_w = utils.compute_xyz_w_from_uv(
            uv_c=uv_c,
            z_c=z_c,
            intrinsic=intrinsics,
            H_c2w=cam_poses,
        )
        assert xyz_gt.shape == xyz_w.shape, f'{xyz_gt.shape}  {xyz_w.shape}'
        assert torch.allclose(xyz_gt, xyz_w)


class Test_Sample_Patch(unittest.TestCase):

    def test_1(self):
        b, c, h, w = 2, 1, 5, 6
        bchw = b * c * h * w
        arr = torch.arange(bchw).reshape(b, c, h, w).float()

        center = torch.tensor([
            [1., 1.],
            [1.5, 2.5,]
        ])

        out = utils.sample_patch(
            arr=arr,
            patch_center=center,
            patch_width_px=3,
            patch_width_pitch_scale=1,
        )
        assert out.shape == (b, c, 3, 3)
        assert torch.allclose(out[1], arr[1, :, 1:4, 0:3])

    def test_2(self):
        b, c, h, w = 2, 1, 5, 6
        bchw = b * c * h * w
        arr = torch.arange(bchw).reshape(b, c, h, w).float()

        center = torch.tensor([
            [1., 1.],
            [1.5, 2.5],
        ])

        out = utils.sample_patch(
            arr=arr,
            patch_center=center,
            patch_width_px=3,
            patch_width_pitch_scale=2,
        )
        assert out.shape == (b, c, 3, 3)
        # print(arr)
        # print(out)


if __name__ == '__main__':
    unittest.main()
