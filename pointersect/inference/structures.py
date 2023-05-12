#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# The file implements the basic data containers.
import json
import sys
import warnings

sys.path.append('cdslib')
import os
import imageio
import torch
import typing as T
from plib import render, rigid_motion, utils, mesh_utils
import open3d as o3d
import numpy as np
import math
import shutil
from scipy.spatial.transform import Rotation
import torch_scatter
import skimage
from plib import sample_utils
import kornia
import pyexr
import scipy as sp
import scipy.signal

INF = 1e12


class PointCloud:
    def __init__(
            self,
            xyz_w: torch.Tensor,  # (b, n, 3)
            rgb: T.Optional[torch.Tensor] = None,  # (b, n, 3)
            normal_w: T.Optional[torch.Tensor] = None,  # (b, n, 3)  vertex normal in world coordinate
            captured_z_direction_w: T.Optional[torch.Tensor] = None,
            # (b, n, 3)  # (z axis of the camera in world coordinate)
            captured_dps: T.Optional[torch.Tensor] = None,  # (b, n, 1)  distance per sample
            captured_dps_u_w: T.Optional[torch.Tensor] = None,
            # (b, n, 3)  distance per sample in camera x direction in the word coord
            captured_dps_v_w: T.Optional[torch.Tensor] = None,
            # (b, n, 3)  distance per sample in camera y direction in the word coord
            captured_view_direction_w: T.Optional[torch.Tensor] = None,
            # (b, n, 3)  # unit vector pointing from capturing camera pinhole to the point
            valid_mask: T.Optional[torch.Tensor] = None,  # (b, n, 1)
            included_point_at_inf: bool = False,  # whether n==0 represents point at inf
            feature: T.Optional[torch.Tensor] = None,  # (b, n, f)
            img_idxs: T.Optional[torch.Tensor] = None,  # (b, n, 1)  linear index of qhw in the original rgbd image
    ):
        self.xyz_w = xyz_w
        self.rgb = rgb
        self.normal_w = normal_w
        self.captured_z_direction_w = captured_z_direction_w
        self.captured_view_direction_w = captured_view_direction_w
        self.captured_dps = captured_dps
        self.captured_dps_u_w = captured_dps_u_w
        self.captured_dps_v_w = captured_dps_v_w
        self.valid_mask = valid_mask
        self.feature = feature
        self.img_idxs = img_idxs

        self.attr_names = [
            'xyz_w', 'rgb', 'captured_z_direction_w', 'captured_view_direction_w',
            'captured_dps', 'captured_dps_u_w', 'captured_dps_v_w', 'normal_w',
            'valid_mask', 'feature', 'img_idxs',
        ]

        self.included_point_at_inf = included_point_at_inf
        self.check_dim()

    @staticmethod
    def from_o3d_pcd(
            o3d_pcd: o3d.geometry.PointCloud,
    ) -> 'PointCloud':
        """
        Return a point cloud object from o3d_pcd
        Args:
            o3d_pcd:
                (n,)
        Returns:
            (b=1, n)
        """
        xyz_w = torch.from_numpy(np.array(o3d_pcd.points)).float().unsqueeze(0)  # (1, n, 3)
        if o3d_pcd.has_normals():
            normals = torch.from_numpy(np.array(o3d_pcd.normals)).float().unsqueeze(0)  # (1, n, 3)
        else:
            normals = None
        if o3d_pcd.has_colors():
            colors = torch.from_numpy(np.array(o3d_pcd.colors)).float().unsqueeze(0)  # (1, n, 3)
            assert colors.ndim == 3
            assert colors.size(-1) == 3
        else:
            colors = None
        return PointCloud(
            xyz_w=xyz_w,  # (1, n, 3)
            rgb=colors,
            normal_w=normals,
        )

    def get_num_points(self) -> int:
        """
        get number of points (excluding point at inf)
        but including the invalid points.
        """
        if self.included_point_at_inf:
            return self.xyz_w.size(1) - 1
        else:
            return self.xyz_w.size(1)

    def get_num_valid_points(self, bidx: int) -> int:
        """
        get number of valid points (excluding point at inf)
        but including the invalid points.
        """
        if self.valid_mask is None:
            if self.included_point_at_inf:
                return self.xyz_w.size(1) - 1
            else:
                return self.xyz_w.size(1)
        else:
            return self.valid_mask[bidx, :].sum().detach().cpu().item()

    def to(self, device: torch.device) -> 'PointCloud':
        for attr_name in self.attr_names:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.to(device=device))
        return self

    def detach(self) -> 'PointCloud':
        for attr_name in self.attr_names:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.detach())
        return self

    def clone(self) -> 'PointCloud':
        data_dict = dict()
        for attr_name in self.attr_names:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                data_dict[attr_name] = arr.clone()
            else:
                data_dict[attr_name] = None
        data_dict['included_point_at_inf'] = self.included_point_at_inf
        return PointCloud(**data_dict)

    def drop_features(self, drop_normal: bool = True):
        """Drop features related to camera pose used to capture the point"""
        for attr_name in [
            'captured_z_direction_w', 'captured_view_direction_w',
            'captured_dps', 'captured_dps_u_w', 'captured_dps_v_w',
        ]:
            setattr(self, attr_name, None)

        if drop_normal:
            setattr(self, 'normal_w', None)

    def check_dim(self):
        """Check all attributes have the same number of points."""
        n = self.xyz_w.size(1)
        for name in self.attr_names:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            else:
                assert arr.size(1) == n, f'{name}.shape = {arr.shape}, num_points not {n}'

    def insert_point_at_inf(self):
        """
        insert a point representing inf at n=0
        """
        if self.included_point_at_inf:
            return

        b = self.xyz_w.size(0)
        # when using pr to find k points within a fixed distance of ray
        # use a far-away point (1e12) to represent that the point is not found
        # this far-away point will be replaced by a learned token in the model later on

        for name in self.attr_names:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            ndim = arr.shape[2:]
            if name == 'xyz_w':
                val = INF
            elif name == 'valid_mask':
                val = 0
            else:
                val = 0

            arr_requires_grad = arr.requires_grad

            arr = torch.cat(
                (
                    (torch.ones(b, 1, *ndim, dtype=arr.dtype, device=arr.device) * val).to(dtype=arr.dtype),
                    arr,
                ),
                dim=1,
            )
            arr.requires_grad = arr_requires_grad
            setattr(self, name, arr)

        self.included_point_at_inf = True
        self.check_dim()

    def reset_point_at_inf(self):
        """
        reset the point representing inf at n=0
        """
        if not self.included_point_at_inf:
            return

        with torch.no_grad():
            for name in self.attr_names:
                arr = getattr(self, name, None)
                if arr is None:
                    continue

                if name == 'xyz_w':
                    val = INF
                elif name == 'valid_mask':
                    val = 0
                else:
                    val = 0

                arr[:, 0, :] = val
                setattr(self, name, arr)
        self.check_dim()

    def remove_point_at_inf(self):
        """
        remove the point representing inf at n=0
        """
        if not self.included_point_at_inf:
            return

        b = self.xyz_w.size(0)
        for name in self.attr_names:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            arr_requires_grad = arr.requires_grad
            arr = arr[:, 1:]
            arr.requires_grad = arr_requires_grad
            setattr(self, name, arr)
        self.included_point_at_inf = False
        self.check_dim()

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""
        to_save = dict()
        for name in self.attr_names:
            to_save[name] = getattr(self, name, None)
        to_save['included_point_at_inf'] = self.included_point_at_inf
        return to_save

    def load_state_dict(
            self,
            state_dict: T.Dict[str, T.Any],
    ):
        """Load the state dictionary."""
        for name in self.attr_names:
            setattr(self, name, state_dict.get(name, None))
        setattr(self, 'included_point_at_inf', state_dict.get('included_point_at_inf', False))

    def extract_valid_attr(self, arr: torch.Tensor, bidx: int) -> T.Union[torch.Tensor, None]:
        """
        Args:
            arr:
                (b, n, *)
            bidx:
                batch index
        Returns:
            (n, dim)
        """

        if arr is None:
            return None

        if self.valid_mask is None:
            if not self.included_point_at_inf:
                return arr[bidx]  # (n, dim)
            else:
                return arr[bidx, 1:]  # (n, dim)
        else:
            if self.valid_mask.dtype == torch.bool:
                valid_mask = self.valid_mask
            else:
                valid_mask = self.valid_mask > 0.5
            if self.included_point_at_inf:
                assert (valid_mask[bidx, 0] < 0.5).all()
            arr = arr[bidx, valid_mask[bidx, :, 0]]  # (n, dim)
            return arr

    def extract_valid_point_cloud(self, bidx: int) -> 'PointCloud':
        """
        Return a new PointCloud `(1, n)` that contains only the valid points.
        Args:
            bidx:

        Returns:
            new_point_cloud: (b=1, n)
        """
        data_dict = dict()
        for attr_name in self.attr_names:
            arr = getattr(self, attr_name, None)
            if arr is None:
                data_dict[attr_name] = None
            else:
                arr = self.extract_valid_attr(arr=arr, bidx=bidx).unsqueeze(0)
                data_dict[attr_name] = arr.clone()
        new_point_cloud = PointCloud(**data_dict)  # include_inf = False
        assert not new_point_cloud.included_point_at_inf
        return new_point_cloud

    def get_o3d_pcds(
            self,
            estimate_normal_if_not_exist: bool = False,
    ) -> T.List[o3d.geometry.PointCloud]:
        """Return a list of b o3d pcds containing xyz and rgb."""

        o3d_pcds = []
        for i in range(self.xyz_w.size(0)):

            xyz_w = self.extract_valid_attr(
                arr=self.xyz_w,
                bidx=i,
            )  # (n, 3)
            if xyz_w is not None:
                xyz_w = xyz_w.detach().cpu().numpy()  # (n, 3)
            rgb = self.extract_valid_attr(
                arr=self.rgb,
                bidx=i,
            )  # (n, 3)
            if rgb is not None:
                rgb = rgb.detach().cpu().numpy()  # (n, 3)
            normal_w = self.extract_valid_attr(
                arr=self.normal_w,
                bidx=i,
            )  # (n, 3)
            if normal_w is not None:
                normal_w = normal_w.detach().cpu().numpy()  # (n, 3)

            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(xyz_w)  # (n, 3)
            if rgb is not None:
                o3d_pcd.colors = o3d.utility.Vector3dVector(rgb)  # (n, 3)
            if normal_w is not None:
                o3d_pcd.normals = o3d.utility.Vector3dVector(normal_w)  # (n, 3)

            if estimate_normal_if_not_exist and normal_w is None:
                o3d_pcd.estimate_normals()  # default parameter (number of neighbor points = 30), random direction

            o3d_pcds.append(o3d_pcd)
        return o3d_pcds

    def get_mesh(
            self,
            bidx: int,
            method: str,
            recompute_normal: bool,
            alpha: float = 0.03,
            ball_radii: T.List[float] = (0.005, 0.01, 0.02, 0.04),
            poisson_depth: int = 8,
    ) -> 'Mesh':
        """
        Reconstruct a mesh from the point cloud.

        Args:
            bidx:
                the batch index.
            method:
                'poisson': use Poisson surface reconstruction
                'alpha': use Alpha shapes
                'ball': use Ball pivoting
            recompute_normal:
                whether to recompute the vertex normal (assuming no normal at the points)
            alpha:
                alpha value used by alpha shapes
            ball_radii:
                radii used by ball pivot.
                radii of the individual balls that are pivoted on the point cloud.
                When the ball touches three points, a triangle is created.
            poisson_depth:
                depth used by poisson reconstruction.
                A higher depth value means a mesh with more details.
        """

        o3d_pcd: o3d.geometry.PointCloud = self.get_o3d_pcds(
            estimate_normal_if_not_exist=True,
        )[bidx]

        if recompute_normal:
            o3d_pcd.estimate_normals()

        if method == 'alpha':
            o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                o3d_pcd, alpha)
        elif method == 'ball':
            o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                o3d_pcd, o3d.utility.DoubleVector(ball_radii),
            )
        elif method == 'poisson':
            o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                o3d_pcd, depth=poisson_depth)
        else:
            raise NotImplementedError

        return Mesh(
            mesh=o3d_mesh,
            scale=None,
            center_w=None,
            preprocess_mesh=False,
        )

    @staticmethod
    def cat(point_clouds: T.List['PointCloud'], dim: int) -> 'PointCloud':
        """concatenate at `dim`."""

        if dim == 0:
            # check all point clouds have the same included_point_at_inf and n
            for i in range(1, len(point_clouds)):
                assert point_clouds[i].included_point_at_inf == point_clouds[0].included_point_at_inf
                assert point_clouds[i].xyz_w.size(1) == point_clouds[0].xyz_w.size(1)

            out_dict = dict()
            for name in point_clouds[0].attr_names:
                arr = [getattr(p, name, None) for p in point_clouds]
                if None in arr:
                    out_dict[name] = None
                else:
                    out_dict[name] = torch.cat(arr, dim=dim)

            if len(point_clouds) > 0:
                out_dict['included_point_at_inf'] = point_clouds[0].included_point_at_inf
            else:
                out_dict['included_point_at_inf'] = False
        elif dim == 1:
            out_dict = dict()
            for name in point_clouds[0].attr_names:
                arrs = []
                for i in range(len(point_clouds)):
                    arr = getattr(point_clouds[i], name, None)
                    if arr is not None:
                        if point_clouds[i].included_point_at_inf and i > 0:
                            start_idx = 1  # remove point at inf for i >= 1
                        else:
                            start_idx = 0
                        arr = arr[:, start_idx:]
                    arrs.append(arr)

                if None in arrs:
                    out_dict[name] = None
                else:
                    out_dict[name] = torch.cat(arrs, dim=dim)

            if len(point_clouds) > 0:
                out_dict['included_point_at_inf'] = point_clouds[0].included_point_at_inf
            else:
                out_dict['included_point_at_inf'] = False
        else:
            raise NotImplementedError
        return PointCloud(**out_dict)

    def voxel_downsampling(
            self,
            cell_width: float,
            sigma: float = 0.5,
            drop_features: bool = True,
            bidx: int = 0,
    ) -> 'PointCloud':
        """
        Voxel downsampling uses a voxel grid to uniformly downsample the input point cloud.

        Procedure:
        - Points are discretized into voxels.
        - Each occupied voxel generates exactly one point by averaging all points inside.

        Args:
            cell_width:
                the width of each grid cell.
                If <0, return self (do nothing)
            sigma:
                the sigma used in computing the gaussian weight

        Returns:
        """
        if cell_width < 0:
            return self

        print(f'voxel downsampling started, original num points = {self.xyz_w.size(1)}')

        assert self.xyz_w.size(0) == 1

        sigma = sigma * cell_width

        xyz_w = self.extract_valid_attr(
            arr=self.xyz_w,
            bidx=bidx,
        ).unsqueeze(0)  # (b=1, n, 3)

        # construct grid
        grid_to = xyz_w.max(dim=-2, keepdim=True)[0] + 1.e-3  # (b, 1, 3)
        grid_from = xyz_w.min(dim=-2, keepdim=True)[0] - 1.e-3  # (b, 1, 3)
        grid_width = grid_to - grid_from  # (b, 1, 3)
        grid_size = torch.ceil(grid_width / cell_width).long()  # (b, 1, 3)
        cell_width = grid_width / grid_size.float()  # (b, 1, 3)

        # discretize to cell idx
        subidxs = torch.floor((xyz_w - grid_from) / cell_width).long()  # (b, n, 3)
        inds = subidxs[..., 2] + \
               subidxs[..., 1] * grid_size[..., 2] + \
               subidxs[..., 0] * (grid_size[..., 1] * grid_size[..., 2])  # (b, n)

        # remap ind to unique index (remove unused grid_cells)
        all_point_clouds = []
        for b in range(self.xyz_w.size(0)):
            # xyz_w = self.xyz_w[b, start_idx:]
            _, idxs, counts = torch.unique(inds[b], return_inverse=True, return_counts=True)
            # idxs: (n,)
            # counts: (num_occupied_cells,)
            num_occupied_cells = counts.size(0)

            # average xyz to get new xyz (so implicitly weighted by sample density)
            xyz_w_mean = torch_scatter.scatter_mean(
                xyz_w[b],  # (n, 3)
                index=idxs.unsqueeze(-1),  # (n, 1)
                dim=-2,
            )  # (num_occupied_cells, 3)

            xyz_w_mean_expanded = xyz_w_mean[idxs]  # (n, 3)
            squared_dists = torch.sum((xyz_w[b] - xyz_w_mean_expanded) ** 2, dim=-1)  # (n,)
            weights = torch.exp(-1 * squared_dists / (2 * sigma ** 2))  # (n,)
            weights_sum = torch_scatter.scatter_sum(
                weights,  # (n,)
                index=idxs,  # (n,)
                dim=0,
            )  # (num_occupied_cells,)
            normalized_weights = weights / weights_sum[idxs]  # (n,)
            normalized_weights = normalized_weights.unsqueeze(-1)  # (n, 1)

            # weighted average each feature
            out_dict = dict()
            out_dict['xyz_w'] = xyz_w_mean.unsqueeze(0)  # (1, num_occupied_cells, 3)

            for attr_name in self.attr_names:
                if attr_name == 'xyz_w':
                    continue

                if attr_name == 'img_idxs':
                    out_dict[attr_name] = None
                    continue

                if not drop_features or attr_name in {'rgb', 'normal_w', 'feature'}:
                    arr = getattr(self, attr_name, None)
                    if arr is None:
                        out_dict[attr_name] = None
                        continue

                    arr = self.extract_valid_attr(
                        arr=arr,
                        bidx=b,
                    ) * normalized_weights  # (n, dim)
                    arr = torch_scatter.scatter_sum(
                        arr,  # (n, dim)
                        index=idxs.unsqueeze(-1),  # (n, 1)
                        dim=0,
                    )  # (num_occupied_cells, dim)

                    # make sure direction are unit norm
                    if attr_name in {'normal_w', 'captured_z_direction_w', 'captured_view_direction_w'}:
                        arr = torch.nn.functional.normalize(arr, p=2., dim=-1)  # (num_occupied_cells, dim)

                    out_dict[attr_name] = arr.unsqueeze(0)  # (1, num_occupied_cells, dim)
                else:
                    out_dict[attr_name] = None

            point_cloud = PointCloud(**out_dict)  # included_point_at_inf = False
            all_point_clouds.append(point_cloud)

        point_cloud = PointCloud.cat(all_point_clouds, dim=0)
        print(
            f'voxel downsampling finished, num points = {point_cloud.xyz_w.size(1)} '
            f'({point_cloud.xyz_w.size(1) / self.xyz_w.size(1) * 100.:.2f}%)')
        return point_cloud

    def remove_outlier(
            self,
            radius: float,
            min_num_points_in_radius: int,
            printout: bool = False,
    ) -> 'PointCloud':
        """
        Remove the outlier points in the point cloud.
        Currently, it removes points that have few neighbors in a given sphere around them.

        Args:
            radius:
                the radius of the sphere
            min_num_points_in_radius:
                minimum number of points within the sphere to consider a point as inlier

        Returns:
            self, with the valid_mask modified
        """
        if radius is None or radius < 0 or min_num_points_in_radius is None or min_num_points_in_radius < 0:
            return self

        # make sure valid mask is not None
        self.realize_valid_mask()

        if printout:
            print('Removing outlier points:', flush=True)
        # we use the raw xyz_w instead of the o3d_pcd, so the index mapping is easy
        for ib in range(self.xyz_w.size(0)):
            xyz_w = self.xyz_w[ib]  # (n, 3)
            if self.included_point_at_inf:
                xyz_w = xyz_w[1:]  # (n, 3)
                idx_offset = 1
            else:
                idx_offset = 0

            xyz_w = xyz_w.detach().cpu().numpy()
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(xyz_w)  # (n, 3)

            cleaned_o3d_pcd, valid_idxs = o3d_pcd.remove_radius_outlier(
                nb_points=min_num_points_in_radius,
                radius=radius
            )  # valid_idxs contains the list of valid index in o3d_pcd

            invalid_mask = torch.ones(self.xyz_w.size(1), dtype=torch.bool)
            for idx in valid_idxs:
                invalid_mask[idx + idx_offset] = 0
            invalid_mask = invalid_mask.to(device=self.valid_mask.device)

            # mark the valid_mask
            self.valid_mask[ib, invalid_mask, :] = 0

            if printout:
                print(
                    f'  ({ib}): removed {xyz_w.shape[0] - len(valid_idxs)} / {xyz_w.shape[0]}'
                    f' = {(xyz_w.shape[0] - len(valid_idxs)) / xyz_w.shape[0] * 100.:.2f}% points',
                    flush=True,
                )

        return self

    def save(
            self,
            output_dir: str,
            overwrite: bool = False,
            save_ply: bool = True,
            save_pt: bool = True,
    ):
        """Save the point cloud as a ply file and a npz."""
        if os.path.exists(output_dir) and not overwrite:
            raise RuntimeError
        os.makedirs(output_dir, exist_ok=True)

        # o3d point cloud
        if save_ply:
            o3d_pcds = self.get_o3d_pcds()
            for i in range(self.xyz_w.size(0)):
                filename = os.path.join(output_dir, f'pcd_{i}.ply')
                o3d.io.write_point_cloud(
                    filename=filename,
                    pointcloud=o3d_pcds[i],
                )

        # pt
        if save_pt:
            filename = os.path.join(output_dir, f'state_dict.pt')
            state_dict = self.state_dict()
            torch.save(state_dict, filename)

    def save_as_npbgpp(
            self,
            filenames: T.List[str],
            overwrite: bool = False,
    ):
        """Save the ib-th point cloud as a ply file."""
        assert len(filenames) == self.xyz_w.size(0)
        for filename in filenames:
            if os.path.exists(filename) and not overwrite:
                raise RuntimeError

        # o3d point cloud
        o3d_pcds = self.get_o3d_pcds()
        for i in range(self.xyz_w.size(0)):
            o3d.io.write_point_cloud(
                filename=filenames[i],
                pointcloud=o3d_pcds[i],
            )

    def realize_valid_mask(self):
        if self.valid_mask is not None:
            return
        b, n, _dim = self.xyz_w.shape
        self.valid_mask = torch.ones(
            b, n, 1,
            dtype=torch.bool, device=self.xyz_w.device)
        if self.included_point_at_inf:
            self.valid_mask[:, 0] = 0

    def reset_valid_mask(self):
        """set all points as valid"""
        if self.valid_mask is None:
            return
        b, n, _dim = self.xyz_w.shape
        self.valid_mask = torch.ones(
            b, n, 1,
            dtype=torch.bool, device=self.xyz_w.device)
        if self.included_point_at_inf:
            self.valid_mask[:, 0] = 0

    def rasterize_surfel(
            self,
            camera: 'Camera',
            point_size: float = 1.,
            default_rgb: T.List[float] = (0.5, 0.5, 0.5),
            render_normal_map: bool = True,
            rgb_shading_mode: str = 'raw',
            # o3d_normal_radius: float = 0.1,
            # o3d_normal_max_nn: int = 30,
    ) -> 'RGBDImage':
        """
        Render the point cloud using surfel rasterization.

        Args:
            camera:
                camera (b, q)
            point_size:
                size of the points
            default_rgb:
                the color of the points when `self.rgb` is None
            render_normal_map:
                whether to rasterize normal_w (as rgb color of points).
                If `self.normal_w` is None, use o3d to estimate with default
                parameters (max_nn = 30).
            rgb_shading_mode:
                'raw': use the rgb values, this is the same as uniform lighting
                'directional': assume directional light comes from the camera center
                'half': 0.5 uniform + 0.5 directional
            # o3d_normal_radius:
            #     search radius (in meter) to use in o3d vertex normal estimation
            # o3d_normal_max_nn:
            #     max number of neighboring points to use in o3d vertex normal estimation

        Returns:
        """

        if isinstance(default_rgb, (int, float)):
            default_rgb = [float(default_rgb)] * 3
        assert len(default_rgb) == 3

        b = camera.H_c2w.size(0)
        q = camera.H_c2w.size(1)
        assert b == self.xyz_w.size(0)

        # get o3d pcds
        estimate_normal = render_normal_map or rgb_shading_mode in {'directional', 'half'}
        o3d_pcds = self.get_o3d_pcds(
            estimate_normal_if_not_exist=estimate_normal,
        )

        intrinsic = camera.intrinsic.detach().cpu().numpy()  # (b, q, 3, 3)
        H_w2c = camera.get_H_w2c().detach().cpu().numpy()  # (b, q, 4, 4)
        H_c2w = camera.H_c2w.detach().cpu().numpy()  # (b, q, 4, 4)

        # render each b
        all_imgs = []
        all_depths = []
        all_hit_maps = []
        all_normal_ws = []
        for ib in range(b):
            # rgb
            if o3d_pcds[ib].colors is None or np.asarray(o3d_pcds[ib].colors).shape[0] == 0:
                n = np.asarray(o3d_pcds[ib].points).shape[0]
                rgb = np.ones((n, 3))
                for c in range(3):
                    rgb[:, c] = default_rgb[c]
                o3d_pcds[ib].colors = o3d.utility.Vector3dVector(rgb)  # (n, 3)
            # rasterize rgb
            if rgb_shading_mode in {'raw', 'uniform'}:
                # make sure o3d_pcd has no normal
                tmp_normal = np.asarray(o3d_pcds[ib].normals)
                o3d_pcds[ib].normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))

                out_dict = render.rasterize(
                    meshes=o3d_pcds[ib],
                    intrinsic_matrix=intrinsic[ib],  # (q, 3, 3)
                    extrinsic_matrices=H_w2c[ib],
                    width_px=camera.width_px,
                    height_px=camera.height_px,
                    get_point_cloud=False,
                    point_size=point_size,
                )  # imgs: list of q images (h, w, 3), z_maps list of q depth map (h, w)

                # recover the original normal
                o3d_pcds[ib].normals = o3d.utility.Vector3dVector(tmp_normal)
            elif rgb_shading_mode in {'directional', 'half'}:
                assert o3d_pcds[ib].normals is not None or np.asarray(o3d_pcds[ib].normals).shape[0] > 0
                assert o3d_pcds[ib].colors is not None or np.asarray(o3d_pcds[ib].colors).shape[0] > 0

                imgs = []
                z_maps = []
                hit_maps = []
                rgb = np.copy(np.asarray(o3d_pcds[ib].colors))  # (n, 3)
                normal_w = np.array(o3d_pcds[ib].normals)  # (n, 3)
                normal_w = normal_w / (np.linalg.norm(normal_w, ord=2, axis=-1, keepdims=True) + 1e-9)

                for iq in range(q):
                    # shading (by treating rgb as albedo)
                    ld = camera.H_c2w[ib, iq, :3, 3].detach().cpu().numpy()
                    ld_norm = (ld ** 2).sum()
                    if ld_norm > 1e-6:
                        ld = ld / ld_norm
                    else:
                        ld = np.zeros((3,))
                        ld[2] = 1
                    if rgb_shading_mode == 'directional':
                        n_dot_l = np.abs(np.sum(normal_w * ld, axis=-1, keepdims=True))  # (n, 1)
                        colors = rgb * n_dot_l  # (n, 3)
                        o3d_pcds[ib].colors = o3d.utility.Vector3dVector(colors)  # (n, 3)
                    elif rgb_shading_mode == 'half':
                        n_dot_l = np.abs(np.sum(normal_w * ld, axis=-1, keepdims=True))  # (n, 1)
                        colors = rgb * (0.5 * n_dot_l + 0.5)  # (n, 3)
                        o3d_pcds[ib].colors = o3d.utility.Vector3dVector(colors)  # (n, 3)
                    else:
                        raise NotImplementedError

                    # make sure o3d_pcd has no normal
                    tmp_normal = np.asarray(o3d_pcds[ib].normals)
                    o3d_pcds[ib].normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))

                    out_dict = render.rasterize(
                        meshes=o3d_pcds[ib],
                        intrinsic_matrix=intrinsic[ib, iq],  # (3, 3)
                        extrinsic_matrices=H_w2c[ib, iq],
                        width_px=camera.width_px,
                        height_px=camera.height_px,
                        get_point_cloud=False,
                        point_size=point_size,
                    )  # imgs: list of q images (h, w, 3), z_maps list of q depth map (h, w)

                    # recover the original normal
                    o3d_pcds[ib].normals = o3d.utility.Vector3dVector(tmp_normal)

                    imgs.append(out_dict['imgs'][0])  # (h, w, 3)
                    z_maps.append(out_dict['z_maps'][0])  # (h, w)
                    hit_maps.append(out_dict['hit_maps'][0])  # (h, w)

                imgs = np.stack(imgs, axis=0)  # (q, h, w, 3)
                z_maps = np.stack(z_maps, axis=0)  # (q, h, w)
                hit_maps = np.stack(hit_maps, axis=0)  # (q, h, w)
                out_dict = dict(
                    imgs=imgs,
                    z_maps=z_maps,
                    hit_maps=hit_maps,
                )
            else:
                raise NotImplementedError

            imgs = np.stack(out_dict['imgs'], axis=0)  # (q, h, w, 3)
            depths = np.stack(out_dict['z_maps'], axis=0)  # (q, h, w)
            hit_maps = np.stack(out_dict['hit_maps'], axis=0)  # (q, h, w)
            all_imgs.append(imgs)
            all_depths.append(depths)
            all_hit_maps.append(hit_maps)

            # normal
            # note that since each normal vector can be flipped randomly,
            # we need to rasterize one image at a time to orient normal to
            # the camera center
            if render_normal_map:
                assert o3d_pcds[ib].normals is not None

                out_imgs = []
                for iq in range(q):
                    # o3d_pcds[ib].orient_normals_to_align_with_direction(
                    #     H_c2w[ib, iq, :3, 3]
                    # )

                    normal_w = np.array(o3d_pcds[ib].normals)  # (n, 3)
                    assert normal_w.shape[0] > 0
                    normal_w = normal_w / (np.linalg.norm(normal_w, ord=2, axis=-1, keepdims=True) + 1e-9)

                    # align normal to point the opposite direction of the camera ray
                    ray_direction_w = np.array(o3d_pcds[ib].points) - np.reshape(H_c2w[ib, iq, :3, 3], (1, 3))  # (n, 3)
                    normal_w = normal_w * \
                               (-1 * np.sign(np.sum(normal_w * ray_direction_w, axis=-1, keepdims=True)))

                    # map normals to rgb ([-1, 1] -> [0, 1])
                    normal_w = (normal_w + 1) * 0.5  # (n, 3)
                    o3d_pcds[ib].colors = o3d.utility.Vector3dVector(normal_w)  # (n, 3)

                    # make sure o3d_pcd has no normal
                    tmp_normal = np.asarray(o3d_pcds[ib].normals)
                    o3d_pcds[ib].normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))

                    out_dict = render.rasterize(
                        meshes=o3d_pcds[ib],
                        intrinsic_matrix=intrinsic[ib, iq],  # (3, 3)
                        extrinsic_matrices=H_w2c[ib, iq],  # (4, 4)
                        width_px=camera.width_px,
                        height_px=camera.height_px,
                        get_point_cloud=False,
                        point_size=point_size,
                    )  # imgs: list of q images (h, w, 3), z_maps list of q depth map (h, w)
                    out_imgs.append(out_dict['imgs'][0])

                    # recover the original normal
                    o3d_pcds[ib].normals = o3d.utility.Vector3dVector(tmp_normal)

                all_normal_ws.append(
                    np.stack(out_imgs, axis=0)  # (q, h, w, 3)
                )

        all_imgs = np.stack(all_imgs, axis=0)  # (b, q, h, w, 3)
        all_depths = np.stack(all_depths, axis=0)  # (b, q, h, w)
        all_hit_maps = np.stack(all_hit_maps, axis=0)  # (b, q, h, w)
        if len(all_normal_ws) > 0:
            all_normal_ws = np.stack(all_normal_ws, axis=0)  # (b, q, h, w, 3)
            # [0, 1] -> [-1, 1]
            all_normal_ws = (all_normal_ws - 0.5) * 2.
            all_normal_ws = all_normal_ws / np.linalg.norm(all_normal_ws, ord=2, axis=-1, keepdims=True)
        else:
            all_normal_ws = None

        return RGBDImage(
            rgb=torch.from_numpy(all_imgs),
            depth=torch.from_numpy(all_depths),
            hit_map=torch.from_numpy(all_hit_maps),
            camera=camera,  # we choose not to deepcopy, watch out
            normal_w=torch.from_numpy(all_normal_ws) if all_normal_ws is not None else None,
        )

    def silhouette_carving(
            self,
            hit_map: torch.Tensor,
            camera: 'Camera',
            dilate: bool = True,
    ) -> 'PointCloud':
        """
        Use the hit_map and the camera in the rgbd_image to invalid points.
        It marks the valid_mask to be False for point that should be carved
        without actually remove them.

        Args:
            hit_map:
                hit_map: (b, q, h, w)
            camera:
                (b, q), h, w
            clone:
                whether to create a new point cloud (new memory)


        Returns:
            a new point cloud  (b, n')
        """

        ori_include_inf = self.included_point_at_inf
        self.remove_point_at_inf()
        self.realize_valid_mask()

        b, n, _3 = self.xyz_w.shape
        _b, q, h, w = hit_map.shape
        bq = b * q
        assert b == 1

        # project onto the sensor
        uv_cs = utils.pinhole_projection(
            xyz_w=self.xyz_w,  # (b, n)
            intrinsics=camera.intrinsic,  # (b, q)
            H_c2w=camera.H_c2w,  # (b, q)
            dim_b=1,
        )  # (b, q, n, 2), [0, w] [0, h]

        # convert to [0, 1]
        uv_cs[..., 0] = uv_cs[..., 0] / camera.width_px
        uv_cs[..., 1] = uv_cs[..., 1] / camera.height_px

        uv_cs = uv_cs.reshape(bq, n, 2)
        hit_map = hit_map.reshape(bq, h, w, 1)  # (bq, h, w, 1)

        # dilation the hit map so the boundary is more robust
        if dilate:
            hit_map = kornia.morphology.dilation(
                tensor=hit_map.permute(0, 3, 1, 2).float(),  # (bq, 1, h, w)
                kernel=torch.ones(3, 3, dtype=torch.float, device=hit_map.device),
            ).permute(0, 2, 3, 1)  # (bq, h, w, 1)
        else:
            hit_map = hit_map.float()

        miss_map = 1 - hit_map  # (bq, h, w, 1)
        # uv-sample to get the miss_map (we use miss map in case a pixel is not visible in some images)
        miss = utils.uv_sampling(
            uv=uv_cs,  # (bq, n, 2)
            feature_map=miss_map,  # (bq, h, w, 1)
            mode='bilinear',
            padding_mode='border',
        )  # (bq, n, 1)
        hit = (miss <= 0.5)  # (bq, n, 1)
        hit = hit.reshape(b, q, n)

        # if a point is not hit in any image, it is invalid
        valid_mask = hit.all(dim=1)  # (b, n)

        # combine valid_mask with the original valid_mask
        self.valid_mask = torch.logical_and(
            self.valid_mask,  # (b, n, 1)
            valid_mask.unsqueeze(-1),  # (b, n, 1)
        )

        if ori_include_inf:
            self.insert_point_at_inf()

        return self


class Ray:
    def __init__(
            self,
            origins_w: torch.Tensor,  # (b, *m_shape, 3)
            directions_w: torch.Tensor,  # (b, *m_shape, 3)
    ):
        self.origins_w = origins_w
        self.directions_w = directions_w

    def to(self, device: torch.device) -> 'Ray':
        for attr_name in ['origins_w', 'directions_w']:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.to(device=device))
        return self

    def reshape(self, *shape: T.List[int]):
        self.origins_w = self.origins_w.reshape(*shape)
        self.directions_w = self.directions_w.reshape(*shape)
        return self

    def chunk(self, chunks: int, dim: int = 0) -> T.List['Ray']:
        """Return a"""
        origins_w_list = self.origins_w.chunk(chunks, dim)
        directions_w_list = self.directions_w.chunk(chunks, dim)
        rays = []
        for origins_w, directions_w in zip(origins_w_list, directions_w_list):
            ray = Ray(
                origins_w=origins_w,
                directions_w=directions_w,
            )
            rays.append(ray)
        return rays

    def random_perturb_direction(
            self,
            shift: T.Optional[float],
            angle: T.Optional[float],
    ):
        """
        Perturb the rays by randomly shifting the ray origin by [-shift, shift],
        and randomly rotating the ray direction with [-angle, angle] in degree. 
        
        Args:
            shift:
                [-shift, shift]
            angle: 
                [-angle, angle] in degrees
            rng:
                the random generator
        """

        if shift is not None and math.fabs(shift) > 1e-6:
            r_shifts = (torch.rand_like(self.origins_w) - 0.5) * 2. * shift  # [-shift, shift]
            self.origins_w = self.origins_w + r_shifts  # (b, *m_shape, 3)
        if angle is not None and math.fabs(angle) > 1e-3:
            r_angles = (
                               torch.rand_like(self.directions_w) - 0.5
                       ) * 2. * angle  # (b, *m_shape, 3) [-angle, angle] in degree
            r_angles = r_angles.view(-1, 3)  # (b*m_shape, 3)
            Rs = torch.from_numpy(
                Rotation.from_euler('xyz', r_angles, degrees=True).as_matrix()
            )  # (bm, 3, 3) rotation matrix
            Rs = Rs.reshape(*(self.directions_w.shape[:-1]), 3, 3)  # (b, *m_shape, 3, 3)
            self.directions_w = (Rs @ self.directions_w.unsqueeze(-1))[..., 0]  # (b, *m, 3)

    @staticmethod
    def cat(rays: T.List['Ray'], dim: int) -> 'Ray':
        out_dict = dict()
        for name in ['origins_w', 'directions_w']:
            arr = [getattr(p, name, None) for p in rays]
            if None in arr:
                out_dict[name] = None
            else:
                out_dict[name] = torch.cat(arr, dim=dim)
        return Ray(**out_dict)

    def masked_fill(self, mask: torch.Tensor, ray_src: 'Ray'):
        self.origins_w[mask] = ray_src.origins_w[mask]
        self.directions_w[mask] = ray_src.directions_w[mask]

    @property
    def dtype(self):
        return self.origins_w.dtype

    @property
    def shape(self):
        return self.origins_w.shape

    @property
    def size(self):
        return self.origins_w.size

    @property
    def device(self):
        return self.origins_w.device

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""
        to_save = dict()
        for name in ['origins_w', 'directions_w']:
            to_save[name] = getattr(self, name, None)
        return to_save

    def load_state_dict(
            self,
            state_dict: T.Dict[str, T.Any],
    ):
        """Load the state dictionary."""
        for name in ['origins_w', 'directions_w']:
            setattr(self, name, state_dict.get(name, None))

    def save(
            self,
            output_dir: str,
            overwrite: bool = False,
            save_ply: bool = True,
            save_pt: bool = True,
            cylinder_radius: float = 0.1,
            cone_radius: float = None,
            max_cone_height: float = None,
            end_xyz_w: torch.Tensor = None,  # same shape as origins_w
    ):
        if self.origins_w is None or self.directions_w is None:
            return

        if cone_radius is None:
            cone_radius = cylinder_radius * 1.5

        if max_cone_height is None:
            max_cone_height = cylinder_radius * 2.

        if os.path.exists(output_dir) and not overwrite:
            raise RuntimeError(f'output dir {output_dir} exists')
        os.makedirs(output_dir, exist_ok=True)

        if save_pt:
            filename = os.path.join(output_dir, 'state_dict.pt')
            torch.save(self.state_dict(), filename)

        if end_xyz_w is None:
            ray_ts = torch.ones(*self.origins_w.shape[:-1], device=self.origins_w.device)
            add_end_ball = False
        else:
            ray_ts = torch.linalg.vector_norm(end_xyz_w - self.origins_w, ord=2, dim=-1)
            add_end_ball = True

        b, *m_shape, _3 = self.origins_w.shape
        origins_w = self.origins_w.reshape(b, -1, 3)
        directions_w = self.directions_w.reshape(b, -1, 3)
        ray_ts = ray_ts.reshape(b, -1)
        if save_ply:
            # we are going to save individual camera frames
            for ib in range(origins_w.size(0)):
                sub_dir = os.path.join(output_dir, f'batch_{ib}')
                os.makedirs(sub_dir, exist_ok=True)

                for im in range(origins_w.size(1)):

                    R = rigid_motion.get_min_R(
                        v1=np.array([0, 0, 1.], dtype=np.float32),
                        v2=directions_w[ib, im].detach().cpu().numpy(),
                    )
                    H_c2w = np.eye(4)
                    H_c2w[:3, :3] = R
                    H_c2w[:3, 3] = origins_w[ib, im].detach().cpu().float().numpy()

                    # ray origin
                    mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=cylinder_radius * 1.1,
                        # width=cylinder_radius * 2.2,
                        # height=cylinder_radius * 2.2,
                        # depth=cylinder_radius * 2.2,
                    )
                    mesh.transform(H_c2w)
                    o3d.io.write_triangle_mesh(
                        filename=os.path.join(sub_dir, f'{im}_from.ply'),
                        mesh=mesh,
                    )

                    # ray direction
                    t = ray_ts[ib, im].detach().cpu().numpy()
                    if t > 0:
                        cone_height = min(t * 0.05, max_cone_height)
                        cylinder_height = t - cone_height
                        mesh = o3d.geometry.TriangleMesh.create_arrow(
                            cylinder_radius=cylinder_radius,
                            cone_radius=cone_radius,
                            cylinder_height=cylinder_height,
                            cone_height=cone_height,
                        )
                        mesh.transform(H_c2w)
                        o3d.io.write_triangle_mesh(
                            filename=os.path.join(sub_dir, f'{im}_arrow.ply'),
                            mesh=mesh,
                        )

                    # end point
                    if add_end_ball:
                        H_c2w2 = np.eye(4)
                        H_c2w2[:3, :3] = R
                        H_c2w2[:3, 3] = origins_w[ib, im].detach().cpu().float().numpy() + \
                                        t * directions_w[ib, im].detach().cpu().float().numpy()

                        mesh = o3d.geometry.TriangleMesh.create_sphere(
                            radius=cylinder_radius * 1.1,
                        )
                        mesh.transform(H_c2w2)
                        o3d.io.write_triangle_mesh(
                            filename=os.path.join(sub_dir, f'{im}_to.ply'),
                            mesh=mesh,
                        )


class PointersectRecord:
    def __init__(
            self,
            intersection_xyz_w: torch.Tensor,  # (b, *m_shape, 3)
            intersection_surface_normal_w: torch.Tensor,  # (b, *m_shape, 3)
            intersection_rgb: torch.Tensor,  # (b, *m_shape, 3)
            blending_weights: torch.Tensor,  # (b, *m_shape, k)  k: # neighbor points
            neighbor_point_idxs: torch.Tensor,  # long (b, *m_shape, k)
            neighbor_point_valid_len: torch.Tensor,  # long (b, *m_shape)
            ray_t: torch.Tensor,  # (b, *m_shape)
            ray_hit: torch.Tensor,  # (b, *m_shape)  bool
            ray_hit_logit: torch.Tensor,  # (b, *m_shape)
            model_attn_weights: torch.Tensor,  # (b, *m_shape, k+1, n_layers)
            refined_ray_hit: T.Optional[torch.Tensor] = None,  # (b, *m_shape)  bool
            model_info: T.Optional[T.Dict[str, T.Any]] = None,
            intersection_plane_normals_w: torch.Tensor = None,  # (b, *m_shape, 3)
            geometry_weights: torch.Tensor = None,  # (b, *m_shape, k)
            valid_neighbor_idx_mask: torch.Tensor = None,  # (b, *m_shape, k)  whether the neighbor_point_idxs is valid
            valid_plane_normal_mask: torch.Tensor = None,  # (b, *m_shape)
            total_time: float = None,
    ):
        self.intersection_xyz_w = intersection_xyz_w
        self.intersection_surface_normal_w = intersection_surface_normal_w
        self.intersection_rgb = intersection_rgb
        self.blending_weights = blending_weights
        self.neighbor_point_idxs = neighbor_point_idxs
        self.neighbor_point_valid_len = neighbor_point_valid_len
        self.ray_t = ray_t
        self.ray_hit = ray_hit
        self.ray_hit_logit = ray_hit_logit
        self.model_attn_weights = model_attn_weights
        self.refined_ray_hit = refined_ray_hit
        self.intersection_plane_normals_w = intersection_plane_normals_w
        self.geometry_weights = geometry_weights
        self.valid_neighbor_idx_mask = valid_neighbor_idx_mask
        self.valid_plane_normal_mask = valid_plane_normal_mask
        self.total_time = total_time
        self.model_info = model_info

        # self.cached_info = cached_info  # not saved nor concat nor reshaped

        self.attr_names = [
            'intersection_xyz_w', 'intersection_surface_normal_w', 'intersection_rgb',
            'blending_weights', 'neighbor_point_idxs', 'neighbor_point_valid_len',
            'ray_t', 'ray_hit', 'ray_hit_logit', 'model_attn_weights',
            'refined_ray_hit', 'model_info',
            'intersection_plane_normals_w', 'geometry_weights',
            'valid_neighbor_idx_mask', 'valid_plane_normal_mask',
        ]

    def to(self, device: torch.device) -> 'PointersectRecord':
        for attr_name in self.attr_names:
            if attr_name == 'model_info':
                continue
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.to(device=device))
        return self

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""
        to_save = dict()
        for name in self.attr_names:
            to_save[name] = getattr(self, name, None)
        return to_save

    def load_state_dict(
            self,
            state_dict: T.Dict[str, T.Any],
    ):
        """Load the state dictionary."""
        for name in self.attr_names:
            setattr(self, name, state_dict.get(name, None))

    @staticmethod
    def cat(records: T.List['PointersectRecord'], dim: int) -> 'PointersectRecord':
        """
        Concatenate a list of PointersectRecord at the given dimension.
        It is useful to split m_shape.
        """
        out = dict()
        for name in [
            'intersection_xyz_w', 'intersection_surface_normal_w', 'intersection_rgb',
            'blending_weights', 'neighbor_point_idxs', 'neighbor_point_valid_len',
            'ray_t', 'ray_hit', 'ray_hit_logit', 'model_attn_weights',
            'refined_ray_hit',
            'intersection_plane_normals_w',
            'geometry_weights',
            'valid_neighbor_idx_mask',
            'valid_plane_normal_mask',
        ]:
            arr = [getattr(r, name, None) for r in records]
            if None in arr:
                out[name] = None
            else:
                out[name] = torch.cat(arr, dim=dim)

        if len(records) > 0:
            out['model_info'] = records[0].model_info

        return PointersectRecord(**out)

    def chunk(self, chunks: int, dim: int) -> T.List['PointersectRecord']:
        """
        Chunk the PointersectRecord at the given dimension.
        As pytorch, the resulted tensors are views to the original ones.
        """

        attr_names = [
            'intersection_xyz_w', 'intersection_surface_normal_w', 'intersection_rgb',
            'blending_weights', 'neighbor_point_idxs', 'neighbor_point_valid_len',
            'ray_t', 'ray_hit', 'ray_hit_logit', 'model_attn_weights',
            'refined_ray_hit',
            'intersection_plane_normals_w',
            'geometry_weights',
            'valid_neighbor_idx_mask',
            'valid_plane_normal_mask',
        ]

        actual_chunks = None
        out = dict()
        for name in attr_names:
            arr = getattr(self, name, None)
            if arr is None:
                out[name] = None
            else:
                out[name] = arr.chunk(chunks=chunks, dim=dim)
                if actual_chunks is None:
                    actual_chunks = len(out[name])
                else:
                    assert len(out[name]) == actual_chunks

        results = []
        for i in range(actual_chunks):
            tmp_dict = dict()
            for name in attr_names:
                arr_list = out[name]
                if arr_list is None:
                    tmp_dict[name] = None
                else:
                    tmp_dict[name] = arr_list[i]
            tmp_dict['model_info'] = self.model_info
            p = PointersectRecord(**tmp_dict)
            results.append(p)

        return results

    @staticmethod
    def aggregate(records: T.List['PointersectRecord']) -> 'PointersectRecord':
        """
        Aggregate a list of PointersectRecord of the same shape.
        Note that it will set many attributes to None.
        """
        out = dict()
        # simple average
        for name in [
            'intersection_xyz_w', 'intersection_rgb',
            'ray_t', 'ray_hit', 'ray_hit_logit', 'model_attn_weights',
            'refined_ray_hit',
        ]:
            arr = [getattr(r, name, None) for r in records]
            arr = [a for a in arr if a is not None]
            if len(arr) == 0:
                out[name] = None
            else:
                out[name] = sum(arr) / len(arr)

        # set to be the first
        for name in [
            'blending_weights', 'neighbor_point_idxs', 'neighbor_point_valid_len',
            'model_attn_weights',
            'geometry_weights',
            'valid_neighbor_idx_mask',
        ]:
            arr = [getattr(r, name, None) for r in records]
            if len(arr) == 0:
                out[name] = None
            else:
                out[name] = arr[0]

        # sum -> normalize to unit norm
        for name in [
            'intersection_surface_normal_w',
            'intersection_plane_normals_w',
        ]:
            arr = [getattr(r, name, None) for r in records]
            arr = [a for a in arr if a is not None]
            if len(arr) == 0:
                out[name] = None
            else:
                out[name] = sum(arr)
                out[name] = torch.nn.functional.normalize(out[name], p=2, dim=-1)

        # set to be and
        for name in [
            'valid_plane_normal_mask'
        ]:
            arr = [getattr(r, name, None) for r in records]
            arr = [a for a in arr if a is not None]
            if len(arr) == 0:
                out[name] = None
            else:
                out[name] = arr[0]
                for i in range(1, len(arr)):
                    out[name] = torch.logical_and(out[name], arr[i])

        if len(records) > 0:
            out['model_info'] = records[0].model_info

        return PointersectRecord(**out)

    def reshape(self, new_b: int, new_m_shape: T.List[int]):
        """Reshape each attributes."""
        if isinstance(new_b, int):
            new_b = [new_b]
        if isinstance(new_m_shape, int):
            new_m_shape = [new_m_shape]

        b, *m_shape, _ = self.intersection_xyz_w.shape

        # (b, *m_shape, d)
        for name in [
            'intersection_xyz_w', 'intersection_surface_normal_w', 'intersection_rgb',
            'blending_weights', 'neighbor_point_idxs',
            'intersection_plane_normals_w', 'geometry_weights',
            'valid_neighbor_idx_mask',
        ]:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            arr = arr.reshape(*new_b, *new_m_shape, arr.size(-1))
            setattr(self, name, arr)

        # (b, *m_shape)
        for name in [
            'neighbor_point_valid_len', 'ray_t', 'ray_hit', 'ray_hit_logit',
            'refined_ray_hit', 'valid_plane_normal_mask',
        ]:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            arr = arr.reshape(*new_b, *new_m_shape)
            setattr(self, name, arr)

        # # (b, *m_shape, k+1, n_layers)
        for name in [
            'model_attn_weights',
        ]:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            arr = arr.reshape(*new_b, *new_m_shape, arr.size(-2), arr.size(-1))
            setattr(self, name, arr)

    def save(
            self,
            output_dir: str,
            overwrite: bool = False,
    ):
        if os.path.exists(output_dir) and not overwrite:
            raise RuntimeError
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, 'state_dict.pt')
        torch.save(self.state_dict(), filename)

    def get_rgbd_image(
            self,
            camera: 'Camera',
            th_hit_prob: float = None,
            th_dot_product: float = None,  # (less than 60 degree)
            use_plane_normal: bool = False,
    ) -> 'RGBDImage':
        """
        Return rgbd_image (with surface normal) to compare with other methods.
        Camera should be the orignal camera used to cast the camera rays.
        Only support shape = (b, q, h, w).
        """
        b, *m_shape, _ = self.intersection_xyz_w.shape  # (b, q, h, w, 3)
        assert len(m_shape) == 3  # (q, h, w)
        q, h, w = m_shape

        if th_hit_prob is None:
            th_hit_prob = 0
        if th_dot_product is None:
            th_dot_product = 0

        # xyz_w -> xyz_c
        H_w2c = camera.get_H_w2c()  # (b, q, 4, 4)
        H_w2c = H_w2c.reshape(b, q, 1, 1, 4, 4)  # (b, q, 1, 1, 4, 4)
        xyz_w = self.intersection_xyz_w.unsqueeze(-1)  # (b, q, h, w, 3, 1)
        xyz_c = H_w2c[..., :3, :3] @ xyz_w + H_w2c[..., :3, 3:4]  # (b, q, h, w, 3, 1)
        # depth is the z in camera coordinate
        depth = xyz_c[..., 2, 0]  # (b, q, h, w)
        assert depth.shape == torch.Size([b, *m_shape])

        # determine hit map to use
        if self.refined_ray_hit is not None:
            hit_map = self.refined_ray_hit  # (b, q, h, w)
        else:
            hit_map = self.ray_hit  # (b, q, h, w)

        if th_dot_product > 1e-6 or th_hit_prob > 1e-6:
            ray = camera.generate_camera_rays(device=camera.H_c2w.device)
            confidence = self.compute_confidence(
                ray_direction_w=ray.directions_w,  # (b, q, h, w, 3)
                th_hit_prob=th_hit_prob,
                th_dot_product=th_dot_product,
            )  # (b, q, h, w)
            hit_map = torch.logical_and(hit_map, confidence)

        if use_plane_normal and self.intersection_plane_normals_w is not None:
            normal_w = self.intersection_plane_normals_w * hit_map.unsqueeze(-1)  # (b, q, h, w, 3)
        else:
            normal_w = self.intersection_surface_normal_w * hit_map.unsqueeze(-1)  # (b, q, h, w, 3)

        return RGBDImage(
            rgb=self.intersection_rgb * hit_map.unsqueeze(-1),  # (b, q, h, w, 3)
            depth=depth * hit_map,  # (b, q, h, w)
            normal_w=normal_w,  # (b, q, h, w, 3)
            hit_map=hit_map,
            camera=camera,
        )

    def compute_confidence(
            self,
            ray_direction_w: T.Optional[torch.Tensor] = None,
            th_hit_prob: float = 0.85,
            th_dot_product: float = 0.5,  # (less than 60 degree)
    ):
        """
        Compute the binary confidence of the ray based on hit
        and the angle between the ray direction and the surface normal
        is small enough.

        Args:
            ray_direction_w:
                (b, *m_shape, 3) ray direction used to query the points
            th_hit_prob:
                confident if hit_prob is large enough
            th_dot_product:
                confident if the angle between ray and surface normal
                is small enough.

        Returns:
            confidence (b, *m_shape,), {0, 1: confident}
        """
        hit_probs = self.ray_hit_logit.sigmoid()
        hit = hit_probs >= min(1 - 1e-4, th_hit_prob)  # (b, *m_shape,) bool
        miss = hit_probs <= max(1e-4, (1 - th_hit_prob))  # (b, *m_shape,) bool
        hit_miss = hit + miss

        if ray_direction_w is not None:
            dot_prod = (ray_direction_w * self.intersection_surface_normal_w).sum(dim=-1)  # (b, *m)
            angle = dot_prod.abs() >= th_dot_product
            return torch.logical_and(hit_miss, angle)
        else:
            return hit_miss


class Camera:
    def __init__(
            self,
            H_c2w: torch.Tensor,  # (b, q, 4, 4)  camera pose in the world coord
            intrinsic: torch.Tensor,  # (b, q, 3, 3)  camera intrinsics
            width_px: int,
            height_px: int,
    ):
        self.H_c2w = H_c2w
        self.intrinsic = intrinsic
        self.width_px = width_px
        self.height_px = height_px

        self.attr_names = ['H_c2w', 'intrinsic', 'width_px', 'height_px']

    def index_select(self, dim: int, index: torch.Tensor) -> 'Camera':
        camera = self.clone()
        for attr_name in ['H_c2w', 'intrinsic']:
            arr = getattr(camera, attr_name, None)
            if arr is not None:
                setattr(camera, attr_name, torch.index_select(arr, dim=dim, index=index))
        return camera

    def chunk(self, chunks: int, dim: int = 0) -> T.List['Camera']:
        out_dict = dict()
        total = None
        for attr_name in ['H_c2w', 'intrinsic']:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                chunked_arr = arr.chunk(chunks=chunks, dim=dim)
                out_dict[attr_name] = chunked_arr
                if total is None:
                    total = len(chunked_arr)
                else:
                    assert len(chunked_arr) == total
        cameras = []
        for i in range(total):
            d = dict()
            for attr_name in out_dict:
                d[attr_name] = out_dict[attr_name][i]
            camera = Camera(**d, width_px=self.width_px, height_px=self.height_px)
            cameras.append(camera)
        return cameras

    def to(self, device: torch.device) -> 'Camera':
        self.H_c2w = self.H_c2w.to(device=device)
        self.intrinsic = self.intrinsic.to(device=device)
        return self

    def detach(self) -> 'Camera':
        self.H_c2w = self.H_c2w.detach()
        self.intrinsic = self.intrinsic.detach()
        return self

    def clone(self) -> 'Camera':
        return Camera(
            H_c2w=self.H_c2w.clone() if self.H_c2w is not None else None,
            intrinsic=self.intrinsic.clone() if self.intrinsic is not None else None,
            width_px=self.width_px,
            height_px=self.height_px,
        )

    @staticmethod
    def cat(cameras: T.List['Camera'], dim: int) -> 'Camera':
        out = dict()
        for name in ['H_c2w', 'intrinsic']:
            arr = [getattr(r, name, None) for r in cameras]
            if None in arr:
                out[name] = None
            else:
                out[name] = torch.cat(arr, dim=dim)
        width_pxs = [getattr(r, 'width_px', None) for r in cameras]
        height_pxs = [getattr(r, 'height_px', None) for r in cameras]
        assert len(np.unique(width_pxs)) == 1
        assert len(np.unique(height_pxs)) == 1
        out['width_px'] = width_pxs[0]
        out['height_px'] = height_pxs[0]

        return Camera(**out)

    def __getitem__(self, ib) -> 'Camera':
        """slice the camera in the b dimension. Always retain (b, q, 4, 4)
        even when ib is int."""
        if isinstance(ib, (int, torch.Size)):
            ib = slice(int(ib), int(ib) + 1)

        camera = Camera(
            H_c2w=self.H_c2w[ib],
            intrinsic=self.intrinsic[ib],
            width_px=self.width_px,
            height_px=self.height_px,
        )
        assert camera.H_c2w.ndim == 4
        assert camera.intrinsic.ndim == 4
        return camera

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""
        to_save = dict()
        for name in self.attr_names:
            to_save[name] = getattr(self, name, None)
        return to_save

    def load_state_dict(
            self,
            state_dict: T.Dict[str, T.Any],
    ):
        """Load the state dictionary."""
        for name in self.attr_names:
            setattr(self, name, state_dict.get(name, None))

    def load_json(
            self,
            filename: str,
            device: torch.device = torch.device('cpu'),
    ) -> 'Camera':
        """
        Load camera from a json file.

        Json file format:
            H_c2w:
                (b, q, 4, 4), a nested list containing the camera pose in the world coord.
                `b` is the batch dimension, `q` is number of camera poses in a batch.
                For example, `H_c2w[i,j]` is the 4x4 camera pose matrix that converts
                a point in the camera coordinate to the world coordinate.
            intrinsic:
                (b, q, 3, 3) a nested list containing the 3x3 camera intrinsic matrices.
            width_px:
                int, number of pixels in width (horizontal)
            height_px:
                int, number of pixels in height (vertical)
        """

        with open(filename, 'r') as f:
            d = json.load(f)

        if 'H_c2w' in d:
            H_c2w = torch.tensor(d['H_c2w'], dtype=torch.float, device=device)
        else:
            H_c2w = None

        if 'intrinsic' in d:
            intrinsic = torch.tensor(d['intrinsic'], dtype=torch.float, device=device)
        else:
            intrinsic = None

        return Camera(
            H_c2w=H_c2w,
            intrinsic=intrinsic,
            width_px=d.get('width_px', None),
            height_px=d.get('height_px', None),
        )

    def get_H_w2c(self) -> torch.Tensor:
        """
        Returns extrinsic matrices (inverse of H_c2w), shape: (b, q, 4, 4).
        """
        return rigid_motion.inv_homogeneous_tensors(self.H_c2w)

    def generate_camera_rays(
            self,
            subsample: int = 1,
            offsets: str = 'center',
            device: torch.device = torch.device('cpu'),
    ) -> Ray:
        """
        Generate camera rays: ray_origin is at pinhole and
        ray directions outward from a pixel location (somewhere withing
        a pixel pitch) to pinhole.

        Args:
            offsets:
                'center' or 0, ray will be coming from the center of a pixel
                'rand': random offset = [-0.5, 0.5)

        Returns:
            camera ray: (b, q, h, w)
        """

        *b_shape, _, _ = self.H_c2w.shape  # (b, q, 4, 4)

        ray_origins_w, ray_directions_w = utils.generate_camera_rays(
            cam_poses=self.H_c2w.reshape(-1, 4, 4),
            intrinsics=self.intrinsic.reshape(-1, 3, 3),
            width_px=self.width_px,
            height_px=self.height_px,
            subsample=subsample,
            offsets=offsets,
            device=device,
        )  # (bq, h, w, 3), (bq, h, w, 3)

        bq, h, w, _ = ray_origins_w.shape

        return Ray(
            origins_w=ray_origins_w.reshape(*b_shape, h, w, 3),  # (b, q, h, w, 3)
            directions_w=ray_directions_w.reshape(*b_shape, h, w, 3),  # (b, q, h, w, 3)
        )

    def generate_random_patch_rays(
            self,
            num_patches_per_q: int,
            patch_width_px: int,
            patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.,  # (*b,)
            patch_height_px: int = None,  # (*b,)
            patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,  # (*b,)
            int_only: bool = True,
    ) -> T.Dict[str, T.Any]:
        """
        Generate rays to form patches on the corresponding images.

        Args:
            num_patches_per_q:
                number of patches from each q
            patch_width_px:
                number of pixels in the patch in width
            patch_width_pitch_scale:
                (*b,) the pitch of the patch (new_pitch / old_pitch)
            patch_height_px:
                if None, the same as `patch_width_px`
            patch_height_pitch_scale:
                if None, the same as `patch_width_pitch_scale`
            int_only:
                whether the center is always at an integer index

        Returns:
            ray:
                (b, q, num_patches_per_q, hp, wp)
            uv:
                (b, q, num_patches_per_q, hp, wp, 2)
        """
        b, q, _41, _42 = self.H_c2w.shape
        bq = b * q

        # sample uv
        uv = utils.sample_random_patch_uv(
            b_shape=[b, q, num_patches_per_q],
            width_px=self.width_px,
            height_px=self.height_px,
            patch_width_px=patch_width_px,
            patch_width_pitch_scale=patch_width_pitch_scale,
            patch_height_px=patch_height_px,
            patch_height_pitch_scale=patch_height_pitch_scale,
            int_only=int_only,
            device=self.H_c2w.device,
        )  # (b, q, num_patches_per_q, hp, wp, 2), [0, w], [0, h]

        # use the uv to create rays
        ray_origins_w, ray_directions_w = utils.generate_camera_rays_from_uv(
            cam_poses=self.H_c2w.reshape(bq, 4, 4),  # (bq, 4, 4)
            intrinsics=self.intrinsic.reshape(bq, 3, 3),  # (bq, 3, 3)
            uv=uv.flatten(start_dim=0, end_dim=1),  # (bq, num_patches_per_q, hp, wp, 2)
            device=self.H_c2w.device,
        )  # (bq, num_patches_per_q, hp, wp, 3)

        _bq, *m_shape, _3 = ray_origins_w.shape

        ray = Ray(
            origins_w=ray_origins_w.reshape(b, q, *m_shape, 3),  # (b, q, np, hp, wp, 3)
            directions_w=ray_directions_w.reshape(b, q, *m_shape, 3),  # (b, q, np, hp, wp, 3)
        )

        return dict(
            ray=ray,  # (b, q, np, hp, wp, 3)
            uv=uv,  # (b, q, np, hp, wp, 2)
        )

    def split(self, chunk_size: int) -> T.List['Camera']:
        """
        Split camera (b, q) into a list of cameras (b', q'),
        such that b' * q' * h * w < chunk_size.

        Note that we only chunk q or chunk b

        Returns:
            list of cameras
        """
        if chunk_size < 0:
            return [self]

        hw = self.width_px * self.height_px
        N = max(1, int(chunk_size / hw))  # max bq for each chunk
        q = self.H_c2w.size(1)
        b = self.H_c2w.size(0)

        if N >= b * q:
            return [self]
        elif N > q:
            # chunk b
            chunk_dim = 0
            chunks = math.ceil(b / int(N / q))

            H_c2w_list = torch.chunk(self.H_c2w, chunks=chunks, dim=chunk_dim)
            intrinsic_list = torch.chunk(self.intrinsic, chunks=chunks, dim=chunk_dim)

            cameras = []
            for H, ins in zip(H_c2w_list, intrinsic_list):
                cameras.append(
                    Camera(
                        H_c2w=H,
                        intrinsic=ins,
                        width_px=self.width_px,
                        height_px=self.height_px,
                    )
                )
            return cameras
        else:
            # chunk b and q
            cameras = []
            for ib in range(b):
                chunk_dim = 1
                chunks = math.ceil(q / N)
                H_c2w_list = torch.chunk(self.H_c2w[ib:ib + 1], chunks=chunks, dim=chunk_dim)
                intrinsic_list = torch.chunk(self.intrinsic[ib:ib + 1], chunks=chunks, dim=chunk_dim)
                for H, ins in zip(H_c2w_list, intrinsic_list):
                    cameras.append(
                        Camera(
                            H_c2w=H,
                            intrinsic=ins,
                            width_px=self.width_px,
                            height_px=self.height_px,
                        )
                    )
            return cameras

    @torch.no_grad()
    def uniformly_sample(self, num_samples: int) -> 'Camera':
        """
        Uniformly sample more cameras from the current ones.
        Currently, we do not support gradient (though nothing stops it theoretically).

        Returns:
            new camera: (b, num_samples)
        """
        length = self.H_c2w.size(1)

        idxs = np.linspace(0, 1 - 1e-8, num_samples) * (length - 1)
        self_H_c2w = self.H_c2w.detach().cpu().numpy()  # (b, q, 4, 4)
        self_intrinsic = self.intrinsic.detach().cpu().numpy()  # (b, q, 3, 3)

        all_H_c2ws = []
        all_intrinsics = []
        for b in range(self.H_c2w.size(0)):
            H_c2ws = []
            intrinsics = []

            for i in range(len(idxs)):
                idx = idxs[i]
                idx_from = math.floor(idx)
                idx_to = idx_from + 1
                t = idx - idx_from
                H_c2w = rigid_motion.interp_homegeneous_matrices(
                    t=t,
                    H0=self_H_c2w[b, idx_from],
                    H1=self_H_c2w[b, idx_to],
                )
                H_c2w = torch.from_numpy(H_c2w)
                if torch.norm(H_c2w, p=2, dim=-2).any() <= 1e-6 or \
                        torch.logical_not(torch.norm(H_c2w, p=2, dim=-2).isfinite()).any():
                    print('oh no!')
                H_c2ws.append(H_c2w)

                intrinsic = (1 - t) * self_intrinsic[b, idx_from] + t * self_intrinsic[b, idx_to]
                intrinsics.append(torch.from_numpy(intrinsic))

            H_c2ws = torch.stack(H_c2ws, dim=0)
            intrinsics = torch.stack(intrinsics, dim=0)
            all_H_c2ws.append(H_c2ws)
            all_intrinsics.append(intrinsics)

        all_H_c2ws = torch.stack(all_H_c2ws, dim=0)
        all_intrinsics = torch.stack(all_intrinsics, dim=0)

        return Camera(
            H_c2w=all_H_c2ws.to(device=self.H_c2w.device, dtype=self.H_c2w.dtype),
            intrinsic=all_intrinsics.to(device=self.intrinsic.device, dtype=self.intrinsic.dtype),
            width_px=self.width_px,
            height_px=self.height_px,
        )

    def get_camera_frames(
            self,
            camera_frame_size: float = 0.1,
    ) -> T.List[T.List[o3d.geometry.TriangleMesh]]:
        """
        Create o3d meshes of camera frames
        """
        all_camera_frames = []
        for ib in range(self.H_c2w.size(0)):
            cam_frames = []
            for iq in range(self.H_c2w.size(1)):
                cam_frame = utils.get_o3d_camera_frame(
                    self.H_c2w[ib, iq],
                    frame_size=camera_frame_size,
                )
                cam_frames.append(cam_frame)
            all_camera_frames.append(cam_frames)
        return all_camera_frames

    def get_camera_trajectory_arrows(
            self,
            radius=0.1,
    ) -> T.List[T.List[o3d.geometry.TriangleMesh]]:
        """
        Create arrows pointing from one camera position to the next one.

        Returns:
            list of list of arrows.  (b, q-1)
        """
        all_camera_arrows = []
        for ib in range(self.H_c2w.size(0)):
            cam_arrows = []
            for iq in range(self.H_c2w.size(1) - 1):
                current_xyz_w = self.H_c2w[ib, iq, :3, 3]  # (3,)
                next_xyz_w = self.H_c2w[ib, iq + 1, :3, 3]  # (3,)
                v = next_xyz_w - current_xyz_w
                v_length = torch.linalg.norm(v, ord=2)  # (,)
                v_direction = v / (v_length + 1e-12)

                cone_height = radius * 1.5
                cylinder_height = max(0.1, v_length - cone_height)

                cam_arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=radius,
                    cone_radius=radius * 1.5,
                    cylinder_height=cylinder_height,
                    cone_height=cone_height,
                )

                # translate and rotate the arrow
                R = rigid_motion.construct_coord_frame(
                    z=v_direction,
                )
                t = current_xyz_w
                H = torch.zeros(4, 4)
                H[:3, :3] = R
                H[:3, 3] = t
                H[3, 3] = 1
                cam_arrow.transform(H)
                cam_arrows.append(cam_arrow)
            all_camera_arrows.append(cam_arrows)
        return all_camera_arrows

    def save(
            self,
            output_dir: str,
            overwrite: bool = False,
            save_ply: bool = True,
            save_individual_ply: bool = True,
            save_pt: bool = True,
            world_frame_size: float = 1.,
            camera_frame_size: float = 0.1,
            scene_meshes: T.Optional[T.List[o3d.geometry.TriangleMesh]] = None,
    ):
        if os.path.exists(output_dir) and not overwrite:
            raise RuntimeError(f'output dir {output_dir} exists')
        os.makedirs(output_dir, exist_ok=True)

        if scene_meshes is not None and not isinstance(scene_meshes, (list, tuple)):
            scene_meshes = [scene_meshes] * self.H_c2w.size(0)

        if save_pt:
            filename = os.path.join(output_dir, 'state_dict.pt')
            torch.save(self.state_dict(), filename)

        if save_individual_ply:
            # save a ply file containing world and camera coordinates
            all_camera_frames = self.get_camera_frames(camera_frame_size=camera_frame_size)

            # we are going to save individual camera frames
            for ib in range(self.H_c2w.size(0)):
                sub_dir = os.path.join(output_dir, f'batch_{ib}')
                os.makedirs(sub_dir, exist_ok=True)

                for iq in range(self.H_c2w.size(1)):
                    filename = os.path.join(sub_dir, f'{iq}.ply')
                    o3d.io.write_triangle_mesh(
                        filename=filename,
                        mesh=all_camera_frames[ib][iq],
                    )

                # save world coord
                world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=world_frame_size)
                filename = os.path.join(sub_dir, f'world.ply')
                o3d.io.write_triangle_mesh(
                    filename=filename,
                    mesh=world_frame,
                )

                # scene
                if scene_meshes is not None and scene_meshes[ib] is not None:
                    filename = os.path.join(sub_dir, f'scene.obj')
                    try:
                        o3d.io.write_triangle_mesh(
                            filename=filename,
                            mesh=scene_meshes[ib],
                        )
                    except:
                        pass

        if save_ply:
            # save a ply file containing world and camera coordinates
            all_camera_frames = self.get_camera_frames(camera_frame_size=camera_frame_size)

            # typical ply (no color)
            for ib in range(self.H_c2w.size(0)):
                cam_frames = all_camera_frames[ib]
                # world coord
                world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=world_frame_size)

                # combine all meshes
                mesh = world_frame

                # scene box
                if scene_meshes is not None:
                    try:
                        mesh = mesh + scene_meshes[ib]
                    except:
                        pass

                for cam_frame in cam_frames:
                    mesh = mesh + cam_frame

                filename = os.path.join(output_dir, f'batch_{ib}.obj')
                o3d.io.write_triangle_mesh(
                    filename=filename,
                    mesh=mesh,
                )


class RGBDImage:
    def __init__(
            self,
            rgb: torch.Tensor,  # (b, q, h, w, 3)  b: different scene, q: multiple imgs of same scene
            depth: T.Optional[torch.Tensor],  # (b, q, h, w)
            camera: Camera,  # (b, q)
            normal_w: T.Optional[torch.Tensor] = None,  # (b, q, h, w, 3)  surface normal in world coord
            hit_map: T.Optional[torch.Tensor] = None,  # (b, q, h, w)  bool, 1: valid
            feature: T.Optional[torch.Tensor] = None,  # (b, q, h, w, f)  feature
    ):
        self.rgb = rgb
        self.depth = depth
        self.camera = camera
        self.normal_w = normal_w
        self.hit_map = hit_map
        self.feature = feature

    def compute_ray_normal_dot_product(self) -> T.Union[torch.Tensor, None]:
        """
        Compute the dot product between the normal_w and the camera ray

        Returns:
            (b, q, h, w) the dot product (cos(theta)) between normal_w and camera ray
        """
        if self.camera is None or self.normal_w is None:
            return None
        ray: Ray = self.camera.generate_camera_rays(device=self.normal_w.device)  # (b, q, h, w)
        ray_direction_w = ray.directions_w  # (b, q, h, w, 3)
        dot_prod = (ray_direction_w * self.normal_w).sum(dim=-1)  # (b, q, h, w)
        return dot_prod  # can be negative

    def index_select(self, dim: int, index: torch.Tensor) -> 'RGBDImage':
        rgbd_image = self.clone()
        for attr_name in ['rgb', 'depth', 'normal_w', 'hit_map', 'camera', 'feature']:
            arr = getattr(rgbd_image, attr_name, None)
            if arr is not None:
                setattr(rgbd_image, attr_name, arr.index_select(dim=dim, index=index))
        return rgbd_image

    def chunk(self, chunks: int, dim: int = 0) -> T.List['RGBDImage']:
        out_dict = dict()
        total = None
        for attr_name in ['rgb', 'depth', 'normal_w', 'hit_map', 'camera', 'feature']:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                chunked_arr = arr.chunk(chunks=chunks, dim=dim)
                out_dict[attr_name] = chunked_arr
                if total is None:
                    total = len(chunked_arr)
                else:
                    assert len(chunked_arr) == total
        rgbd_images = []
        for i in range(total):
            d = dict()
            for attr_name in out_dict:
                d[attr_name] = out_dict[attr_name][i]
            rgbd_image = RGBDImage(**d)
            rgbd_images.append(rgbd_image)
        return rgbd_images

    def to(self, device: torch.device) -> 'RGBDImage':
        for attr_name in ['rgb', 'depth', 'normal_w', 'hit_map', 'camera', 'feature']:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.to(device=device))
        return self

    def detach(self) -> 'RGBDImage':
        for attr_name in ['rgb', 'depth', 'normal_w', 'hit_map', 'camera', 'feature']:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.detach())
        return self

    def clone(self) -> 'RGBDImage':
        return RGBDImage(
            rgb=self.rgb.clone() if self.rgb is not None else None,
            depth=self.depth.clone() if self.depth is not None else None,
            camera=self.camera.clone() if self.camera is not None else None,
            normal_w=self.normal_w.clone() if self.normal_w is not None else None,
            hit_map=self.hit_map.clone() if self.hit_map is not None else None,
            feature=self.feature.clone() if self.feature is not None else None,
        )

    @staticmethod
    def cat(rgbd_images: T.List['RGBDImage'], dim: int) -> 'RGBDImage':
        out = dict()
        for name in ['rgb', 'depth', 'normal_w', 'hit_map', 'feature']:
            arr = [getattr(r, name, None) for r in rgbd_images]
            if None in arr:
                out[name] = None
            else:
                out[name] = torch.cat(arr, dim=dim)

        # concat camera
        cameras = [getattr(r, 'camera', None) for r in rgbd_images]
        assert None not in cameras
        out['camera'] = Camera.cat(cameras, dim=1)
        return RGBDImage(**out)

    def get_pcd(
            self,
            subsample: int = 1,
            remove_background: bool = True,
            keep_img_idxs: bool = False,
    ) -> PointCloud:
        """
        Reproject RGBD pixels to world coordinate.

        Args:
            subsample:
                use 1 out of every `subsample` pixels
            remove_background:
                whether to remove background pixels (depth > 1e6).
                only has effect if b == 1.
            keep_img_idxs:
                whether to record the original image index in the point cloud
        """
        assert self.depth is not None
        # only has effect if b == 1
        valid_mask = self.hit_map  # None or (b, q, h, w)
        if remove_background:
            if valid_mask is None:
                valid_mask = self.depth < 1.e6  # (b, q, h, w)
            else:
                valid_mask = torch.logical_and(valid_mask, self.depth < 1.e6)  # (b, q, h, w)
        depth = self.depth.masked_fill(
            mask=torch.logical_not(valid_mask),
            value=INF,
        )

        if keep_img_idxs:
            b, q, h, w = self.depth.shape
            qhw = q * h * w
            img_idxs = torch.arange(qhw, device=self.depth.device).expand(b, -1)  # (b, qhw)
            img_idxs = img_idxs.reshape(b, q, h, w, 1)  # (b, q, h, w, 1)  it saves the linear index of qhw
        else:
            img_idxs = None

        other_maps = [
            self.rgb,
            self.normal_w,
            valid_mask.unsqueeze(-1),
            self.feature,
            img_idxs,
        ]

        xyz_dict = utils.compute_3d_xyz(
            z_map=depth,
            intrinsic=self.camera.intrinsic,
            H_c2w=self.camera.H_c2w,
            subsample=subsample,
            other_maps=other_maps,
        )  # (b, q, h', w', 3)  can contain nan

        points_w = xyz_dict['xyz_w']  # (b, q, h', w', 3)
        b, *qhw, c = points_w.shape
        points_w = points_w.reshape(b, -1, 3)  # (b, n, 3)
        points_rgb = xyz_dict['other_maps'][0]  # (b, q, h', w', 3)
        points_rgb = points_rgb.reshape(b, -1, 3)  # (b, n, 3)
        points_normal_w = xyz_dict['other_maps'][1]
        if points_normal_w is not None:
            points_normal_w = points_normal_w.reshape(b, -1, 3)  # (b, n, 3)
        valid_mask = xyz_dict['other_maps'][2].reshape(b, -1, 1)  # (b, n, 1)

        points_feature = xyz_dict['other_maps'][3]
        if points_feature is not None:
            dim_feature = points_feature.size(-1)
            points_feature = points_feature.reshape(b, -1, dim_feature)  # (b, n, f)

        points_img_idxs = xyz_dict['other_maps'][4]
        if points_img_idxs is not None:
            points_img_idxs = points_img_idxs.reshape(b, -1, 1)  # (b, n, 1)

        # compute ray-based features
        feature_dict = utils.compute_3d_zdir_and_dps(
            z_map=self.depth,
            intrinsic=self.camera.intrinsic,
            H_c2w=self.camera.H_c2w,
            subsample=subsample,
        )
        for key in feature_dict:
            feature_dim = feature_dict[key].size(-1)
            feature_dict[key] = feature_dict[key].reshape(b, -1, feature_dim)

        # compute view direction
        camera_pinhole_w = self.camera.H_c2w[..., :3, 3]  # (b, q, 3)
        *bq_shape, _ = camera_pinhole_w.shape
        camera_pinhole_w = camera_pinhole_w.view(*bq_shape, 1, 1, 3).expand(b, *qhw, 3).reshape(b, -1, 3)  # (b, n, 3)
        vdir_w = points_w - camera_pinhole_w  # (b, n, 3)
        vdir_w = torch.nn.functional.normalize(
            input=vdir_w,
            p=2,
            dim=-1,
        )  # (b, n, 3)

        point_cloud = PointCloud(
            xyz_w=points_w,  # (b, n, 3)
            rgb=points_rgb,  # (b, n, 3)
            normal_w=points_normal_w,  # (b, n, 3)
            captured_z_direction_w=feature_dict['zdir_w'],  # (b, n, 3)
            captured_dps=feature_dict['dps_w'],  # (b, n, 1)
            captured_dps_u_w=feature_dict['dps_uw'],  # (b, n, 3)
            captured_dps_v_w=feature_dict['dps_vw'],  # (b, n, 3)
            captured_view_direction_w=vdir_w,  # (b, n, 3)
            valid_mask=valid_mask,  # (b, n, 1)
            included_point_at_inf=False,
            feature=points_feature,  # (b, n, f)
            img_idxs=points_img_idxs,  # (b, n, 1)
        )

        if b == 1 and remove_background:
            point_cloud = point_cloud.extract_valid_point_cloud(bidx=0)

        return point_cloud

    def sample_random_patches(
            self,
            num_patches_per_q: int,
            patch_width_px: int,
            patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.,  # (*b,)
            patch_height_px: int = None,  # (*b,)
            patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,  # (*b,)
            int_only: bool = True,
            mode: str = 'bilinear',
            padding_mode: str = 'zeros',
            required_attributes: T.List[str] = None,
    ) -> T.Dict[str, T.Any]:
        """
        Generate rays to form patches on the corresponding images.

        Args:
            num_patches_per_q:
                number of patches from each q
            patch_width_px:
                number of pixels in the patch in width
            patch_width_pitch_scale:
                (*b,) the pitch of the patch (new_pitch / old_pitch)
            patch_height_px:
                if None, the same as `patch_width_px`
            patch_height_pitch_scale:
                if None, the same as `patch_width_pitch_scale`
            int_only:
                whether the center is always at an integer index
            mode:
                mode used by grid_sample
            padding_mode:
                padding mode used by grid_sample. "zeros", "border", "reflection"
            required_attributes:
                list of str containing the field wanted. If None: all possible fields

        Returns:
            ray:
                (b, q, num_patches_per_q, hp, wp)
            uv:
                (b, q, num_patches_per_q, hp, wp, 2)  [0, w], [0, h]
            rgb:
                (b, q, num_patches_per_q, hp, wp, 3)
            depth:
                (b, q, num_patches_per_q, hp, wp) or None, same coord as the original depth
            normal_w:
                (b, q, num_patches_per_q, hp, wp, 3) or None, same coord as the original surface normal
            hit_map:
                (b, q, num_patches_per_q, hp, wp) or None,  float, [0, 1],  0: not valid, 1: valid
            feature:
                (b, q, num_patches_per_q, hp, wp, f) or None

        Note:
             part of the patches may go out of bound.  The padding_mode determines the behavior
        """
        b, q, h, w, _3 = self.rgb.shape

        # sample random patch and get uv and rays
        ray_uv_dict = self.camera.generate_random_patch_rays(
            num_patches_per_q=num_patches_per_q,
            patch_width_px=patch_width_px,
            patch_width_pitch_scale=patch_width_pitch_scale,
            patch_height_px=patch_height_px,
            patch_height_pitch_scale=patch_height_pitch_scale,
            int_only=int_only,
        )  # uv:  (b, q, num_patches_per_q, hp, wp, 2) [0,w] [0,h]
        uv = ray_uv_dict['uv']

        # [0, w] -> [0, 2] -> [-1, 1]
        u = uv[..., 0] / w
        v = uv[..., 1] / h
        uv = torch.stack([u, v], dim=-1)  # (b, q, num_patches_per_q, hp, wp, 2)  [0, 1]
        _b, _q, *p_shape, _2 = uv.shape

        # interpolate individual attributes
        for key in ['rgb', 'depth', 'normal_w', 'hit_map', 'feature']:
            if required_attributes is not None and key not in required_attributes:
                continue

            arr = getattr(self, key, None)
            if arr is None:
                ray_uv_dict[key] = None
                continue

            if key in ['depth', 'hit_map']:
                squeeze = True
                arr = arr.unsqueeze(-1)
            else:
                squeeze = False

            dim = arr.size(-1)
            out = utils.uv_sampling(
                uv=uv.flatten(0, 1),  # (bq, num_patches_per_q, hp, wp, 2)  [0, 1]
                feature_map=arr.flatten(0, 1).float(),  # (bq, h, w, dim)
                mode=mode,
                padding_mode=padding_mode,
            )  # (bq, num_patches_per_q, hp, wp, dim)
            out = out.reshape(b, q, *p_shape, dim)

            if squeeze:
                out = out.squeeze(-1)

            ray_uv_dict[key] = out  # (b, q, num_patches_per_q, hp, wp, dim)

        return ray_uv_dict

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""
        to_save = dict()
        for name in ['rgb', 'depth', 'normal_w', 'hit_map', 'feature']:
            to_save[name] = getattr(self, name, None)
        to_save['camera'] = self.camera.state_dict()
        return to_save

    def load_state_dict(
            self,
            state_dict: T.Dict[str, T.Any],
    ):
        """Load the state dictionary."""
        for name in ['rgb', 'depth', 'normal_w', 'hit_map', 'feature']:
            setattr(self, name, state_dict.get(name, None))
        self.camera.load_state_dict(state_dict.get('camera', None))

    def save(
            self,
            output_dir: str,
            overwrite: bool = False,
            save_png: bool = True,
            save_pt: bool = True,
            save_gif: bool = True,
            save_video: bool = True,
            gif_fps: float = 10,
            background_color: T.Union[float, T.List[float]] = 1.,
            global_min_depth: float = None,
            global_max_depth: float = None,
            hit_only: bool = True,
    ):

        if isinstance(background_color, (int, float)):
            background_color = [background_color] * 3

        if os.path.exists(output_dir) and not overwrite:
            raise RuntimeError(f'{output_dir} exists')
        os.makedirs(output_dir, exist_ok=True)

        if save_pt:
            filename = os.path.join(output_dir, 'state_dict.pt')
            torch.save(self.state_dict(), filename)

        # deal with hit_map
        b, q, h, w, _3 = self.rgb.shape
        if hit_only and self.hit_map is not None:
            hit_map = (self.hit_map > 0.5).float()  # (b, q, h, w)
        else:
            hit_map = torch.ones(b, q, h, w, dtype=self.rgb.dtype, device=self.rgb.device)  # (b, q, h, w)

        background_img = torch.ones_like(self.rgb)  # (b, q, h, w, 3)
        for c in range(3):
            background_img[..., c] = background_color[c]

        # deal with depth
        if self.depth is not None:
            masked_depth = self.depth * hit_map
        else:
            masked_depth = None
        masked_rgb = self.rgb * hit_map.unsqueeze(-1)
        masked_rgb = masked_rgb + (1 - hit_map).unsqueeze(-1) * background_img
        if self.normal_w is not None:
            masked_normal_w = self.normal_w * hit_map.unsqueeze(-1)
            masked_normal_w = masked_normal_w + (1 - hit_map).unsqueeze(-1) * background_img
        else:
            masked_normal_w = None

        if save_png:
            # normal: [-1, 1] -> [0, 1]
            if self.normal_w is not None:
                normal_w = (masked_normal_w + 1) / 2.
            else:
                normal_w = None

            for ib in range(self.rgb.size(0)):
                sub_dir = os.path.join(output_dir, f'batch_{ib}')
                os.makedirs(sub_dir, exist_ok=True)
                if masked_depth is not None:
                    if global_min_depth is None:
                        min_depth = masked_depth[ib].min()
                    else:
                        min_depth = global_min_depth
                else:
                    min_depth = None
                if masked_depth is not None:
                    if global_max_depth is None:
                        max_depth = masked_depth[ib].max()
                    else:
                        max_depth = global_max_depth
                else:
                    max_depth = None

                for iq in range(self.rgb.size(1)):
                    # rgb
                    filename = os.path.join(sub_dir, f'rgb_{iq}.png')
                    imageio.imwrite(
                        filename,
                        (masked_rgb[ib, iq] * 255.).detach().cpu().clamp(min=0, max=255).numpy().astype(np.uint8)
                    )

                    # normalized depth
                    if masked_depth is not None:
                        filename = os.path.join(sub_dir, f'depth_{iq}.png')
                        dd = (masked_depth[ib, iq] - min_depth) / (max_depth - min_depth)
                        imageio.imwrite(
                            filename,
                            (
                                    (dd * hit_map[ib, iq] + (1 - hit_map[ib, iq].float()) * background_img[
                                        ib, iq, ..., 0]) * 255.
                            ).unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy().astype(np.uint8)
                        )

                    # normal
                    if normal_w is not None:
                        filename = os.path.join(sub_dir, f'normal_w_{iq}.png')
                        imageio.imwrite(
                            filename,
                            ((normal_w[ib, iq] * hit_map[ib, iq].unsqueeze(-1) + (
                                        1 - hit_map[ib, iq].unsqueeze(-1).float()) * background_img[
                                  ib, iq]) * 255.).detach().cpu().numpy().astype(np.uint8)
                        )

                    # hit map
                    if self.hit_map is not None:
                        filename = os.path.join(sub_dir, f'hit_map_{iq}.png')
                        imageio.imwrite(filename, (self.hit_map[ib, iq] * 255.).detach().cpu().numpy().astype(np.uint8))

        if save_gif:
            # normal: [-1, 1] -> [0, 1]
            if self.normal_w is not None:
                normal_w = (masked_normal_w + 1) / 2.
            else:
                normal_w = None

            for ib in range(self.rgb.size(0)):
                # rgb
                sub_dir = os.path.join(output_dir, 'rgb')
                os.makedirs(sub_dir, exist_ok=True)
                filename = os.path.join(sub_dir, f'batch_{ib}.gif')
                render.create_gif(
                    images=masked_rgb[ib].clamp(min=0, max=1.),
                    filename=filename,
                    fps=gif_fps,
                )

                # depth
                if masked_depth is not None:
                    if global_min_depth is None:
                        min_depth = masked_depth[ib].min()
                    else:
                        min_depth = global_min_depth

                    if global_max_depth is None:
                        max_depth = masked_depth[ib].max()
                    else:
                        max_depth = global_max_depth

                    dd = (masked_depth[ib] - min_depth) / (max_depth - min_depth)
                    sub_dir = os.path.join(output_dir, 'depth')
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f'batch_{ib}.gif')
                    render.create_gif(
                        images=(dd * hit_map[ib] + (1 - hit_map[ib].float()) * background_img[ib, ..., 0]).unsqueeze(
                            -1).expand(-1, -1, -1, 3),
                        filename=filename,
                        fps=gif_fps,
                    )

                # normal_w
                if normal_w is not None:
                    sub_dir = os.path.join(output_dir, 'normal_w')
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f'batch_{ib}.gif')
                    render.create_gif(
                        images=normal_w[ib] * hit_map[ib].unsqueeze(-1) + (1 - hit_map[ib].float().unsqueeze(-1)) *
                               background_img[ib],
                        filename=filename,
                        fps=gif_fps,
                    )

                # hit map
                if self.hit_map is not None:
                    sub_dir = os.path.join(output_dir, 'hit_map')
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f'batch_{ib}.gif')
                    render.create_gif(
                        images=self.hit_map[ib].unsqueeze(-1).expand(-1, -1, -1, 3).float(),
                        filename=filename,
                        fps=gif_fps,
                    )

        if save_video:
            # normal: [-1, 1] -> [0, 1]
            if self.normal_w is not None:
                normal_w = (masked_normal_w + 1) / 2.
            else:
                normal_w = None

            for ib in range(self.rgb.size(0)):
                # rgb
                sub_dir = os.path.join(output_dir, 'rgb')
                os.makedirs(sub_dir, exist_ok=True)
                filename = os.path.join(sub_dir, f'batch_{ib}.mp4')
                render.create_video(
                    images=masked_rgb[ib].clamp(min=0, max=1.),
                    filename=filename,
                    fps=gif_fps,
                    color_format='rgb',
                )

                # depth
                if masked_depth is not None:
                    if global_min_depth is None:
                        min_depth = masked_depth[ib].min()
                    else:
                        min_depth = global_min_depth

                    if global_max_depth is None:
                        max_depth = masked_depth[ib].max()
                    else:
                        max_depth = global_max_depth

                    dd = (masked_depth[ib] - min_depth) / (max_depth - min_depth)
                    sub_dir = os.path.join(output_dir, 'depth')
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f'batch_{ib}.mp4')
                    render.create_video(
                        images=(dd * hit_map[ib] + (1 - hit_map[ib].float()) * background_img[ib, ..., 0]).unsqueeze(
                            -1).expand(-1, -1, -1, 3),
                        filename=filename,
                        fps=gif_fps,
                        color_format='rgb',
                        val_range='01',
                    )

                # normal_w
                if normal_w is not None:
                    sub_dir = os.path.join(output_dir, 'normal_w')
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f'batch_{ib}.mp4')
                    render.create_video(
                        images=normal_w[ib] * hit_map[ib].unsqueeze(-1) + (1 - hit_map[ib].float().unsqueeze(-1)) *
                               background_img[ib],
                        filename=filename,
                        fps=gif_fps,
                        color_format='rgb',
                        val_range='01',
                    )

                # hit map
                if self.hit_map is not None:
                    sub_dir = os.path.join(output_dir, 'hit_map')
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f'batch_{ib}.video')
                    render.create_video(
                        images=self.hit_map[ib].unsqueeze(-1).expand(-1, -1, -1, 3).float(),
                        filename=filename,
                        fps=gif_fps,
                        color_format='rgb',
                        val_range='01',
                    )

    def save_as_npbgpp_input(
            self,
            output_dirs: T.List[str],
            type: str,  # 'input', 'ground-truth'
            start_idx: int = 0,
            exist_ok: bool = True,
            overwrite: bool = False,
            hit_only: bool = False,
    ):
        """
        Save as the format used by NPBG++ (https://github.com/rakhimovv/npbgpp).
        We use the DTU dataset's format:
        - image:  000000.png ... 000010.png  (all images, input and ground truth target)
        - mask:   000.png ... 010.png  (binary masks, 1: object, 0: background)
        - mvs_pc.ply:  xyz input point cloud (can be without rgb color)
        - cameras.npz:  world_mat_0 ... world_mat_10  (4x4 Projection matrix from world to image, ie, intrinsics * H_w2c)

        image background is white.
        reccommend using b==1.

        Args:
            output_dirs:
                (b,) output folders for each b

        Returns:
            list of output_dir, one per each b
        """
        background_color = 1.
        assert len(output_dirs) == self.rgb.size(0)
        for output_dir in output_dirs:
            if overwrite:
                try:
                    shutil.rmtree(output_dir)
                except:
                    pass
            if os.path.exists(output_dir) and not exist_ok:
                raise RuntimeError
            os.makedirs(output_dir, exist_ok=True)

        if type == 'input':
            save_pcd = True
        else:
            save_pcd = False

        if hit_only and self.hit_map is not None:
            hit_map = self.hit_map.float()  # (b, q, h, w)
        else:
            hit_map = torch.ones_like(self.depth)  # (b, q, h, w)

        if self.hit_map is not None:
            actual_hit_map = self.hit_map.float()  # (b, q, h, w)
        else:
            actual_hit_map = torch.ones_like(self.depth)  # (b, q, h, w)

        b, q, h, w, _3 = self.rgb.shape
        assert self.camera.width_px == w, f'self.camera.width_px = {self.camera.width_px}, w = {w}'
        assert self.camera.height_px == h, f'self.camera.height_px = {self.camera.height_px}, h = {h}'

        H_w2c = self.camera.get_H_w2c()  # (b, q, 4, 4)
        intrinsics_44 = torch.zeros_like(H_w2c)
        intrinsics_44[..., :3, :3] = self.camera.intrinsic
        intrinsics_44[..., 3, 3] = 1
        P_w2c = intrinsics_44 @ H_w2c  # (b, q, 4, 4)

        masked_rgb = self.rgb * hit_map.unsqueeze(-1)
        masked_rgb = masked_rgb + (1 - hit_map).unsqueeze(-1).expand_as(masked_rgb) * background_color
        ply_filenames = []
        for ib in range(self.rgb.size(0)):
            output_dir = output_dirs[ib]
            img_dir = os.path.join(output_dir, 'image')
            mask_dir = os.path.join(output_dir, 'mask')
            ply_filename = os.path.join(output_dir, 'mvs_pc.ply')
            camera_filename = os.path.join(output_dir, 'cameras.npz')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            camera_dict = dict()
            if os.path.exists(camera_filename):
                tmp = np.load(camera_filename)
                for n in tmp:
                    camera_dict[n] = tmp[n]

            for iq in range(self.rgb.size(1)):
                # rgb
                filename = os.path.join(img_dir, f'{start_idx + iq:06d}.png')
                imageio.imwrite(
                    filename,
                    (masked_rgb[ib, iq] * 255.).detach().cpu().numpy().astype(np.uint8)
                )
                # mask (actual hit_map if available)
                filename = os.path.join(mask_dir, f'{start_idx + iq:03d}.png')
                imageio.imwrite(filename, ((actual_hit_map[ib, iq] > 0.5) * 255.).detach().cpu().numpy().astype(np.uint8))
                # camera
                camera_dict[f'world_mat_{start_idx + iq}'] = P_w2c[ib, iq].detach().cpu().numpy()  # (4, 4)

            # save camera npz
            np.savez(camera_filename, **camera_dict)

            # point cloud
            ply_filenames.append(ply_filename)

        # save point cloud
        if save_pcd:
            point_cloud = self.get_pcd()  # (b, n, 3)
            point_cloud.save_as_npbgpp(
                filenames=ply_filenames,
                overwrite=overwrite,
            )

    def save_as_rtmv(
            self,
            output_dirs: T.List[str],
            start_idx: int = 0,
            exist_ok: bool = True,
            overwrite: bool = False,
            srgb_to_linear: bool = True,
            hit_only: bool = False,
    ) -> T.List[str]:
        """
        Save as the format used in the RTMV dataset https://www.cs.umd.edu/~mmeshry/projects/rtmv/.

        Each scene in the dataset contains
        - {id:05d}.depth.exr  # (h, w, 3) grayscale, float32, background is set to -1e10,  "ray traveling distance", not z in camera coordinate
        - {id:05d}.exr  # (h, w, 4), float32,  rgba   (alpha: foreground mask [0, 1])
        - {id:05d}.json  # camera information, see below
        - {id:05d}.seg.exr  # (h, w, 3), float32, [0, 1]  1 foreground, 0 background

        ex:
        - 00000.depth.exr
        - 00000.exr
        - 00000.json
        - 00000.seg.exr

        RTMV uses blender coordinate system (x to right, y to far, z to up).
        We use opengl's (x to right, y to up, z to us).
        So we need to first convert our H_c2w to H_b2w before saving the camera.
        Moreover, Kaolin uses intrinsic matrix to convert world coordinate
        (x to right, y to up, z to us) to image coordinate (x to right, y to down,
        z to far). So we also need to handle this.

        # Camera information
        {
            "camera_data": {
                "cam2world": [  # it is the "transpose" of H_c2w
                    [
                        -0.6331584453582764,
                        0.7740222811698914,
                        0.0,
                        0.0
                    ],
                    [
                        -0.09314906597137451,
                        -0.07619690895080566,
                        0.9927322864532471,
                        0.0
                    ],
                    [
                        0.7683968544006348,
                        0.6285567283630371,
                        0.12034416198730469,
                        0.0
                    ],
                    [
                        0.5917379856109619,
                        0.5100606083869934,
                        0.17243748903274536,
                        1.0
                    ]
                ],
                "camera_look_at": {  # in world coordinate
                    "at": [
                        -0.05198334725487073,
                        -0.016510273653732227,
                        0.0716196402346452
                    ],
                    "eye": [
                        0.5917379969124275,
                        0.5100606335385356,
                        0.17243748276105064
                    ],
                    "up": [
                        0,
                        0,
                        1
                    ]
                },
                "camera_view_matrix": [  # H_w2c.T
                    [
                        -0.6331584453582764,
                        -0.09314906597137451,
                        0.7683968544006348,
                        0.0
                    ],
                    [
                        0.7740222811698914,
                        -0.07619690895080566,
                        0.6285567283630371,
                        0.0
                    ],
                    [
                        0.0,
                        0.9927322864532471,
                        0.12034416198730469,
                        0.0
                    ],
                    [
                        -0.020134389400482178,
                        -0.07719936966896057,
                        -0.7960434556007385,
                        1.0
                    ]
                ],
                "height": 1600,
                "intrinsics": {
                    "cx": 800.0,
                    "cy": 800.0,
                    "fx": 1931.371337890625,
                    "fy": 1931.371337890625
                },
                "location_world": [
                    0.5917379856109619,
                    0.5100606083869934,
                    0.17243748903274536
                ],
                "width": 1600
            },
            "objects": []
        }

        Also note that:
        kaolin-wisp calls linear_to_srgb when loading data (but directly save data in srgb domain).

        Args:
            output_dirs:
                (b,) output folders for each b

        Returns:
            list of output_dir, one per each b
        """

        assert len(output_dirs) == self.rgb.size(0)
        for output_dir in output_dirs:
            if overwrite:
                try:
                    shutil.rmtree(output_dir)
                except:
                    pass
            if os.path.exists(output_dir) and not exist_ok:
                raise RuntimeError
            os.makedirs(output_dir, exist_ok=True)

        b, q, h, w, _3 = self.rgb.shape
        assert self.camera.width_px == w, f'self.camera.width_px = {self.camera.width_px}, w = {w}'
        assert self.camera.height_px == h, f'self.camera.height_px = {self.camera.height_px}, h = {h}'
        with torch.no_grad():
            if hit_only and self.hit_map is not None:
                hit_map = self.hit_map.float()  # (b, q, h, w)
            else:
                hit_map = torch.ones_like(self.depth)  # (b, q, h, w)

            # Our H_c2w actually contains two parts:
            # H_c2w (H_i2w) = H_c2l * H_i2c,
            # where i is the image coordinate: c: x to right, y to down, z to far
            #       c is the camara coordinate (our invariant)
            #       l is the world coordinate in OpenGL convention:
            #       l: x to right, y to up, z to us
            # However, rtmv uses blender convention:
            #       b: x to right, y to far, z to up
            # and it should not contain H_i2c.

            H_i2w = self.camera.H_c2w

            H_c2i = torch.tensor([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, 0],
                [0,  0,  0, 1],
            ]).to(dtype=torch.float, device=self.camera.H_c2w.device)
            H_c2w = H_i2w @ H_c2i.view(1, 1, 4, 4) # (b, q, 4, 4)

            # represent opengl axis in blender axis
            H_w2b = torch.tensor([
                [1,  0,  0, 0],
                [0,  0, -1, 0],
                [0,  1,  0, 0],
                [0,  0,  0, 1],
            ]).to(dtype=torch.float, device=self.camera.H_c2w.device)
            H_c2b = H_w2b.view(1, 1, 4, 4) @ H_c2w  # (b, q, 4, 4)

            rgba = torch.cat([
                self.rgb,
                hit_map.unsqueeze(-1).to(dtype=self.rgb.dtype),
            ], dim=-1).float().detach().cpu().numpy()  # (b, q, h, w, 4)

            xyz_w = utils.compute_3d_xyz(
                z_map=self.depth,  # (b, q, h, w)
                intrinsic=self.camera.intrinsic,  # (b, q, 3, 3)
                H_c2w=self.camera.H_c2w,  # (b, q, 4, 4,)
            )['xyz_w']  # (b, q, h, w, 3)

        for ib in range(self.rgb.size(0)):
            output_dir = output_dirs[ib]

            for iq in range(self.rgb.size(1)):
                # camera
                camera_filename = os.path.join(output_dir, f'{start_idx+iq:05d}.json')
                camera_dict = dict(camera_data=dict(), objects=[])
                camera_data = camera_dict['camera_data']
                camera_data['cam2world'] = H_c2b[ib, iq].t().detach().cpu().tolist()
                camera_data['camera_look_at'] = dict(
                    at=(-H_c2b[ib, iq, :3, 2] + H_c2b[ib, iq, :3, 3]).detach().cpu().tolist(),  # notice the negative
                    eye=H_c2b[ib, iq, :3, 3].detach().cpu().tolist(),
                    up=H_c2b[ib, iq, :3, 1].detach().cpu().tolist(),
                )
                camera_data['location_world'] = H_c2b[ib,iq,:3,3].detach().cpu().tolist()
                camera_data['camera_view_matrix'] = rigid_motion.inv_homogeneous_tensors(
                    H_c2b[ib, iq]).t().detach().cpu().tolist()
                camera_data['height'] = self.camera.height_px
                camera_data['width'] = self.camera.width_px
                camera_data['intrinsics'] = dict(
                    cx=self.camera.intrinsic[ib, iq, 0, 2].detach().cpu().item(),
                    cy=self.camera.intrinsic[ib, iq, 1, 2].detach().cpu().item(),
                    fx=self.camera.intrinsic[ib, iq, 0, 0].detach().cpu().item(),
                    fy=self.camera.intrinsic[ib, iq, 1, 1].detach().cpu().item(),
                )
                with open(camera_filename, 'w') as f:
                    json.dump(camera_dict, f, indent=2)

                # rgb
                filename = os.path.join(output_dir, f'{start_idx + iq:05d}.exr')
                if srgb_to_linear:
                    img = render.srgb_to_linear(img=torch.from_numpy(rgba[ib, iq, ..., :3]))  # (h, w, 3)
                    img = torch.cat([img, torch.from_numpy(rgba[ib, iq, ..., 3:4])], dim=-1).detach().cpu().numpy()  # (h, w, 4)
                    pyexr.write(filename, img)  # (h, w, 4)
                else:
                    pyexr.write(filename, rgba[ib, iq])  # (h, w, 4)

                # mask
                filename = os.path.join(output_dir, f'{start_idx + iq:05d}.seg.exr')
                pyexr.write(
                    filename,
                    hit_map[ib, iq].unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy(),
                )  # (h, w, 3)

                # depth (ray travelling distance)
                # we still use H_c2w since the distance is the same under rigid transformation
                filename = os.path.join(output_dir, f'{start_idx + iq:05d}.depth.exr')
                dist = torch.linalg.norm(
                    xyz_w[ib, iq] - self.camera.H_c2w[ib, iq, :3, 3].reshape(1, 1, 3),
                    ord=2, dim=-1, keepdim=True,
                ).expand(-1, -1, 3)  # (h, w, 3)
                dist = dist.masked_fill(hit_map[ib, iq].unsqueeze(-1) < 0.5, -1e10)
                dist = dist.detach().cpu().numpy()

                pyexr.write(filename, dist)  # (h, w, 3)

        return output_dirs

    def save_as_llff(
            self,
            output_dirs: T.List[str],
            start_idx: int = 0,
            exist_ok: bool = True,
            overwrite: bool = False,
            hit_only: bool = False,
    ) -> T.List[str]:
        """
        Save as the format used by LLFF.

        LLFF processes the outputs of COLMAP, and we will only output a subset of it.

        See: https://github.com/Fyusion/LLFF#using-your-own-poses-without-running-colmap

        Each scene in the dataset contains
        - images/
            - xxxx.jpg  (total of n (h, w, 3) images)
        - images_2/
            - xxxx.png  (total of n (h, w, 3/4) images
        - poses_bounds.npy: (n, 17)
            the first dimension is ordered by sorted filenames of files in images
            poses[:, 0:12]: vec(H_c2b) (x down, y right, z us)
            poses[:, 12:15]:  hwf in pixel
            poses[:, 15:17]: z_min z_max in the camera coordinate

        We can get the camera poses in opengl coordinate, intrinsics (hwf), z_c range by
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # (3, 5, n)
        bds = poses_arr[:, -2:].transpose([1, 0])  # (2, n)  z_near, z_far

        # Convert R matrix from the form [down right back] to [right up back]
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)  # (3, 5, n)

        poses[:3, :4] is now H_c2w in the coordinate of x to right, y to up, z to us
        poses[:3, 4] is now hwf in pixel

        intrinsic_matrix = np.array([[f, 0, w/2],
                                     [0, f, h/2],
                                     [0, 0, 1]]).astype(np.float32)

        Args:
            output_dirs:
                (b,) output folders for each b

        Returns:
            list of output_dir, one per each b
        """

        assert len(output_dirs) == self.rgb.size(0)
        for output_dir in output_dirs:
            if overwrite:
                try:
                    shutil.rmtree(output_dir)
                except:
                    pass
            if os.path.exists(output_dir) and not exist_ok:
                raise RuntimeError
            os.makedirs(output_dir, exist_ok=True)

        b, q, h, w, _3 = self.rgb.shape
        assert self.camera.width_px == w, f'self.camera.width_px = {self.camera.width_px}, w = {w}'
        assert self.camera.height_px == h, f'self.camera.height_px = {self.camera.height_px}, h = {h}'
        with torch.no_grad():
            if hit_only and self.hit_map is not None:
                hit_map = self.hit_map.float()  # (b, q, h, w)
            else:
                hit_map = torch.ones_like(self.depth)  # (b, q, h, w)

            if self.hit_map is not None:
                actual_hit_map = self.hit_map.float()  # (b, q, h, w)
            else:
                actual_hit_map = torch.ones_like(self.depth)  # (b, q, h, w)

            # Our H_c2w actually contains two parts:
            # H_c2w (H_i2w) = H_c2l * H_i2c,
            # where i is the image coordinate: c: x to right, y to down, z to far
            #       c is the camara coordinate (our invariant)
            #       l is the world coordinate in OpenGL convention:
            #       l: x to right, y to up, z to us
            # However, llff uses the coodinate:
            #       b: x to down, y to right, z to us
            # and it should not contain H_i2c.

            H_i2w = self.camera.H_c2w

            H_b2i = torch.tensor(
                [
                    [0,  1,  0, 0],
                    [1,  0,  0, 0],
                    [0,  0, -1, 0],
                    [0,  0,  0, 1],
                ]).to(dtype=torch.float, device=self.camera.H_c2w.device)
            H_b2w = H_i2w @ H_b2i.view(1, 1, 4, 4)  # (b, q, 4, 4)

            masked_rgb = self.rgb * hit_map.unsqueeze(-1)
            masked_rgb = masked_rgb + (1 - hit_map).unsqueeze(-1) # background is white

            rgba = torch.cat(
                [
                    masked_rgb,
                    hit_map.unsqueeze(-1).to(dtype=masked_rgb.dtype),
                ], dim=-1).float().detach().cpu()  # (b, q, h, w, 4)

            nan_depth = self.depth.clone()
            nan_depth[actual_hit_map < 0.5] = torch.nan

        for ib in range(self.rgb.size(0)):
            output_dir = output_dirs[ib]

            camera_filename = os.path.join(output_dir, f'poses_bounds.npy')
            if os.path.exists(camera_filename):
                ori_poses_arr = np.load(camera_filename)  # (n, 17)
            else:
                ori_poses_arr = np.zeros([0, 17], dtype=np.float32)
            assert start_idx == ori_poses_arr.shape[0]

            os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'images_2'), exist_ok=True)


            dd = self.depth[actual_hit_map > 0.5]
            if dd.numel() > 0:
                z_min_global = dd.min().detach().cpu().item()
                z_max_global = dd.min().detach().cpu().item()
            else:
                z_min_global = 1e-3
                z_max_global = 1

            poses_arrs = []
            for iq in range(self.rgb.size(1)):
                # camera poses, hwf, z_min, z_max
                cam_pose = H_b2w[ib, iq, :3].detach().cpu().numpy()  # (3, 4)
                hwf = np.array([
                    self.camera.height_px,
                    self.camera.width_px,
                    self.camera.intrinsic[ib, iq, 0, 0].detach().cpu().item(),
                ])  # (3,)

                dd = self.depth[ib, iq]
                dd = dd[actual_hit_map[ib, iq] > 0.5]
                if dd.numel() > 0:
                    z_min = dd.min().detach().cpu().item()
                    z_max = dd.max().detach().cpu().item()
                else:
                    z_min = z_min_global
                    z_max = z_max_global

                # arr = np.concatenate([cam_pose.ravel(), hwf, np.array([z_min, z_max])], axis=0)  # (17,)
                arr = np.concatenate([cam_pose, hwf.reshape(-1, 1)], axis=1)  # (3, 5)
                arr = np.concatenate([arr.ravel(), np.array([z_min, z_max])], axis=0)  # (17,)
                poses_arrs.append(arr)

                # rgb
                filename = os.path.join(output_dir, 'images', f'{start_idx + iq:05d}.jpg')
                imageio.imwrite(
                    filename,
                    (rgba[ib, iq, :, :, :3] * 255.).detach().cpu().numpy().astype(np.uint8)
                )
                filename = os.path.join(output_dir, 'images_2', f'{start_idx + iq:05d}.png')
                imageio.imwrite(
                    filename,
                    (rgba[ib, iq] * 255.).detach().cpu().numpy().astype(np.uint8)
                )

            # write camera pose to npy
            poses_arrs = np.stack(poses_arrs, axis=0)  # (iq, 17)
            poses_arrs = np.concatenate([ori_poses_arr, poses_arrs], axis=0)  # (n, 17)
            np.save(camera_filename, poses_arrs, allow_pickle=False)

        return output_dirs


class Mesh:
    def __init__(
            self,
            mesh: T.Union[o3d.geometry.TriangleMesh, str],
            scale: T.Optional[float] = 1.,
            center_w: T.Optional[T.List[float]] = (0., 0., 0.),
            preprocess_mesh: bool = True,
    ):
        if isinstance(mesh, str):
            # load mesh
            mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(
                mesh, enable_post_processing=True)

        # preprocess mesh (clean uv, shift to center, rescale to [-scale, scale])
        mesh = mesh_utils.preprocess_mesh(
            mesh=mesh,
            scale=scale,
            center_w=center_w,
            clean=preprocess_mesh
        )

        self.scale = np.max(mesh.get_axis_aligned_bounding_box().get_half_extent())
        self.center_w = mesh.get_axis_aligned_bounding_box().get_center()
        self.mesh = mesh

        # for ray tracing
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh_t)

    def replace_texture(self, texture_imgs: T.List[np.ndarray]):
        """
        Replace the texture maps in the o3d mesh.

        Args:
            texture_imgs:
                a list of texture maps (may be in a different shape than the o3d textures)
            # method:
            #     'crop_resize': if a new texture is larger in dimension -> crop; if smaller -> resize.
        """

        texture_maps = self.mesh.textures
        num_textures = len(texture_maps)

        if len(texture_imgs) != num_textures and len(texture_maps) > 0:
            warnings.warn(f'num of texture_imgs {len(texture_imgs)} != number of need {num_textures}')
            return

        new_textures = []
        for i in range(len(texture_maps)):
            new_textures.append(o3d.geometry.Image(texture_imgs[i]))
        self.mesh.textures = new_textures

    def get_rgbd_image(
            self,
            camera: Camera,
            render_normal_w: bool = True,
            device: torch.device = torch.device('cpu'),
            render_method: str = 'rasterization',
            camera_for_normal: T.Optional[Camera] = None,
    ) -> RGBDImage:
        """
        Given camera poses, return the captured RGBD images.

        Args:
            camera:
                (b, q)  already on device
            render_normal_w:
                whether to ray trace to get surface normal in world coordinate
            render_method:
                'rasterization': use o3d rasterization (may have anti-aliasing applied)
                'ray_cast': use ray_casting to sample rgb
            camera_for_normal:
                (b, q) camera for computing normal (in case intrinsic is negative at (2,2))
                used only when using rasterization.

        Returns:
            rgbdimage: (b, q)  on device
        """

        if render_method == 'rasterization':
            return self._rasterize_rendering(
                camera=camera,
                render_normal_w=render_normal_w,
                device=device,
                camera_for_normal=camera_for_normal,
            )

        elif render_method == 'ray_cast':
            # run ray intersection to get normal
            ray = camera.generate_camera_rays(device=device)  # (b, q, h, w)
            out_dict = self.get_ray_intersection(
                ray=ray,
                device=device,
            )
            rgb = out_dict['ray_rgbs']  # (b, q, h, w, 3)
            normal_w = out_dict['surface_normals_w']  # (b, q, h, w, 3)
            hit_map = out_dict['hit_map']  # (b, q, h, w)
            ray_ts = out_dict['ray_ts']  # (b, q, h, w)

            # convert ray_ts to depth
            xyz_w = ray.origins_w + ray_ts.unsqueeze(-1) * ray.directions_w  # (b, q, h, w, 3)
            xyz1_w = torch.cat((xyz_w, torch.ones_like(xyz_w[..., 0:1])), dim=-1)  # (b, q, h, w, 4)
            H_w2c = camera.get_H_w2c()  # (b, q, 4, 4)
            xyz1_c = (H_w2c.unsqueeze(2).unsqueeze(2) @ xyz1_w.unsqueeze(-1)).squeeze(-1)  # (b, q, h, w, 4)
            z_map = xyz1_c[..., 2]  # (b, q, h, w)

            valid_mask = torch.logical_and(hit_map > 0.5, z_map.isfinite())
            z_map[valid_mask.logical_not()] = INF

            return RGBDImage(
                rgb=rgb,
                depth=z_map,
                camera=camera,
                normal_w=normal_w,
                hit_map=hit_map,
            )
        else:
            raise NotImplementedError

    def _rasterize_rendering(
            self,
            camera: Camera,
            render_normal_w: bool = True,
            device: torch.device = torch.device('cpu'),
            camera_for_normal: T.Optional[Camera] = None,
    ) -> RGBDImage:
        """
        Given camera poses, return the rasterized RGBD images.

        Args:
            camera:
                (b, q)  already on device
            render_normal_w:
                whether to ray trace to get surface normal in world coordinate
            render_method:
                'rasterization': use o3d rasterization (may have anti-aliasing applied)
                'ray_cast': use ray_casting to sample rgb
            camera_for_normal:
                (b, q) camera for computing normal (in case intrinsic is negative at (2,2))

        Returns:
            rgbdimage: (b, q)  on device
        """

        intrinsic = camera.intrinsic.detach().cpu().numpy()  # (b, q, 3, 3)
        H_c2w = camera.H_c2w.detach().cpu().numpy()  # (b, q, 4, 4)
        b, q = H_c2w.shape[0], H_c2w.shape[1]
        assert H_c2w.shape[2] == 4
        assert H_c2w.shape[3] == 4

        # convert (b, q) dimensino to list
        intrinsic_list = []
        H_c2w_list = []
        for i in range(b):
            for j in range(q):
                intrinsic_list.append(intrinsic[i, j])
                H_c2w_list.append(H_c2w[i, j])

        extrinsic_matrices = [rigid_motion.RigidMotion.invert_homogeneous_matrix(H) for H in H_c2w_list]

        out_dict = render.rasterize(
            meshes=[self.mesh],
            intrinsic_matrix=intrinsic_list,
            extrinsic_matrices=extrinsic_matrices,
            width_px=camera.width_px,
            height_px=camera.height_px,
            get_point_cloud=False,
            dtype=sample_utils.get_np_dtype(camera.H_c2w.dtype),
        )
        # out_dict contains
        #   imgs: a list of (h, w, 3)  rgb
        #   z_maps: a list of (h, w)  z of the scene points in the camera coordinate
        #   hit_maps: a list of (h, w)  true: valid
        # using imgs, cam_pose, intrinsics, and z_maps, we can generate point cloud

        imgs = out_dict['imgs']  # list of (h, w, 3)
        z_maps = out_dict['z_maps']  # list of (h, w)
        hit_maps = out_dict['hit_maps']  # list of (h, w)

        # convert list of b*q back to (b,q)
        rgb = []
        depth = []
        hit_map = []
        current_idx = 0
        for i in range(b):
            tmp = np.stack(imgs[current_idx:current_idx + q], axis=0)  # (q, h, w, 3)
            rgb.append(tmp)
            tmp = np.stack(z_maps[current_idx:current_idx + q], axis=0)  # (q, h, w)
            depth.append(tmp)
            tmp = np.stack(hit_maps[current_idx:current_idx + q], axis=0)  # (q, h, w)
            hit_map.append(tmp)
            current_idx += q
        rgb = torch.from_numpy(np.stack(rgb, axis=0)).to(device=device)  # (b, q, h, w, 3)
        depth = torch.from_numpy(np.stack(depth, axis=0)).to(device=device)  # (b, q, h, w)
        hit_map = torch.from_numpy(np.stack(hit_map, axis=0)).to(device=device)  # (b, q, h, w)

        if render_normal_w:
            # run ray intersection to get normal
            if camera_for_normal is None:
                camera_for_normal = camera
            out_dict = self.get_ray_intersection(
                ray=camera_for_normal.generate_camera_rays(device=device),  # (b, q, h, w)
                device=device,
            )
            normal_w = out_dict['surface_normals_w']  # (b, q, h, w, 3)
            hit_map = torch.logical_and(hit_map, out_dict['hit_map'])  # (b, q, h, w)
        else:
            normal_w = None

        return RGBDImage(
            rgb=rgb,  # (b, q, h, w, 3)
            depth=depth,  # (b, q, h, w)
            camera=camera,
            hit_map=hit_map,  # (b, q, h, w)
            normal_w=normal_w,  # (b, q, h, w, 3)
        )

    def get_ray_intersection(
            self,
            ray: Ray,
            device: torch.device = torch.device('cpu'),
    ):
        """
        Intersect the mesh with rays to get ground truth

        Args:
            ray:
                (b, *m_shape)

        Returns:
            ray_rgbs:
                (b, *m_shape, 3)
            ray_ts:
                (b, *m_shape)
            surface_normals_w:
                (b, *m_shape, 3)  in the world coordinate
            hit_map:
                (b, *m_shape)  1: hit, 0: miss
        """

        torch_dtype = ray.origins_w.dtype
        b, *m_shape, _ = ray.origins_w.shape
        rays = torch.cat(
            (
                ray.origins_w,
                ray.directions_w,
            ), dim=-1)  # (b, *m, 6)
        rays = rays.detach().cpu().float().numpy()

        # cast the rays, get the intersections
        raycast_results = self.scene.cast_rays(rays)
        t_hits = raycast_results['t_hit'].numpy()  # (b, *m), inf if not hit the mesh
        hit_map = 1 - np.isinf(t_hits)  # (b, *m)  1 if hit a surface, 0 otherwise

        # render rgb of the ray
        if self.mesh.has_textures():
            ray_rgbs = render.interp_texture_map_from_ray_tracing_results(
                mesh=self.mesh,
                raycast_results=raycast_results,
                texture_maps=[skimage.img_as_float(np.array(img)).astype(np.float32) for img in self.mesh.textures],
                merge_textures=True,  # combine results from multiple textures.
            )[0]
        else:
            ray_rgbs = np.ones((b, *m_shape, 3), dtype=np.float32)

        # note that primitive_normals is the normal of the triangle face
        # we can use uv map to interpolate vertex normal
        # interpolate surface normal using uv map to get better normal estimation
        if self.mesh.has_vertex_normals():
            surface_normals = render.interp_surface_normal_from_ray_tracing_results(
                mesh=self.mesh,
                raycast_results=raycast_results,
            )  # (b, *m, 3)
        else:
            surface_normals = raycast_results['primitive_normals'].numpy()  # (b, *m, 3)

        # if not hit a surface, set surface normal to (0, 0, 0)
        surface_normals = surface_normals * np.expand_dims(hit_map, axis=-1)  # (b, *m, 3)

        # normalize surface normal, and avoid dividing zero by zero
        # note that if not hit a surface, surface normal is set to (0, 0, 0)
        surface_normals_norm = np.linalg.norm(surface_normals, ord=2, axis=-1, keepdims=True)  # (b, *m, 1)
        surface_normals_norm = np.repeat(surface_normals_norm, 3, axis=-1)  # (b, *m, 3)
        valid_mask = surface_normals_norm != 0  # (b, *m, 3)
        surface_normals[valid_mask] = surface_normals[valid_mask] / surface_normals_norm[valid_mask]  # (b, *m, 3)

        # make sure surface normal points to the ray origin (opposite direction of ray_direction)
        ray_directions = ray.directions_w.detach().cpu().numpy()  # (b, *m, 3)
        surface_normals = surface_normals * \
                          (-1 * np.sign(np.sum(surface_normals * ray_directions, axis=-1, keepdims=True)))

        # convert to tensor
        ray_rgbs = torch.from_numpy(ray_rgbs).to(dtype=torch.float, device=device)  # (b, *m, 3)
        ray_ts = torch.from_numpy(t_hits).to(dtype=torch.float, device=device)  # (b, *m)
        surface_normals = torch.from_numpy(surface_normals).to(dtype=torch.float, device=device)  # (b, *m, 3)
        hit_map = torch.from_numpy(hit_map).to(
            dtype=torch.float, device=device)  # (b, *m) 1 if hit a surface, 0 otherwise

        return dict(
            ray_rgbs=ray_rgbs,
            ray_ts=ray_ts,
            surface_normals_w=surface_normals,
            hit_map=hit_map,
        )

    def sample_point_cloud(
            self,
            num_points: int,
            method: str = 'poisson_disk',
            device: torch.device = torch.device('cpu'),
            width_px: int = 10,
            height_px: int = 10,
            fov: float = 60.,  # degree
            dtype: np.dtype = np.float32,
    ) -> T.Dict[str, T.Any]:
        """
        Sample the mesh to create a point cloud
        Args:
            num_points:
                number of point to sample
            method:
                'poisson_disk'
                'uniform_camera'

        Returns:
            point_cloud: (1, num_points)
            rgbd_image: (1, num_img)
            camera: (1, num_img)

        """
        if dtype in {np.float32, float}:
            torch_dtype = torch.float32
        elif dtype == np.float64:
            torch_dtype = torch.float64
        else:
            raise NotImplementedError

        if method == 'poisson_disk':
            o3d_pcd = self.mesh.sample_points_poisson_disk(int(num_points))
            # create rays to get uv and texture -> color of points
            xyz_w = np.array(o3d_pcd.points, dtype=dtype)  # (n, 3)
            ray_ends = torch.from_numpy(xyz_w).to(dtype=torch_dtype, device=device).unsqueeze(0)  # (1, n, 3)
            ray_directions = torch.ones_like(ray_ends)  # (1, n, 3)
            ray_ts = torch.ones(1, ray_ends.size(1), dtype=torch_dtype, device=device) * 1.0e-5  # (1, n)
            ray_origins = ray_ends - ray_directions * ray_ts.unsqueeze(-1)
            ray = Ray(
                origins_w=ray_origins,
                directions_w=ray_directions,
            )
            out_dict = self.get_ray_intersection(
                ray=ray,
                device=device,
            )
            xyz_w = ray_origins + out_dict['ray_ts'].unsqueeze(-1) * ray_directions
            idxs = (out_dict['hit_map'].squeeze(0) > 0.5)  # (n,)

            point_cloud = PointCloud(
                xyz_w=xyz_w[:, idxs],  # (1, n, 3)
                rgb=out_dict['ray_rgbs'][:, idxs],  # (1, n, 3)
                normal_w=out_dict['surface_normals_w'][:, idxs],  # (1, n, 3)
            )
            rgbd_image = None
            camera = None

        elif 'uniform_camera':
            # adjust resolution settings
            n_imgs = max(1, num_points // (width_px * height_px))
            n_pixels_per_img = num_points / n_imgs
            width_px = max(2, math.floor(n_pixels_per_img / (width_px * height_px) * width_px))
            width_px = max(2, width_px - (width_px % 2))
            height_px = max(2, math.floor(n_pixels_per_img / width_px))
            height_px = max(2, height_px - (height_px % 2))

            # get mesh scale and center
            cs = self.center_w
            s = self.scale

            # create uniformly placed camera
            camera = CameraTrajectory(
                mode='random',
                n_imgs=n_imgs,
                total=1,
                params=dict(
                    max_angle=180,
                    min_r=2 * s,
                    max_r=2 * s + 1.e-9,
                    origin_w=cs.tolist(),
                    method='LatinHypercube',
                ),
                dtype=dtype,
            ).get_camera(
                fov=fov,
                width_px=width_px,
                height_px=height_px,
                device=device,
            )
            rgbd_image = self.get_rgbd_image(
                camera=camera,
                render_method='ray_cast',  # 'ray_cast',
                device=device,
            )
            point_cloud = rgbd_image.get_pcd()

        else:
            raise NotImplementedError

        return dict(
            point_cloud=point_cloud,
            rgbd_image=rgbd_image,
            camera=camera,
        )


class CameraTrajectory:
    """
    CameraTrajectory is a pattern of camera poses
    """

    def __init__(
            self,
            mode: str,
            n_imgs: int,
            total: int,
            rng_seed: T.Union[np.random.RandomState, int] = 0,
            params: T.Dict[str, T.Any] = None,
            dtype: np.dtype = np.float32,
    ):
        """
        Args:
            mode:

            n_imgs:
                number of cameras in a set
            total:
                total number of sets
            rng_seed:
                random seed
            params:
                parameters for the mode
        """
        self.mode = mode
        self.n_imgs = n_imgs
        self.total = total
        self.np_dtype = sample_utils.get_np_dtype(dtype)
        self.torch_dtype = sample_utils.get_torch_dtype(dtype)

        if rng_seed is not None:
            if isinstance(rng_seed, int):
                self.rng = np.random.RandomState(seed=rng_seed)
            elif isinstance(rng_seed, np.random.RandomState):
                self.rng = rng_seed
            else:
                self.rng = rng_seed
        else:
            self.rng = np.random

        if params is None:
            params = dict()

        self.params = params

        if self.mode == 'assign':
            assert self.params.get('H_c2w', None) is not None
            H_c2w = self.params['H_c2w']
            if H_c2w.ndim == 3:
                self.n_imgs = H_c2w.size(0)
                self.cam_poses = H_c2w
            elif H_c2w.ndim == 4:
                self.n_imgs = H_c2w.size(1)
                self.total = H_c2w.size(0)
                self.cam_poses = H_c2w
            else:
                raise NotImplementedError
        elif self.mode == 'random':
            # within random camera in a random cone
            self._set_random()
        elif self.mode == 'circle':
            self._set_circle()
        elif self.mode == 'udlrfb':
            self._set_udlrfb()
        elif self.mode == 'spiral':
            self._set_spiral()
        elif self.mode == 'sketchfab_poisson':
            raise NotImplementedError
        elif self.mode == 'rex_in':
            raise NotImplementedError
        elif self.mode == 'rect':
            raise NotImplementedError
        elif self.mode == 'basic':
            raise NotImplementedError
        elif self.mode == 'grid':
            raise NotImplementedError
        elif self.mode == 'polar_grid':
            raise NotImplementedError
        elif self.mode == 'manual':
            self._set_manual()
        elif self.mode.lower().endswith('.pt') or self.mode.lower().endswith('.pth'):
            # state dict of Camera
            # filename of the camera pose pt file
            camera = Camera(H_c2w=None, intrinsic=None, width_px=None, height_px=None)
            checkpoint = torch.load(self.mode, map_location=torch.device('cpu'))
            camera.load_state_dict(checkpoint)
            # uniformly sample the path
            if self.n_imgs is not None:
                camera = camera.uniformly_sample(num_samples=self.n_imgs)
                H_c2w = camera.H_c2w  # (b, q, 4, 4)
                self.cam_poses = H_c2w
            else:
                self.n_imgs = camera.H_c2w.size(1)
                self.cam_poses = camera.H_c2w
        elif self.mode.lower().endswith('.json'):
            camera = Camera.load_json(filename=self.mode)
            if self.n_imgs is not None:
                # uniformly sample the path
                camera = camera.uniformly_sample(num_samples=self.n_imgs)
                H_c2w = camera.H_c2w  # (b, q, 4, 4)
                self.cam_poses = H_c2w
            else:
                self.n_imgs = camera.H_c2w.size(1)
                self.cam_poses = camera.H_c2w
        else:
            raise NotImplementedError

        assert self.n_imgs is not None

    def _set_random(self):
        assert 'max_angle' in self.params
        assert 'min_r' in self.params
        assert 'max_r' in self.params

        self.cam_poses: T.List[T.List[np.ndarray]] = [
            rigid_motion.generate_random_camera_poses(
                n=self.n_imgs,
                max_angle=self.params.get('max_angle'),
                min_r=self.params.get('min_r'),
                max_r=self.params.get('max_r'),
                center_direction_w=self.params.get('center_direction_w', None),
                local_max_angle=self.params.get('local_max_angle', 0),
                rand_r=self.params.get('rand_r', 0),
                origin_w=self.params.get('origin_w', None),
                rng=self.rng,
                method=self.params.get('method', 'random'),
                dtype=self.np_dtype,
            )
            for _ in range(self.total)
        ]  # list of list of H_c2w
        self.cam_poses = utils.to_dtype(
            utils.to_tensor(self.cam_poses),
            dtype=self.torch_dtype,
        )

    def _set_circle(self):

        self.cam_poses = []  # (b, q)
        for i in range(self.total):
            poses = []
            center_angles = self.params.get('center_angles', None)  # (2,) in degree,  (to_x, to_z)
            if center_angles is None:
                # determine random center direction
                center_angles = self.rng.rand(2) * 360.  # angle in degree

            # determine random d for input_imgs
            d = self.params.get('d', None)  # (,)
            if d is None:
                assert 'min_r' in self.params
                assert 'max_r' in self.params
                max_r = self.params['max_r']
                min_r = self.params['min_r']
                d = self.rng.rand(1) * (max_r - min_r) + min_r

            # determine circle r for input_imgs
            r = self.params.get('r', None)  # (,)
            if r is None:
                assert 'max_angle' in self.params
                max_angle = self.params['max_angle']
                r = self.rng.rand(1) * np.tan(max_angle * np.pi / 180.) * d

            # generate input camera path
            Hs_c2w = utils.generate_camera_circle_path(
                num_poses=self.n_imgs,
                d_to_origin=d,
                r_circle=r,
                center_angles=center_angles,
            )  # (n, 4, 4)
            for j in range(Hs_c2w.size(0)):
                poses.append(Hs_c2w[j])  # list of H_c2w
            self.cam_poses.append(poses)  # list of list of H_c2w

    def _set_udlrfb(self):
        # fixed 6 input views: up down left right front back
        assert 'min_r' in self.params
        assert 'max_r' in self.params
        max_r = self.params['max_r']
        min_r = self.params['min_r']

        assert self.n_imgs == 6
        self.cam_poses = []  # (total, n_imgs, 4, 4)
        for i in range(self.total):
            r = self.rng.rand(1) * (max_r - min_r) + min_r
            poses = []

            Hs_c2w_ud = utils.generate_camera_circle_path(
                num_poses=3,
                d_to_origin=0,
                r_circle=r,
                center_angles=[0, 0],
                alt_yaxis=True,
            )  # (n, 4, 4)
            Hs_c2w_lrfb = utils.generate_camera_circle_path(
                num_poses=5,
                d_to_origin=0,
                r_circle=r,
                center_angles=[0, 90],
                alt_yaxis=True,
            )  # (n, 4, 4)
            poses.append(Hs_c2w_ud[0])  # u
            for j in range(Hs_c2w_lrfb.size(0) - 1):  # lfrb
                poses.append(Hs_c2w_lrfb[j])
            poses.append(Hs_c2w_ud[1])  # d
            self.cam_poses.append(poses)

    def _set_spiral(self):
        assert 'min_r' in self.params
        assert 'max_r' in self.params
        max_r = self.params['max_r']
        min_r = self.params['min_r']
        num_circle = self.params.get('num_circle', 4)
        r_freq = self.params.get('r_freq', 1)

        self.cam_poses = []
        for i in range(self.total):
            Hs_c2w_spiral = utils.generate_camera_spiral_path(
                num_poses=self.n_imgs,
                num_circle=num_circle,
                init_phi=np.pi / 2,
                r_min=min_r,
                r_max=max_r,
                r_freq=r_freq,
                center_angles=[-90, 0],
            )

            poses = []
            for j in range(Hs_c2w_spiral.size(0)):
                poses.append(Hs_c2w_spiral[j])
            self.cam_poses.append(poses)

    def _set_sketchfab_poisson(self):

        self.cam_poses = []

        for i in range(self.total):
            r = 2.2  # fixed, ignore input arguments regarding r
            poses = []
            target_poses_neighbors = []
            # for target camera
            Hs_c2w_lrfb = utils.generate_camera_circle_path(
                num_poses=self.n_imgs,
                d_to_origin=0,
                r_circle=r,
                center_angles=[0, -90],
                alt_yaxis=True,
            )  # (n, 4, 4)
            for j in range(Hs_c2w_lrfb.size(0) - 1):  # lfrb
                poses.append(Hs_c2w_lrfb[j])
            self.cam_poses.append(poses)

    def _set_rex_in(self):
        # assumption:
        # input is from 14 input cameras.
        # mesh size (-4, 4)
        assert 'max_r' in self.params
        max_r = self.params['max_r']

        self.cam_poses = []
        for i in range(self.total):
            # r = max_r
            poses = []
            target_poses_neighbors = []

            # sample input from grid points on polar axis
            # details are in self.cam_path_mode == 'polar_grid':

            num_phi = int(np.ceil((np.sqrt(2 * self.n_imgs - 3) + 3.) / 2.))
            num_theta = 2 * (num_phi - 1)
            total_imgs = num_theta * (num_phi - 2) + 2

            Hs_c2w, neighbor_ids = utils.generate_camera_polar_grids(
                num_phi=num_phi,
                num_theta=num_theta,
                r=max_r
            )  # (n, 4, 4)

            for j in range(self.n_imgs):
                poses.append(Hs_c2w[j])

            self.cam_poses.append(poses)

    def _set_rect(self):
        assert self.total == 1
        for i in range(self.total):
            poses = []

            # note that the center can be adjusted here
            # camera will face the center
            Hs_c2w_rect = utils.generate_camera_rect_path(
                num_poses=self.n_imgs,
                d_to_origin=0,
                x_length=7,
                y_length=7,
                x_center=0,
                y_center=2,
                center_angles=[-90, 0],
                alt_yaxis=True,
            )

            for j in range(Hs_c2w_rect.size(0)):
                poses.append(Hs_c2w_rect[j])

            self.cam_poses.append(poses)

    def _set_basic(self):
        assert 'min_r' in self.params
        assert 'max_r' in self.params
        max_r = self.params['max_r']
        min_r = self.params['min_r']

        # both input and target are around the same circle path
        self.cam_poses = []
        for i in range(self.total):
            poses = []

            Hs_c2w_input = utils.generate_camera_circle_path(
                num_poses=self.n_imgs,
                d_to_origin=-max_r / 2,
                r_circle=max_r,
                center_angles=[-90, 0],
                alt_yaxis=True,
            )  # (n, 4, 4)

            for j in range(Hs_c2w_input.size(0) - 1):  # lfrb
                poses.append(Hs_c2w_input[j])
            self.cam_poses.append(poses)

    def _set_grid(self):
        # build camera grid around one point
        # currently only used in one case in inverse rendering

        assert 'min_r' in self.params
        assert 'max_r' in self.params
        max_r = self.params['max_r']
        min_r = self.params['min_r']

        grid_width = int(np.ceil(np.sqrt(self.n_imgs)))
        total_imgs = grid_width * grid_width

        self.cam_poses = []
        for i in range(self.total):
            poses = []

            Hs_c2w = utils.generate_camera_grids(
                num_x=grid_width,
                num_y=grid_width,
                cam_position_center=np.array([-max_r, 0, 0]),
            )  # (n, 4, 4)

            for j in np.random.permutation(total_imgs)[range(self.n_imgs)]:
                poses.append(Hs_c2w[j])

            self.cam_poses.append(poses)

    def _set_polar_grid(self):
        # both input and target are around the same circle path
        # sample num_theta points from [0,2*pi]
        # sample num_phi points from [0,pi], including two polars (0,pi)
        # assume sample theta and phi contain same number of angles on circle, 2*(num_phi-1) = num_theta
        # total point: num_theta* (num_phi-2) +2 = 2*num_phi^2 -6*num_phi +6
        # to make total point> self.n_target_imgs, num_phi >= (sqrt(2*self.n_imgs-3)+3)/2
        # common choice: self.n_target_imgs = 6, 14, 26, 42, ...
        # corresponding to num_phi = 3, 4, 5, 6,....
        assert 'min_r' in self.params
        assert 'max_r' in self.params
        max_r = self.params['max_r']
        min_r = self.params['min_r']

        assert self.n_imgs >= 3, 'polar grid sample at least 3 points'

        num_phi = int(np.ceil((np.sqrt(2 * self.n_imgs - 3) + 3.) / 2.))
        num_theta = 2 * (num_phi - 1)
        total_imgs = num_theta * (num_phi - 2) + 2

        self.cam_poses = []
        # cam_poses_neighbors: used in inverse rendering
        # neighbor camera positions to be updated
        # note that self.cam_poses_neighbors is different from self.target_cam_poses_neighbors
        # the former are used in inverse rendering, the later are used in multivew geometry
        # but their concept are the same
        self.cam_poses_neighbors = []

        for i in range(self.total):
            poses = []

            Hs_c2w, neighbor_ids = utils.generate_camera_polar_grids(
                num_phi=num_phi,
                num_theta=num_theta,
                r=max_r
            )  # (n, 4, 4)

            for j in range(self.n_imgs):
                poses.append(Hs_c2w[j])

            self.cam_poses.append(poses)

    def _set_manual(self):
        """
        Manually assign camera
            eye: list of (3,) where the cameras are (before global transform)
            up:  None (assume to be (0,1,0)), (1,3) used for all cameras, or list of (3,)
            look_at:  None (assume to be (0,0,0)), (1, 3) used for all cameras, or list of (3,)
            t_c2w:  (3,) or None
            y_c2w:  (3,) or None
            z_c2w:  (3,) or None
        """
        assert 'eye' in self.params
        eyes = self.params['eye']
        eyes = [[float(i) for i in eye.split(' ')] for eye in eyes]
        eyes = torch.tensor(eyes).float().reshape(-1, 3)  # (q, 3)
        assert self.n_imgs == eyes.size(0)

        ups = self.params.get('up', None)
        if ups is None:
            ups = [0, 1., 0]
        else:
            ups = [[float(i) for i in x.split(' ')] for x in ups]
        ups = torch.tensor(ups).float().reshape(-1, 3)  # (q, 3)
        if ups.size(0) == 1:
            ups = ups.expand_as(eyes)  # (q, 3)

        look_ats = self.params.get('look_at', None)
        if look_ats is None:
            look_ats = [0, 0., 0]
        else:
            look_ats = [[float(i) for i in x.split(' ')] for x in look_ats]
        look_ats = torch.tensor(look_ats).float().reshape(-1, 3)  # (q, 3)
        if look_ats.size(0) == 1:
            look_ats = look_ats.expand_as(eyes)  # (q, 3)

        t_c2w = self.params.get('t_c2w', None)
        if t_c2w is None:
            t_c2w = torch.zeros(3)  # (3,)
        else:
            t_c2w = [float(i) for i in t_c2w.split(' ')]
            t_c2w = torch.tensor(t_c2w).float()
        y_c2w = self.params.get('y_c2w', None)
        if y_c2w is None:
            y_c2w = torch.tensor([0, 1, 0]).float()  # (3,)
        else:
            y_c2w = [float(i) for i in y_c2w.split(' ')]
            y_c2w = torch.tensor(y_c2w).float()
        z_c2w = self.params.get('z_c2w', None)
        if z_c2w is None:
            z_c2w = torch.tensor([0, 0, 1]).float()  # (3,)
        else:
            z_c2w = [float(i) for i in z_c2w.split(' ')]
            z_c2w = torch.tensor(z_c2w).float()
        R_c2w = rigid_motion.construct_coord_frame(
            z=z_c2w,
            y=y_c2w,
        )
        H_c2w_global = torch.zeros(4, 4)
        H_c2w_global[:3, :3] = R_c2w
        H_c2w_global[:3, 3] = t_c2w
        H_c2w_global[3, 3] = 1

        self.cam_poses = []  # (total, q)
        for i in range(self.total):
            H_c2ws = rigid_motion.get_H_c2w_lookat(
                pinhole_location_w=eyes,  # (q, 3)
                look_at_w=look_ats,  # (q, 3)
                up_w=ups,  # (q, 3)
                invert_y=True,
            )  # (q, 4, 4)

            H_c2ws = H_c2w_global.unsqueeze(0) @ H_c2ws  # (q, 4, 4)
            self.cam_poses.append(H_c2ws)

    @staticmethod
    def get_spiral_trajectory(
            H_c2w: torch.Tensor,
            period: int,
            radius: float,
    ) -> 'CameraTrajectory':
        """
        Given a trajectory of camera poses, create a trajectory that is
        a spiral near the trajectory.

        The function only moves the camera center. It does not change the
        rotation matrix.

        Args:
            H_c2w:
                (b, q, 4, 4), q >= 2
            period:
                number of cam_poses (q) to finish a full circle
            radius:
                how large the circles are

        Returns:
            a trajectory (containing self.cam_poses (b, q', 4, 4))
        """

        b, q, _41, _42 = H_c2w.shape
        assert q >= 2

        # figure out z direction
        cs = H_c2w[:, :-1, :3, 3]  # (b, q-1, 3)
        cs_next = H_c2w[:, 1:, :3, 3]  # (b, q-1, 3)
        delta_zs = cs_next - cs  # (b, q-1, 3)
        delta_zs = torch.cat([delta_zs, delta_zs[:, -1:]], dim=-2)  # (b, q, 3)
        dzs = torch.nn.functional.normalize(delta_zs, p=2, dim=-1)  # (b, q, 3)

        # decide y and x direction
        dys = torch.zeros_like(dzs)  # (b, q, 3)
        dys[..., 1] = 1  # (b, q, 3)
        coord_frames = rigid_motion.construct_coord_frame(
            z=dzs,  # (b, q, 3)
            y=dys,
        )  # (b, q, 3, 3)
        dxs = coord_frames[..., 0]  # (b, q, 3)
        dys = coord_frames[..., 1]  # (b, q, 3)

        # create circle shift
        thetas = torch.linspace(start=0., end=2 * torch.pi, steps=period)  # (period,)
        xs = torch.cos(thetas) * radius  # (period, )
        ys = torch.sin(thetas) * radius  # (period, )

        xs = xs.repeat((q + period - 1) // period)[:q]  # (q, )
        ys = ys.repeat((q + period - 1) // period)[:q]  # (q, )

        shift = dxs * xs.view(1, q, 1) + dys * ys.view(1, q, 1)  # (b, q, 3)

        new_H_c2w = H_c2w.clone()  # (b, q, 4, 4)
        new_H_c2w[:, :, :3, 3] = new_H_c2w[:, :, :3, 3] + shift

        return CameraTrajectory(
            mode='assign',
            n_imgs=None,
            total=None,
            params=dict(H_c2w=new_H_c2w),
        )

    def get_camera(
            self,
            fov: float,  # in degree
            width_px: int,
            height_px: int,
            device: torch.device = torch.device('cpu'),
    ) -> Camera:
        """
        Returns cameras in the trajactory
        """

        intrinsics = render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=fov,
            dtype=self.np_dtype,
        )  # (3, 3) np.ndarray
        intrinsics = torch.from_numpy(intrinsics).to(device=device)  # (3, 3)

        if isinstance(self.cam_poses, (list, tuple)):
            H_c2w = []
            for i in range(len(self.cam_poses)):
                poses = [pose for pose in self.cam_poses[i]]
                H = torch.stack(poses, dim=0)  # (n_img, 4, 4)
                H_c2w.append(H)
            H_c2w = torch.stack(H_c2w, dim=0).to(device=device)  # (total, n_img, 4, 4)
        elif isinstance(self.cam_poses, torch.Tensor):
            if self.cam_poses.ndim == 3:
                H_c2w = self.cam_poses.unsqueeze(0)  # (1, q, 4, 4)
            elif self.cam_poses.ndim == 2:
                H_c2w = self.cam_poses.view(1, 1, 4, 4)  # (1, 1, 4, 4)
            else:
                assert self.cam_poses.ndim == 4
                H_c2w = self.cam_poses
        elif isinstance(self.cam_poses, np.ndarray):
            self.cam_poses = torch.tensor(self.cam_poses, dtype=self.torch_dtype)
            if self.cam_poses.ndim == 3:
                H_c2w = self.cam_poses.unsqueeze(0)  # (1, q, 4, 4)
            elif self.cam_poses.ndim == 2:
                H_c2w = self.cam_poses.view(1, 1, 4, 4)  # (1, 1, 4, 4)
            else:
                assert self.cam_poses.ndim == 4
                H_c2w = self.cam_poses
        else:
            raise NotImplementedError

        *b_shape, _, _ = H_c2w.shape

        return Camera(
            H_c2w=H_c2w,
            intrinsic=intrinsics.expand(*b_shape, 3, 3),
            width_px=width_px,
            height_px=height_px,
        )


class ColorCorrector(torch.nn.Module):
    def __init__(
            self,
            correction_type: str = 'wrgb',
    ):
        """
        Apply the color correction to an rgbd_image

        Args:
            correction_type:
                'wrgb': the correction is 3 scalars \in [0, 1] that multiply to RGB channels separately
                'identify': do nothing
        """
        super().__init__()
        self.correction_type = correction_type
        if self.correction_type == 'wrgb':
            self.wrgb = torch.nn.parameter.Parameter(torch.ones(3))
        elif self.correction_type == 'identify':
            self.register_buffer('wrgb', torch.ones(3))
        else:
            raise NotImplementedError

    def get_extra_state(self):
        return dict(
            correction_type=self.correction_type,
        )

    def set_extra_state(self, state):
        self.correction_type = state['correction_type']

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        apply the color correction
        Args:
            x:
                (*, 3)
        Returns:
            (*, 3) corrected x
        """
        if self.correction_type == 'wrgb':
            y = x * self.wrgb.reshape(*([1] * (x.ndim - 1)), -1)
            return y
        elif self.correction_type == 'identify':
            return x
        else:
            raise NotImplementedError
