#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# This file implements a dataloader for hypersim dataset.


import csv
import os
import typing as T
from glob import glob

import h5py
import numpy as np
import torch

from plib import rigid_motion
from pointersect.inference import structures

trove_root_dir = 'trove'
camera_csv_filename = 'metadata_camera_parameters.csv'
trove_mounted = False


def get_hypersim_scene_info(
        volume_id: int,
        scene_id: int,
        dataset_root_dir: str,
) -> T.Tuple[str, str]:
    """
    Returns the scene dir and name (eg, "ai_VVV_NNN", "ai_001_001") .
    Args:
        volume_id:
        scene_id:
        dataset_root_dir:

    Returns:
    """
    scene_name = f"ai_{volume_id:03d}_{scene_id:03d}"
    scene_dir = os.path.join(
        dataset_root_dir,
        scene_name,
    )
    return scene_dir, scene_name


def load_hypersim_camera_info(
        filename: str,
) -> T.Dict[str, T.Dict[str, T.Any]]:
    """
    Load the csv file containing the camera information about each scene.

    Args:
        filename:
            csv file containing the camera information

    Returns:
        all_camera_info:
            a dict from scene_name 'ai_001_001' -> info_dict.
    """

    # load the csv file
    all_camera_info_dict = dict()  # scene_name (ai_001_001) -> info_dict
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            d = dict(row)
            all_camera_info_dict[d["scene_name"]] = d
    return all_camera_info_dict


def get_hypersim_camera_names(
        volume_id: T.Optional[int],
        scene_id: T.Optional[int],
        scene_dir: str = None,
        dataset_root_dir: str = 'datasets/hypersim',
) -> T.List[str]:
    """Returns the camera name in a scene, eg, cam_00, cam_01, etc. """
    if scene_dir is None:
        scene_dir, scene_name = get_hypersim_scene_info(
            volume_id=volume_id,
            scene_id=scene_id,
            dataset_root_dir=dataset_root_dir,
        )
    # print(f'{os.path.join(scene_dir, "_detail", "cam_*")}')
    cam_names = glob(os.path.join(scene_dir, "_detail", "cam_*"))
    cam_names = [os.path.basename(n) for n in cam_names]  # "cam_00", "cam_01", ...
    cam_names = sorted(cam_names)
    return cam_names


class HypersimDataset:
    """
    Dataset content: https://github.com/apple/ml-hypersim

    We will load the data to the following convention:
    For world coordinate: x-axis to right, y-axis to up, and z-axis to us.
    For image coordinate: u-axis to right, v-axis to down. Origin (0,0,) is at top left of the image.
    We will incorporate the flipping of y->v axis in H_c2w (the camera pose homogeneous matrix
    from the camera coordinate to the world coordinate).
    """

    def __init__(
            self,
            scene_dir: str,
            camera_info: T.Dict[str, T.Any],
            cam_idx: int = -1,
            num_images_per_item: int = None,
            image_subsample: int = 1,
            tonemap: bool = True,
            tonemap_gamma: float = 1. / 2.2,
            tonemap_percentile: float = 90,
            tonemap_percentile_target: float = 0.7,
            tonemap_scale: float = None,
            num_hold_out_items: int = 0,
    ):
        """

        Args:
            scene_dir:
                the root dir of the scene,
                e.g., /mnt/task_runtime/trove/ml-hypersim-1.0.0/data/raw/ai_001_001.
            camera_info:
                a dict containing the corresponding row of metadata_camera_parameters.csv
            cam_idx:
                camera index, e.g., 0, 1.  If -1, combine all cameras.
            num_images_per_item:
                number of images/frame_ids to group and form a single sample.
                If None, use all
                If negative, it will divide the total samples into `-num_images_per_item` samples.
            image_subsample: int = 1

            tonemap_gamma:
                for tone mapping hdr color
            tonemap_percentile:
                for tone mapping hdr color
            tonemap_percentile_target:
                for tone mapping hdr color.  We will set the n-th percentile brightness to the target.

            num_hold_out_items:
                number of hold out items to keep in the last sample
        Ref:
        https://github.com/apple/ml-hypersim/blob/main/contrib/mikeroberts3000/jupyter/01_casting_rays_that_match_hypersim_images.ipynb
        """

        self.scene_dir = scene_dir
        self.camera_info = camera_info
        self.cam_idx = cam_idx
        self.image_subsample = image_subsample
        self.num_images_per_item = num_images_per_item
        self.tonemap = tonemap
        self.tonemap_gamma = tonemap_gamma
        self.tonemap_percentile = tonemap_percentile
        self.tonemap_percentile_target = tonemap_percentile_target
        self.tonemap_scale = tonemap_scale
        self.num_hold_out_items = num_hold_out_items

        # gather camera trajectories
        self.cam_names = get_hypersim_camera_names(
            volume_id=None,
            scene_id=None,
            scene_dir=self.scene_dir,
        )
        if self.cam_idx is not None and self.cam_idx >= 0:
            self.cam_names = [self.cam_names[self.cam_idx]]

        # gather camera information
        self.width_px = round(float(self.camera_info["settings_output_img_width"]))
        self.height_px = round(float(self.camera_info["settings_output_img_height"]))
        self.meters_per_asset_unit = float(self.camera_info["settings_units_info_meters_scale"])

        # Hypersim's u-axis is to right, v-axis is to up, w-axis is to us.
        # The origin is at the image center.
        self.M_u2c = torch.from_numpy(
            np.array(
                [
                    [float(self.camera_info["M_cam_from_uv_00"]), float(self.camera_info["M_cam_from_uv_01"]),
                     float(self.camera_info["M_cam_from_uv_02"])],
                    [float(self.camera_info["M_cam_from_uv_10"]), float(self.camera_info["M_cam_from_uv_11"]),
                     float(self.camera_info["M_cam_from_uv_12"])],
                    [float(self.camera_info["M_cam_from_uv_20"]), float(self.camera_info["M_cam_from_uv_21"]),
                     float(self.camera_info["M_cam_from_uv_22"])],
                ], dtype=np.float32))
        self.M_c2u = torch.linalg.inv(self.M_u2c)

        # Transformation from hypersim's uv-coordinate to ours pq-coordinate
        # (p to right, q to down, r to far). top-left is (0,0), bottom-right is (w,h)
        #
        # Theoretically, M_u2p[2,2] should be -1, but
        # we deliberately set M_u2p[2,2] to be 1 (instead of -1)
        # this allows us to use r to be +1
        self.M_u2p = torch.from_numpy(
            np.array(
                [
                    [self.width_px / 2., 0, self.width_px / 2.],
                    [0., -self.height_px / 2., self.height_px / 2.],
                    [0., 0, 1.],
                ], dtype=np.float32))  # (3, 3)

        self.M_c2p = self.M_u2p @ self.M_c2u
        self.intrinsic = self.M_c2p  # from camera coodinate to pq-coordinate

        # read camera poses and other info
        self.H_c2ws: T.Dict[str, torch.Tensor] = dict()  # cam_name -> H_c2ws  (n_img, 4, 4)
        self.cam_name_to_frame_idxs = dict()  # cam_name -> frame_idxs (n_imgs,)
        self.cam_name_to_frame_idxs_to_idxs = dict()  # cam_name -> (frame_idx -> idx of H_c2w)
        for cam_name in self.cam_names:
            cam_dir = os.path.join(self.scene_dir, "_detail", cam_name)
            camera_positions_hdf5_file = os.path.join(cam_dir, "camera_keyframe_positions.hdf5")
            camera_orientations_hdf5_file = os.path.join(cam_dir, "camera_keyframe_orientations.hdf5")
            camera_frame_idx_hdf5_file = os.path.join(cam_dir, "camera_keyframe_frame_indices.hdf5")
            with h5py.File(camera_positions_hdf5_file, "r") as f:
                cam_positions = f["dataset"][:]  # (n_img, 3) xyz_w, camera positions (in asset coordinates)
            cam_positions = torch.from_numpy(cam_positions.astype(np.float32))
            cam_positions = cam_positions * self.meters_per_asset_unit  # (n_img, 3) xyz_w, (in meters)
            with h5py.File(camera_orientations_hdf5_file, "r") as f:
                R_c2ws = f["dataset"][:]  # (n_img, 3, 3)
            R_c2ws = torch.from_numpy(R_c2ws.astype(np.float32))  # (n_img, 3, 3)
            with h5py.File(camera_frame_idx_hdf5_file, "r") as f:
                frame_idxs = f["dataset"][:]  # (n_img,)
            self.cam_name_to_frame_idxs[cam_name] = frame_idxs.tolist()

            # some frame_idx does not have the hdf5 file, remove them from frame_idxs
            frame_idxs = []
            for frame_idx in self.cam_name_to_frame_idxs[cam_name]:
                exist = True
                for type in ['depth_meters', 'position', 'normal_bump_world', ]:
                    filename = self._get_geometry_hdf5_filename(
                        type=type,
                        camera_name=cam_name,
                        frame_id=frame_idx,
                    )
                    if not os.path.exists(filename):
                        exist = False
                        break
                for type in ['color']:
                    filename = self._get_imagary_hdf5_filename(
                        type=type,
                        camera_name=cam_name,
                        frame_id=frame_idx,
                    )
                    if not os.path.exists(filename):
                        exist = False
                        break
                if exist:
                    frame_idxs.append(frame_idx)
            self.cam_name_to_frame_idxs[cam_name] = frame_idxs
            R_c2ws = R_c2ws[self.cam_name_to_frame_idxs[cam_name]]
            cam_positions = cam_positions[self.cam_name_to_frame_idxs[cam_name]]

            # book keeping
            frame_idx_to_idx = dict()
            for i, fid in enumerate(self.cam_name_to_frame_idxs[cam_name]):
                frame_idx_to_idx[fid] = i
            self.cam_name_to_frame_idxs_to_idxs[cam_name] = frame_idx_to_idx

            n_img = len(self.cam_name_to_frame_idxs[cam_name])
            H_c2w = torch.zeros(n_img, 4, 4)
            H_c2w[:, :3, :3] = R_c2ws
            H_c2w[:, :3, 3] = cam_positions
            H_c2w[:, 3, 3] = 1
            self.H_c2ws[cam_name] = H_c2w

        # divide all images into samples
        total_images = 0
        for cam_name in self.cam_names:
            total_images += len(self.cam_name_to_frame_idxs[cam_name])
        if self.num_images_per_item is None:
            self.num_images_per_item = total_images - self.num_hold_out_items
        elif self.num_images_per_item < 0:
            self.num_images_per_item = (total_images - self.num_hold_out_items) // max(
                1, int(-self.num_images_per_item))

        # gather all images
        cam_name_frame_id_pairs = []
        for cam_name in self.cam_names:
            for frame_idx in self.cam_name_to_frame_idxs[cam_name]:
                cam_name_frame_id_pairs.append([str(cam_name), int(frame_idx)])

        # permute the items
        np.random.shuffle(cam_name_frame_id_pairs)

        self.idx_to_cam_name_and_frame_idx = []  # sample idx -> list of (cam_name, frame_idx)
        current_idx = 0
        while current_idx < (total_images - self.num_hold_out_items):
            fids = cam_name_frame_id_pairs[current_idx:current_idx + self.num_images_per_item]
            if len(fids) < self.num_images_per_item:
                break
            self.idx_to_cam_name_and_frame_idx.append(fids)
            current_idx += self.num_images_per_item

        # add the hold out items as the last sample
        if self.num_hold_out_items > 0:
            fids = cam_name_frame_id_pairs[-self.num_hold_out_items:]
            self.idx_to_cam_name_and_frame_idx.append(fids)

        # use the first image to determine the tonemapping scale
        if self.tonemap and self.tonemap_scale is None:
            tonemap_info = self.get_tonemap_info()
            self.tonemap_scale = tonemap_info['scale']

    def get_tonemap_info(self) -> T.Dict[str, float]:
        # use the first image to determine the scale
        cam_name_and_frame_idx_pairs = self.idx_to_cam_name_and_frame_idx[0]  # list of (cam_name, frame_idx)
        cam_name, frame_id = cam_name_and_frame_idx_pairs[0]
        filename = self._get_imagary_hdf5_filename(
            type='color',
            camera_name=cam_name,
            frame_id=frame_id,
        )
        with h5py.File(filename, "r") as f:
            rgbs = f["dataset"][:]  # (h, w, 3)
        rgbs = torch.from_numpy(rgbs.astype(np.float32))  # (h, w, 3)
        tonemap_scale = self.determine_tonemap_scale(
            rgb=rgbs,
            gamma=self.tonemap_gamma,
            percentile=self.tonemap_percentile,
            target_val=self.tonemap_percentile_target,
        )
        return dict(
            scale=tonemap_scale,
            gamma=self.tonemap_gamma,
            percentile=self.tonemap_percentile,
            target_val=self.tonemap_percentile_target,
        )

    @staticmethod
    def determine_tonemap_scale(
            rgb: torch.Tensor,
            gamma: float,
            percentile: float,
            target_val: float,
    ) -> float:
        """rgb: (h, w, 3)"""
        # "CCIR601 YIQ" method for computing brightness
        brightness = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]

        # if the kth percentile brightness value in the unmodified image is less than eps,
        # set the scale to 0.0 to avoid divide-by-zero
        eps = 0.0001
        brightness_nth_percentile_current = np.percentile(
            brightness.detach().cpu().numpy(),
            percentile,
        )

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            scale = np.power(target_val, 1. / gamma) / brightness_nth_percentile_current

        return scale

    @staticmethod
    def apply_gamma(rgb: torch.Tensor, gamma: float, scale: float) -> torch.Tensor:
        """apply the scaling and gamma curve.
        rgb: (h, w, 3)
        """
        if scale is None:
            return rgb

        rgb = (scale * rgb).clamp(min=0) ** gamma
        rgb = rgb.clamp(min=0, max=1.)
        return rgb

    def _get_geometry_hdf5_filename(
            self,
            type: str,
            camera_name: str,
            frame_id: int,
    ) -> str:
        """

        Args:
            type:
                "depth_meters":  Euclidean distances (in meters) to the optical center of the camera (ray_t)
                "position": world-space positions (in asset coordinates)
                "normal_cam": surface normals in camera-space (ignores bump mapping)
                "normal_world": surface normals in world-space (ignores bump mapping)
                "normal_bump_cam": surface normals in camera-space (takes bump mapping into account)
                "normal_bump_world": surface normals in world-space (takes bump mapping into account)
                "render_entity_id": fine-grained segmentation where each V-Ray node has a unique ID
                "semantic": NYU40 semantic labels
                "semantic_instance": semantic instance IDs
                "tex_coord": texture coordinates
            camera_name: cam_00
            frame_id: 1

        Returns:
            hdf5 filename
        """
        img_dir = os.path.join(self.scene_dir, "images")
        hdf5_filename = os.path.join(
            img_dir,
            "scene_" + camera_name + "_geometry_hdf5",
            f"frame.{frame_id:04d}.{type}.hdf5",
        )
        return hdf5_filename

    def _get_imagary_hdf5_filename(
            self,
            type: str,
            camera_name: str,
            frame_id: int,
    ) -> str:
        """

        Args:
            type:
                "color":  color image before any tone mapping has been applied
                "diffuse_illumination": diffuse illumination
                "diffuse_reflectance": diffuse reflectance (many authors refer to this modality as "albedo")
                "residual": non-diffuse residual

            camera_name: cam_00
            frame_id: 1

        Returns:
            hdf5 filename
        """
        img_dir = os.path.join(self.scene_dir, "images")
        hdf5_filename = os.path.join(
            img_dir,
            "scene_" + camera_name + "_final_hdf5",
            f"frame.{frame_id:04d}.{type}.hdf5",
        )
        return hdf5_filename

    def _get_frame(
            self,
            camera_name: str,
            frame_id: int,
    ) -> T.Dict[str, torch.Tensor]:
        """
        Load depth map (z in camera coordinate (x to right, y to down, z to far))
        and surface normal in world coordinate (after bump map).

        Args:
            camera_name:
            frame_id:

        Returns:
            z_map:  (h, w)  z coordinate value in the camera coordinate
            surface_normal_w: (h, w, 3)  surface normal in world coordinate
            rgb: (h, w, 3) color before tone mapping
        """

        # calculate z_map
        filename = self._get_geometry_hdf5_filename(
            type='depth_meters',
            camera_name=camera_name,
            frame_id=frame_id,
        )
        with h5py.File(filename, "r") as f:
            ray_ts = f["dataset"][:]  # (h, w)  in meter
        ray_ts_abs = torch.from_numpy(ray_ts.astype(np.float32))  # (h, w)  in meter

        u_min, u_max, v_min, v_max = -1.0, 1.0, -1.0, 1.0
        half_du = 0.5 * (u_max - u_min) / self.width_px
        half_dv = 0.5 * (v_max - v_min) / self.height_px
        u, v = torch.meshgrid(
            torch.linspace(u_min + half_du, u_max - half_du, self.width_px),
            torch.linspace(v_min + half_dv, v_max - half_dv, self.height_px).flip(dims=[0]),
            indexing='xy',
        )  # (h, w)  # (h, w)
        uv1 = torch.stack((u, v, torch.ones_like(u)), dim=-1)  # (h, w, 3)
        uv1 = uv1.reshape(-1, 3)  # (hw, 3)

        # compute our own rays
        H_c2w = self.H_c2ws[camera_name][self.cam_name_to_frame_idxs_to_idxs[camera_name][frame_id]]  # (4, 4)
        tmp_c = (self.M_u2c @ uv1.T).T  # (hw, 3)
        tmp_w = (H_c2w[:3, :3] @ tmp_c.T).T  # (hw, 3)
        ray_direction_w = torch.nn.functional.normalize(tmp_w, p=2, dim=-1)  # (hw, 3)
        xyz_w = H_c2w[:3, 3].reshape(1, 3) + ray_direction_w * ray_ts_abs.reshape(-1, 1)  # (hw, 3)

        H_w2c = rigid_motion.inv_homogeneous_tensors(H_c2w)  # (4, 4)
        xyz1_w = torch.cat((xyz_w, torch.ones_like(xyz_w[:, :1])), dim=-1)  # (hw, 4)
        xyz1_c = (H_w2c @ xyz1_w.T).T  # (hw, 4)
        xyz_c = xyz1_c[:, :3]  # (hw, 3)
        xyz_c = xyz_c.reshape(self.height_px, self.width_px, 3)  # (h, w, 3)
        z_map = xyz_c[..., 2]  # (h, w)  all < 0
        z_map = -1 * z_map  # (h, w) all >= 0

        # surface_normal_w
        filename = self._get_geometry_hdf5_filename(
            type='normal_bump_world',
            camera_name=camera_name,
            frame_id=frame_id,
        )
        with h5py.File(filename, "r") as f:
            surface_normal_w = f["dataset"][:]  # (h, w, 3)
        surface_normal_w = torch.from_numpy(surface_normal_w.astype(np.float32))  # (h, w, 3)

        # make sure surface normal points to the ray origin (opposite direction of ray_direction)
        ray_direction_w = ray_direction_w.reshape(self.height_px, self.width_px, 3)  # (h, w, 3)
        surface_normal_w = surface_normal_w * \
                           (-1 * torch.sign(torch.sum(surface_normal_w * ray_direction_w, dim=-1, keepdims=True)))

        # color
        filename = self._get_imagary_hdf5_filename(
            type='color',
            camera_name=camera_name,
            frame_id=frame_id,
        )
        with h5py.File(filename, "r") as f:
            rgbs = f["dataset"][:]  # (h, w, 3)
        rgbs = torch.from_numpy(rgbs.astype(np.float32))  # (h, w, 3)

        # tonemap rgb
        if self.tonemap:
            rgbs = self.apply_gamma(
                rgb=rgbs,
                gamma=self.tonemap_gamma,
                scale=self.tonemap_scale,
            )

        return dict(
            surface_normal_w=surface_normal_w,  # (h, w, 3)
            z_map=z_map,  # (h, w)
            rgb=rgbs,  # (h, w, 3)
        )

    def _get_xyz_w(
            self,
            camera_name: str,
            frame_id: int,
    ) -> torch.Tensor:
        """Get the xyz_w contained in the dataset in meters."""
        filename = self._get_geometry_hdf5_filename(
            type='position',
            camera_name=camera_name,
            frame_id=frame_id,
        )
        with h5py.File(filename, "r") as f:
            xyz_w_asset = f["dataset"][:]  # (h, w, 3)
        xyz_w_asset = torch.from_numpy(xyz_w_asset.astype(np.float32))
        xyz_w = xyz_w_asset * self.meters_per_asset_unit  # (h, w, 3)
        return xyz_w  # (h, w, 3)

    def __len__(self):
        return len(self.idx_to_cam_name_and_frame_idx)

    def __getitem__(self, i):
        cam_name_and_frame_idx_pairs = self.idx_to_cam_name_and_frame_idx[i]  # list of (cam_name, frame_idx)
        surface_normal_w = []
        z_map = []
        rgb = []
        H_c2ws = []
        for i in range(len(cam_name_and_frame_idx_pairs)):
            cam_name, frame_idx = cam_name_and_frame_idx_pairs[i]
            # print(f'cam_name = {cam_name}, frame_idx = {frame_idx}', flush=True)
            d_dict = self._get_frame(
                camera_name=cam_name,
                frame_id=frame_idx,
            )
            surface_normal_w.append(d_dict['surface_normal_w'])  # (h, w, 3)
            z_map.append(d_dict['z_map'])  # (h, w)
            rgb.append(d_dict['rgb'])  # (h, w, 3)
            H_c2ws.append(self.H_c2ws[cam_name][self.cam_name_to_frame_idxs_to_idxs[cam_name][frame_idx]])  # (4, 4)

        surface_normal_w = torch.stack(surface_normal_w, dim=0)  # (num_images_per_item, h, w, 3)
        z_map = torch.stack(z_map, dim=0)  # (num_images_per_item, h, w)
        rgb = torch.stack(rgb, dim=0)  # (num_images_per_item, h, w, 3)
        H_c2ws = torch.stack(H_c2ws, dim=0)  # (num_images_per_item, 4, 4)

        # handle image subsample
        ori_w = self.width_px
        h = self.height_px // self.image_subsample
        w = self.width_px // self.image_subsample
        intrinsic = self.intrinsic.clone()
        intrinsic[:2, :] = intrinsic[:2, :] * w / ori_w
        # intrinsic = self.intrinsic * w / ori_w
        # intrinsic[..., 2, 2] = ori22
        rgb = rgb[..., ::self.image_subsample, ::self.image_subsample, :]
        z_map = z_map[..., ::self.image_subsample, ::self.image_subsample]
        surface_normal_w = surface_normal_w[..., ::self.image_subsample, ::self.image_subsample, :]

        camera = structures.Camera(
            H_c2w=H_c2ws.unsqueeze(0),  # (1, num_images_per_item, 4, 4)
            intrinsic=intrinsic.view(1, 1, 3, 3).expand(1, rgb.size(0), 3, 3),
            width_px=w,
            height_px=h,
        )

        return dict(
            rgbd_image=structures.RGBDImage(
                rgb=rgb.unsqueeze(0),  # (1, num_images_per_item, h, w, 3)
                depth=z_map.unsqueeze(0),  # (1, num_images_per_item, h, w)
                camera=camera,
                normal_w=surface_normal_w.unsqueeze(0),  # (1, num_images_per_item, h, w, 3)
                hit_map=z_map.unsqueeze(0) < 1e6,  # (1, num_images_per_item, h, w)
            )
        )
