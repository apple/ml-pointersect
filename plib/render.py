#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
import copy

from plib.uv_mapping import UVMap
import open3d as o3d
import numpy as np
import os
import typing as T
from plib import rigid_motion
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager
import math
from plib import sample_utils
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xatlas


def sample_point_cloud_with_ray_tracing(
        mesh: o3d.geometry.TriangleMesh,
        cam_pose: T.Dict[str, T.Any],
        texture_maps: T.List[np.ndarray] = None,
) -> T.Dict[str, T.Any]:
    """
    Sample a set of points (with color) on the mesh.

    Args:
        mesh:
            a triangle mesh representing the surface.
            We assume it only have one texture map
        cam_pose:
            a dictionary containing:
                - intrinsic_matrix: the (3,3) intrinsic matrix
                - extrinsic_matrix: the (4,4) homogeneous matrix (from world to local)
                - width_px: number of pixels of the sensor
                - height_px: number of pixels of the sensor
        texture_maps:
            a list of texture maps (h, w, c) that we want to get the values using uv_mapping
            For example, if it is the rgb albedo, it is be retrieved with
            :py:`texture = np.asarray(mesh.textures[0]) / 255.  # (h,w,3)`.
            It can also be surface normals, features, etc.
            If `None`, not uv interpolation is performed.
    Returns:
        raycast_results:
            the output of `RaycastingScene`
        uv_outputs:
            a list of uv-interpolated values from each texture map.
    """

    # set up the raycasting scene
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_t)

    # create the pinhole camera rays
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=cam_pose['intrinsic_matrix'],
        extrinsic_matrix=cam_pose['extrinsic_matrix'],
        width_px=cam_pose['width_px'],
        height_px=cam_pose['height_px'],
    )

    # cast the rays, get the intersections
    raycast_results = scene.cast_rays(rays)
    # if the ray does not hit the mesh, it goes to inf
    hit_map = 1 - np.isinf(raycast_results['t_hit'].numpy())  # (h',w')

    if texture_maps is None or len(texture_maps) == 0:
        return dict(
            hits=hit_map,  # (h', w')  1: hit, 0: not hit
            rays=rays,  # (h', w', 6)  x, y, z, dx, dy, dz
            raycast_results=raycast_results,
            uv_outputs=[],
        )

    uv_outputs = interp_texture_map_from_ray_tracing_results(
        mesh=mesh,
        raycast_results=raycast_results,
        texture_maps=texture_maps,
    )

    return dict(
        hits=hit_map,  # (h', w')  1: hit, 0: not hit
        rays=rays,  # (h', w', 6)  x, y, z, dx, dy, dz
        raycast_results=raycast_results,
        uv_outputs=uv_outputs,  # list of (h', w', dim)
    )


def interp_texture_map_from_ray_tracing_results(
        mesh: o3d.geometry.TriangleMesh,
        raycast_results: T.Dict[str, T.Any],
        texture_maps: T.List[np.ndarray],
        merge_textures: bool = False
) -> T.List[np.ndarray]:
    """
    Interpolate a uv_map given ray_tracing results and a mesh

    Args:
        mesh:
            o3d mesh
        raycast_results:
            (*, h', w')  the output of ray casting of o3d,
        texture_maps:
            a list of texture maps (h, w, c) that we want to get the values using uv_mapping
            For example, if it is the rgb albedo, it is be retrieved with
            :py:`texture = np.asarray(mesh.textures[0]) / 255.  # (h,w,3)`.
            It can also be surface normals, features, etc.
            If `None`, not uv interpolation is performed.
        merge_textures:
            If True, assume each texture map corresponds to a material_ids in the mesh
            For each texture map, only apply if the material_id is matched with the pixel
            Sum all contribution from all texture maps

    Returns:
        a list of uv-interpolated values from each texture map.  (*, dim_texture)
        The size `*` is determined by the rays's shape to ray tracing.
        If merge_textures is true, return sum of uv-interpolated values from all texture map instead. (dim_texture)
    """

    hit_map = 1 - np.isinf(raycast_results['t_hit'].numpy())  # (*, h',w')

    # barycentric coordinates of the intersection in the intersected triangle
    barycentric_coords = raycast_results['primitive_uvs'].numpy()  # (*, h',w',2), the third weight is 1-sum()
    # (*, h',w',2) -> (*, h',w',3)
    barycentric_coords = np.concatenate(
        (1 - np.sum(barycentric_coords, axis=-1, keepdims=True), barycentric_coords),
        # note that in open 3d, the primitive_uvs indicates last two coordinates rather than first two
        # (barycentric_coords, 1 - np.sum(barycentric_coords, axis=-1, keepdims=True)),
        axis=-1,
    )  # (*, h', w', 3)

    # the intersected triangle index
    primitive_ids = raycast_results['primitive_ids'].numpy()  # (*, h', w'),
    # fillin a dummy primitive_id for the rays that go to inf
    primitive_ids[primitive_ids == o3d.t.geometry.RaycastingScene.INVALID_ID] = 0  # (*, h', w')

    # get the uv coordinates of the vertices of each triangle
    triangle_uvs = np.asarray(mesh.triangle_uvs)  # (num_triangles*3, 2)
    triangle_uvs = np.reshape(triangle_uvs, (-1, 3, 2))  # (num_triangles, 3, 2)

    # compute the uv coordinate on the texture map
    vertex_uvs = triangle_uvs[primitive_ids]  # (*, h', w', 3, 2)
    uvs = np.sum(np.expand_dims(barycentric_coords, axis=-1) * vertex_uvs, axis=-2)  # (*, h', w', 2)

    # compute the material id for each pixel in the image
    if merge_textures:
        triangle_material_ids = np.asarray(mesh.triangle_material_ids)  # (num_triangle,)
        material_ids = triangle_material_ids[primitive_ids]  # (*, h', w')

    uv_outputs = []  # Now return sum of output from all texture maps []
    for id_t, texture in enumerate(texture_maps):
        uv_map = UVMap(texture)  # texture:  (h, w, dim)
        out = uv_map(uvs)  # (*, h', w', dim)

        # set the unintersected rays to zero
        axis = list(range(hit_map.ndim, out.ndim))
        if len(axis) > 0:
            tmp_hit_map = np.expand_dims(hit_map, axis=axis)  # (*, h', w', 1)
        else:
            tmp_hit_map = hit_map  # (*, h', w')
        out = out * tmp_hit_map  # (*, h', w', dim)

        if merge_textures:
            material_map = (material_ids == id_t)
            if len(axis) > 0:
                material_map = np.expand_dims((material_ids == id_t), axis=axis)
            out = out * tmp_hit_map * material_map
        uv_outputs.append(out)

    if merge_textures:
        uv_outputs = [sum(uv_outputs)]  # use list to be compatible with legacy usage

    return uv_outputs


def interp_surface_normal_from_ray_tracing_results(
        mesh: o3d.geometry.TriangleMesh,
        raycast_results: T.Dict[str, T.Any],
) -> np.ndarray:
    """
    Interpolate the surface normal of the intersection points using surface normal on the vertices.

    Args:
        mesh:
        raycast_results:
            the output of ray casting of o3d

    Returns:
        (*, 3) surface normal.  Note that it fills in a random value if a ray does not hit a surface

    """
    if not mesh.has_vertex_normals():
        return raycast_results['primitive_normals'].numpy()  # (*,3)

    # barycentric coordinates of the intersection in the intersected triangle
    barycentric_coords = raycast_results['primitive_uvs'].numpy()  # (h,w,2), the third weight is 1-sum()
    barycentric_coords = np.concatenate(
        (1 - np.sum(barycentric_coords, axis=-1, keepdims=True), barycentric_coords),
        # note that in open 3d, the primitive_uvs indicates last two coordinates rather than first two
        # (barycentric_coords, 1 - np.sum(barycentric_coords, axis=-1, keepdims=True)),
        axis=-1,
    )  # (h,w,3)

    # the intersected triangle index
    primitive_ids = raycast_results['primitive_ids'].numpy()  # (h, w)
    # fillin a dummy primitive_id for the rays that go to inf
    primitive_ids[primitive_ids == o3d.t.geometry.RaycastingScene.INVALID_ID] = 0  # (h, w)

    # get triangle vertex
    triangle_vidxs = np.asarray(mesh.triangles)  # (n_triangle, 3)  index of vertices
    vertex_normals = np.asarray(mesh.vertex_normals)  # (n_vertex, 3)  surface normal of each vertex

    vidxs = triangle_vidxs[primitive_ids]  # (h, w, 3)  the vertex id of each camera ray
    v_normals = vertex_normals[vidxs]  # (h, w, 3, 3)  last dimension is normal (dx, dy, dz)
    interped_normals = np.sum(np.expand_dims(barycentric_coords, axis=-1) * v_normals, axis=-2)  # (h, w, 3)
    return interped_normals


def rasterize(
        meshes: T.Union[o3d.geometry.TriangleMesh, T.List[o3d.geometry.TriangleMesh],
        o3d.geometry.PointCloud, T.List[o3d.geometry.PointCloud]],
        intrinsic_matrix: T.Union[np.ndarray, T.List[np.ndarray]],
        extrinsic_matrices: T.Union[np.ndarray, T.List[np.ndarray]],
        width_px: int,
        height_px: int,
        get_point_cloud: bool = True,
        pcd_subsample: int = 1,
        point_size: float = -1,
        show_backface=True,
        dtype: np.dtype = np.float32,
) -> T.Dict[str, T.Any]:
    """
    Use open3d's visualizer to render image and depth_map from the camera.

    Args:
        meshes: a list of meshes
        intrinsic_matrix:
            (3,3) intrinsic matrix shared among all cameras.
            or a list of (3, 3) intrinsic matrices of each cameras.
            We only support fx = fy

            For example:
                intrinsic_matrix=np.array([
                    [128, 0, 128],
                    [0, 128, 128],
                    [0,   0,   1],
                ], dtype=np.float),

        extrinsic_matrices:
            a list of (4,4) homogeneous matrix (from world coordinate to camera coordinate)

            For example:
                extrinsic_matrix=np.array([
                    [1, 0, 0, 0,],
                    [0, 1, 0, 0,],
                    [0, 0, 1, 0,],
                    [0, 0, 0, 1.,],
                ], dtype=np.float),,  # world to camera (cV = H * wV)

        width_px:
            number of pixels of the sensor.  ex: 256
        height_px:
            number of pixels of the sensor.  ex: 256
        get_point_cloud:
            whether to construct a point cloud (in the world coordinate) from
            the rendered images
        pcd_subsample:
            subsample the point cloud (1 point in every n pixel).  >= 1
        point_size:
            new option when input is a point cloud, change the render size of points
    Returns:
        imgs: a list of (h, w, 3)  rgb
        z_maps:  a list of (h, w)  z of the scene points in the camera coordinate
        pcds: a list of o3d.geometry.PointCloud in the world coordinate (one for each camera pose)
        hit_maps: a list of (h, w)  true: valid, false: not valid
    """

    np_dtype = sample_utils.get_np_dtype(dtype)

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    # if not isinstance(extrinsic_matrices, (list, tuple)) or (
    if isinstance(extrinsic_matrices, np.ndarray) and extrinsic_matrices.ndim == 2:
        extrinsic_matrices = [extrinsic_matrices]

    if isinstance(intrinsic_matrix, np.ndarray) and intrinsic_matrix.ndim == 2:
        intrinsic_matrix = [intrinsic_matrix] * len(extrinsic_matrices)
    assert len(intrinsic_matrix) == len(extrinsic_matrices)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width_px, height=height_px, visible=False)
    # show back face to make sure ray-casting and rendering results are the same
    vis.get_render_option().mesh_show_back_face = show_backface
    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    for mesh in meshes:
        vis.add_geometry(mesh)

    all_points = []
    all_colors = []
    imgs = []
    z_maps = []
    hit_maps = []

    for i in range(len(extrinsic_matrices)):
        # assert np.isclose(intrinsic_matrix[i][0, 0], intrinsic_matrix[i][1, 1])
        assert np.isclose(np.abs(intrinsic_matrix[i][0, 0]), np.abs(intrinsic_matrix[i][1, 1]))
        view_ctl = vis.get_view_control()
        cam_pose_ctl = view_ctl.convert_to_pinhole_camera_parameters()
        cam_pose_ctl.intrinsic.height = height_px
        cam_pose_ctl.intrinsic.width = width_px
        cam_pose_ctl.intrinsic.intrinsic_matrix = intrinsic_matrix[i]
        cam_pose_ctl.extrinsic = extrinsic_matrices[i]
        view_ctl.convert_from_pinhole_camera_parameters(cam_pose_ctl, allow_arbitrary=True)

        if point_size > 0:
            vis.get_render_option().point_size = point_size

        # render
        vis.poll_events()
        vis.update_renderer()
        z_map = vis.capture_depth_float_buffer(do_render=False)
        z_map = np.asarray(z_map).astype(dtype=np_dtype)
        hit_map = np.logical_not(z_map == 0)
        z_map[z_map == 0] = 1e12  # avoid points appear at camera center # not set to inf: avoid numerical problem
        img = vis.capture_screen_float_buffer(do_render=False)
        img = np.asarray(img).astype(dtype=np_dtype)
        imgs.append(img)
        z_maps.append(z_map)
        hit_maps.append(hit_map)

        # convert point cloud to world coordinate
        if get_point_cloud:
            H_cam_to_world = rigid_motion.RigidMotion.invert_homogeneous_matrix(cam_pose_ctl.extrinsic)
            points, colors = generate_point(
                rgb_image=img,
                depth_image=z_map,
                intrinsic=cam_pose_ctl.intrinsic.intrinsic_matrix,
                subsample=pcd_subsample,
                world_coordinate=True,
                pose=H_cam_to_world,
            )
            all_points.append(points)
            all_colors.append(colors)

    # clear the visualizer
    vis.clear_geometries()
    vis.destroy_window()
    del cam_pose_ctl
    del view_ctl
    del vis

    # create point cloud
    pcds = []
    for i in range(len(all_points)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points[i])
        pcd.colors = o3d.utility.Vector3dVector(all_colors[i])
        pcds.append(pcd)

    return dict(
        pcds=pcds,
        imgs=imgs,
        z_maps=z_maps,
        hit_maps=hit_maps,
    )


def generate_point(
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsic: np.ndarray,
        subsample: int = 1,
        world_coordinate: bool = True,
        pose: np.ndarray = None,
):
    """
    Generate 3D point coordinates and related rgb feature

    Args:
        rgb_image: (h, w, 3) rgb
        depth_image: (h, w) depth, along z direction (not along individual camera ray)
        intrinsic: (3, 3)
        subsample: int
            resize stride
        world_coordinate: bool
        pose: (4, 4) matrix
            transfer from camera to world coordindate

    Returns:
        points: (N, 3) point cloud coordinates
            in world-coordinates if world_coordinate==True
            else in camera coordinates
        rgb_feat: (N, 3) rgb feature of each point

    Important note:
        The function uses the image coordinate system: x to right, y to "down", z to far.
        If the world coordinate is a different one (say x to right, y to "up", z to us),
        H_c2w need to include the coordinate conversion.
    """
    intrinsic_4x4 = np.identity(4)
    intrinsic_4x4[:3, :3] = intrinsic

    u, v = np.meshgrid(
        range(0, depth_image.shape[1], subsample),
        range(0, depth_image.shape[0], subsample),
    )
    # u: (depth_image.shape[0]//subsample, depth_image.shape[1]//subsample), x
    # v: (depth_image.shape[0]//subsample, depth_image.shape[1]//subsample), y
    d = depth_image[v, u]
    d_filter = d != 0
    mat = np.vstack(
        (
            u[d_filter] * d[d_filter],
            v[d_filter] * d[d_filter],
            d[d_filter],
            np.ones_like(u[d_filter]),
        )
    )
    new_points_3d = np.dot(np.linalg.inv(intrinsic_4x4), mat)[:3]
    if world_coordinate:
        new_points_3d_padding = np.vstack(
            (new_points_3d, np.ones((1, new_points_3d.shape[1])))
        )
        world_coord_padding = np.dot(pose, new_points_3d_padding)
        new_points_3d = world_coord_padding[:3]

    rgb_feat = rgb_image[v, u][d_filter]

    return new_points_3d.T, rgb_feat


def derive_camera_intrinsics(
        width_px: int,
        height_px: int,
        fov: float,
        dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Derive camera intrinsic matrix.

    Args:
        width_px: width (pixel)
        height_px: height (pixel)
        fov: field-of-view (degree)

    Returns:
        3x3 intrinsic matrix
    """
    camera_f = 0.5 * float(width_px) / np.tan(0.5 * fov / 180. * np.pi)
    camera_intrinsics = np.array(
        [
            [camera_f, 0., width_px * 0.5],
            [0., camera_f, height_px * 0.5],
            [0., 0., 1.],
        ], dtype=sample_utils.get_np_dtype(dtype))

    return camera_intrinsics


def create_gif(
        images: T.Union[torch.Tensor, np.ndarray, T.List[torch.Tensor], T.List[np.ndarray]],
        filename: str,
        fps: float,
        loop: bool = True,
):
    """
    Create a gif from the images

    Args:
        images:
            (n, h, w, 3) or list of (h, w, 3), float, range = 0-1
        filename:
            filename of the output gif
    """

    assert filename.lower().endswith('gif'), f'{filename}'
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    if isinstance(images, (list, tuple)):
        images = [
            img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img
            for img in images]

    if isinstance(images, np.ndarray):
        images = [images[i] for i in range(images.shape[0])]

    # make sure the range is 0-255
    images = [(np.clip(img, a_min=0, a_max=1) * 255).astype(np.uint8) for img in images]

    # numpy to pil image
    images = [Image.fromarray(img) for img in images]

    # avoid dithering
    try:
        images = [img.quantize(method=Image.Quantize.MEDIANCUT) for img in images]
    except:
        images = [img.quantize(method=Image.MEDIANCUT) for img in images]

    # save gif
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=int((1000 + fps - 1) / fps),
        loop=loop,
    )

    # save png for first image as a reference
    images[0].save(
        filename[:-4] + '_oneimg.png'
    )


def gif_to_nparray(
        filename: str,
        crop_ratio: float = 0,
        crop_dir: str = 'left',
) -> np.ndarray:
    """
    Load gif to numpy array.

    Args:
        filename:  filename of the gif
        crop_ratio: ratio of part to be cropped
        crop_dir: direction to be cropped, default left
    Returns:
        ndarray (n,h,w), n: number of frames
    """
    imglist = []
    if not os.path.exists(filename):
        return None

    imageObject = Image.open(filename)
    for frame in range(0, imageObject.n_frames):
        imageObject.seek(frame)
        tmp = imageObject.convert()  # Make without palette
        tmp = np.asarray(tmp) / 255.
        # print(tmp.shape)
        if crop_ratio != 0:
            if crop_dir == 'left':
                tmp = tmp[:, int(tmp.shape[1] * crop_ratio):]
            elif crop_dir == 'right':
                tmp = tmp[:, :int(tmp.shape[1] * crop_ratio)]
            else:
                raise NotImplementedError

        imglist.append(np.expand_dims(tmp, axis=0))
    gifarray = np.concatenate(imglist, axis=0)  # (n,h,w)
    return gifarray


def add_title_to_image(
        image: np.ndarray,
        title: str,
        font_size: int = 24,
        font_color: T.Union[float, T.List[float], None] = None,  # [0, 1]
        font_name: str = "DejaVuSans",
        background_color: T.Union[float, T.List[float]] = 0.,  # [0, 1]
        pad_height_px: int = 30,
        align_width: str = 'center',
        align_height: str = 'center',
) -> np.ndarray:
    """
    Given an image, pad the image at the top and add a title text.
    Note that the function converts image to uint8

    Args:
        image:
            (*, h, w, 3) uint8 [0, 255]
        title:
            the str to add to the image
        font_size:
            font size to add
        font_color:
            font color of the text. If None, it will use the complement color of background_color.
        font_name:
            name of the font.
        background_color:
            background color
        pad_height_px:
            number of pixels to pad at the top of the image
        align_width:
            'center', 'left', 'right'
        align_height:
            'center', 'top', 'bottom'

    Returns:
        the padded image:
            (*, h+pad_height_px, w, 3)  uint8 [0, 255]
    """

    if isinstance(background_color, (int, float)):
        background_color = [background_color] * 3
    background_color = [int(c * 255) for c in background_color]

    if font_color is not None and isinstance(font_color, (int, float)):
        font_color = [int(font_color * 255)] * 3
    if font_color is None:
        font_color = [max(0, 255 - int(c)) for c in background_color]

    *b_shape, h, w, _c = image.shape
    pad_region = np.ones((*b_shape, pad_height_px, w, 3), dtype=image.dtype)
    for c in range(3):
        pad_region[..., c] = background_color[c]
    image = np.concatenate([pad_region, image], axis=-3)  # (*b, h', w, 3)

    if title is None or len(title) == 0:
        return image

    assert image.dtype == np.dtype(np.uint8)

    # get font, image, draw objs
    font_path = matplotlib.font_manager.findfont(font_name)
    font = ImageFont.truetype(font_path, font_size)
    pad_region = np.ones((pad_height_px, w, 3), dtype=image.dtype)  # (hp, w, 3)
    for c in range(3):
        pad_region[..., c] = background_color[c]
    img = Image.fromarray(pad_region)
    draw = ImageDraw.Draw(img)

    # calculate the text width of the resulted text
    text_w_px = draw.textlength(title, font)  # width of the text in pixel (float)

    # compute the starting location of the text image
    if align_width == 'center':
        w_start = max(0, int((w - text_w_px) / 2.))
    elif align_width == 'left':
        w_start = 0
    elif align_width == 'right':
        w_start = max(0, int(w - text_w_px))
    else:
        raise NotImplementedError

    if align_height == 'center':
        h_start = max(0, int((pad_height_px - font_size) / 2.)) if pad_height_px > 0 else 0
    elif align_height == 'top':
        h_start = 0
    elif align_height == 'bottom':
        h_start = int(font_size)
    else:
        raise NotImplementedError

    draw.text((w_start, h_start), title, fill=tuple(font_color), font=font)
    pad_region = np.asarray(img)  # (hp, w, 3)
    image[..., :pad_height_px, :, :] = pad_region  # (*b, h', w, 3)

    assert image.dtype == np.dtype(np.uint8)
    return image


def tile_images(
        images: T.Union[T.List[torch.Tensor], T.List[np.ndarray]],
        ncols: int = -1,
        background_color: T.Union[float, T.List[float]] = 0.,  # [0, 1]
) -> T.Union[torch.Tensor, np.ndarray]:
    """
    Tile a list of images as a matrix (along h and w dimensions).

    Args:
        images:
            list of (*, h, w, c)
        ncols:
            number of images in a column. If -1, unlimited.
        background_color:
            If None, the image will be replaced by (h,w,c) * background

    Returns:
        tiled image:
            (*, h', w', c)
    """
    total = len(images)
    if ncols < 0:
        ncols = total

    # find the first non-None image
    img = None
    for i in range(total):
        if images[i] is not None:
            img = images[i]
            break
    if img is None:
        raise RuntimeError

    if isinstance(img, np.ndarray):
        is_numpy = True
        images = [
            torch.from_numpy(img)
            if img is not None else None
            for img in images
        ]
    else:
        is_numpy = False

    *b_shape, h, w, c = img.shape
    if isinstance(background_color, (int, float)):
        background_color = [background_color] * c

    blank = torch.ones(*b_shape, h, w, c)
    for ic in range(c):
        blank[..., ic] = background_color[ic]

    nrows = math.ceil(total / ncols)
    if nrows == 1:
        ncols = total
    rows = []
    for _ in range(nrows):
        rows.append([blank] * ncols)

    ridx = 0
    cidx = 0
    for i in range(total):
        img = images[i]
        if img is None:
            img = blank
        rows[ridx][cidx] = img
        cidx += 1
        if cidx == ncols:
            cidx = 0
            ridx += 1

    # tile each column, then row
    rows = [torch.cat(row, dim=-2) for row in rows]  # list of (*b, h, w', c)
    img = torch.cat(rows, dim=-3)  # list of (*b, h', w', c)

    if is_numpy:
        img = img.detach().cpu().numpy()
    return img


def create_o3d_plane_mesh(
        top_left: T.List[float],  # (3,)
        top_right: T.List[float],  # (3,)
        bottom_left: T.List[float],  # (3,)
        bottom_right: T.Optional[T.List[float]] = None,  # (3,)
) -> o3d.geometry.TriangleMesh:
    """
    Create an o3d mesh representing a plane.
    The mesh contains two triangles.

    Args:
        top_left:
            the top left coordinate of the plane
        top_right:
            the top right coordinate of the plane
        bottom_left:
            the bottom left coordinate of the plane
        bottom_right:
            the bottom right coordinate of the plane.
            If None, will assumed to be a parallelgram

    Returns:
        o3d mesh
    """
    if isinstance(top_left, list):
        top_left = np.array(top_left, dtype=np.float64)
    if isinstance(top_right, list):
        top_right = np.array(top_right, dtype=np.float64)
    if isinstance(bottom_left, list):
        bottom_left = np.array(bottom_left, dtype=np.float64)
    if isinstance(bottom_right, list):
        bottom_right = np.array(bottom_right, dtype=np.float64)

    if bottom_right is None:
        bottom_right = top_right + (bottom_left - top_left)

    mesh = o3d.geometry.TriangleMesh()
    np_vertices = np.stack(
        [
            top_left,
            top_right,
            bottom_left,
            bottom_right,
        ], axis=0)  # (4, 3)
    np_triangles = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],
        ]).astype(np.int32)
    mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np_triangles)

    return mesh


def create_video(
        images: T.Union[torch.Tensor, np.ndarray, T.List[torch.Tensor], T.List[np.ndarray]],
        filename: str,
        fps: float,
        color_format: str = 'rgb',
        val_range: str = '01',
):
    """
    Create a video from the images

    Args:
        images:
            (n, h, w, 3) or list of (h, w, 3), float, range = 0-1
        filename:
            filename of the output video
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    # read images
    if len(images) == 0:
        return

    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(len(images)):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        if img.dtype != np.uint8:
            if val_range == '01':
                img = np.clip(img, a_min=0, a_max=0.9999) * 255
            elif val_range == '0255':
                img = np.clip(img, a_min=0, a_max=255)
            else:
                img = img / np.max(img) * 255
            # img = np.clip(img, a_min=0, a_max=0.9999) * 255
            img = img.astype(np.uint8)

        if color_format == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def remesh_file(
        filename: str,
        out_filename: str = None,
) -> T.Dict[str, T.Any]:
    """
    Uvmap a mesh file using xatlas.  All existing textures are discarded.

    Args:
        filename:
            input mesh file
        out_filename:
            an obj file with uv mapping (without mtl)

    Returns:

    """

    mesh = o3d.io.read_triangle_mesh(filename, enable_post_processing=True)
    vertices = np.asarray(mesh.vertices)
    triangle_ids = np.asarray(mesh.triangles)

    out_dict = remesh(vertices, triangle_ids)

    if out_filename is not None:
        xatlas.export(
            out_filename,
            vertices[out_dict['vmapping']],
            out_dict['indices'],
            out_dict['uvs'],
        )
    return out_dict


def remesh(
        vertices: np.ndarray,
        triangle_ids: np.ndarray,
) -> T.Dict[str, T.Any]:
    """
    Uvmap a mesh (provided as vertices and triangles) using xatlas.

    Args:
        vertices:
            (n, 3)  float
        triangle_ids:
            (m, 3) int

    Returns:
        vmapping:
            (n,),  uint32, contains the original vertex index for each new vertex.
        indices:
            (num_triangles, 3), uint32, contains the vertex indices of the new triangles.
        uvs:
            (n, 2), contains texture coordinates of the new vertices.
    """

    vmapping, indices, uvs = xatlas.parametrize(
        vertices,
        triangle_ids,
    )
    # `vmapping` contains the original vertex index for each new vertex, (n,), type uint32.
    # `indices` contains the vertex indices of the new triangles, (num_triangles, 3), type uint32.
    # `uvs` contains texture coordinates of the new vertices, (n, 2), type float32.

    return dict(
        vmapping=vmapping,
        indices=indices,
        uvs=uvs,
    )


def srgb_to_linear(x: torch.Tensor):
    """
    Converts an image from sRGB to Linear colorspace.
    We treat each pixel independently (to color channels as well).

    Args:
        x:
            (*,) image in sRGB space.

    Returns:
        (*,) converted image
    """
    limit = 0.04045
    return torch.where(x > limit, torch.pow((x + 0.055) / 1.055, 2.4), x / 12.92)