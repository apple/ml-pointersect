#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

# The file implements functions to clean and preprocess meshes loaded by open3d.

import open3d as o3d
import numpy as np
import typing as T
import copy


def clean_mesh_uv(triangle_uvs: o3d.utility.Vector2dVector) -> o3d.utility.Vector2dVector:
    """
    Ensure mesh uv is wrapped between 0 and 1, and triangles with identical uv in vertices are properly handled.

    Args:
        triangle_uvs: input uv vectors

    Returns:
        cleaned uv vectors
    """

    uvs = np.asarray(triangle_uvs)
    uvs_mod = np.reshape(uvs, [int(uvs.shape[0] / 3), 3, uvs.shape[1]])
    single_uvs = np.logical_and(uvs_mod[:, 0, :] == uvs_mod[:, 1, :], uvs_mod[:, 0, :] == uvs_mod[:, 2, :])
    single_uvs = np.logical_and(single_uvs[:, 0], single_uvs[:, 1])
    # for identical uvs, change uvs to center of texture maps
    uvs_mod[single_uvs, 0, :] = np.array([0.5, 0.5])
    uvs_mod[single_uvs, 1, :] = np.array([0.5, 0.51])
    uvs_mod[single_uvs, 2, :] = np.array([0.51, 0.5])
    uvs_mod = np.reshape(uvs_mod, uvs.shape)

    # wrap uv, note that this would remove repetitive textures
    uvs_mod = uvs_mod - np.floor(uvs_mod)
    return o3d.utility.Vector2dVector(uvs_mod)


def clean_texture(img: T.Union[o3d.geometry.Image, np.ndarray]) -> T.Union[o3d.geometry.Image, np.ndarray]:
    """
    Make sure the texture is a rgb image (not gray one and no alpha channel).

    Args:
        img: input texture image

    Returns:
        img: ensure the output texture image has size (:,:,3)
    """
    img_type = type(img)
    # convert to np array
    img = np.asarray(img)
    assert len(img.shape) == 2 or len(img.shape) == 3, "wrong image size"

    if len(img.shape) == 2:  # gray image
        img = np.tile(np.expand_dims(img, axis=2), (1, 1, 3))
    elif img.shape[2] == 2:  # gray image with alpha
        img = np.tile(np.expand_dims(img[:, :, 0], axis=2), (1, 1, 3))
    elif img.shape[2] == 4:  # rgb image with alpha
        img = img[:, :, :3]

    # need to copy to a new image, or it would cause problem when convert
    # to o3d.cpu.pybind.geometry.Image
    img1 = copy.deepcopy(img)

    # convert back
    if img_type == o3d.geometry.Image:
        img1 = o3d.geometry.Image(img1)
    return img1


def preprocess_mesh(
        mesh: o3d.geometry.TriangleMesh,
        scale: T.Optional[float] = 1.,
        center_w: T.Optional[T.List[float]] = (0., 0., 0.),
        clean: bool = True,
) -> o3d.geometry.TriangleMesh:
    """
    Clean the mesh uv and textures, normalize the mesh within [-scale, scale].

    Args:
        mesh: input mesh
        scale: parameter to scaling the mesh
        center_w: new center in the world coordinate
        clean: whether to clean the mesh

    Returns:
        preprocessed mesh
    """

    # avoid affecting input
    mesh = copy.deepcopy(mesh)

    # center the mesh to (0,0,0)
    if center_w is not None:
        center_w = np.array(center_w)
        cs = mesh.get_axis_aligned_bounding_box().get_center()
        mesh.translate(center_w - cs, relative=True)

    # scale the mesh equally along xyz so that it lies within [-scale, scale]
    if scale is not None:
        s = np.max(mesh.get_axis_aligned_bounding_box().get_half_extent())
        mesh.scale(scale=scale / s, center=np.zeros((3, 1)))

    if clean:
        # wrap mesh uv to [0,1], handle triangles with all vertex have same uv
        mesh.triangle_uvs = clean_mesh_uv(mesh.triangle_uvs)

        # clean non rgb textures, such as ones with alpha or gray images
        mesh.textures = [clean_texture(img) for img in mesh.textures]

    return mesh
