#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
import os
import typing as T
import warnings
from glob import glob

import open3d as o3d
import torch
import numpy as np

import pointersect.meta_script.meta_script_utils as meta_utils
from plib import mesh_utils
from plib import preprocess_obj
from plib import utils
from pointersect.data import mesh_dataset_v2
from pointersect.inference import structures


def get_dataset_mesh_filename() -> str:
    repo_root = os.path.normpath(
        os.path.join(
            os.path.abspath(__file__),
            '../../..',
        ))
    filename = os.path.join(
        repo_root,
        'pointersect/meta_script/configs/mesh_filenames.yaml',
    )
    return filename


def get_dataset_mesh_filename_config() -> T.Dict[str, T.Any]:
    filename = get_dataset_mesh_filename()
    return meta_utils.read_config(filename)


def clean_meshes(
        mesh_filenames: T.List[str],
        dataset_root_dir: str,
        cleaned_root_dir: str,
        skip_exist: bool,
        printout: bool = True,
) -> T.List[str]:
    """
    Clean the meshes in mesh_filenames.

    Note that the actual cleaning will only be executed by rank=0.
    The rest of the ranks should wait for rank=0 to finish.

    Args:
        mesh_filenames:
            a list containing the meshes to be cleaned.
            The mesh_filenames are relative filepath within `dataset_root_dir` or full path.
        dataset_root_dir:
            the root dir where all meshes locate
        cleaned_root_dir:
            only for cleaning obj

    Returns:
        cleaned_mesh_filenames:
            a list containing the cleaned meshes
    """

    # print(f'clean_meshes: dataset_root_dir = {dataset_root_dir}', flush=True)
    # print(f'clean_meshes: cleaned_root_dir = {cleaned_root_dir}', flush=True)

    if isinstance(mesh_filenames, str):  # str, list, or tuple
        mesh_filenames = [mesh_filenames]

    cleaned_mesh_filenames = []
    for mesh_filename in mesh_filenames:
        name, ext = os.path.splitext(mesh_filename)
        if ext.lower() == '.obj':
            rel_mesh_filename = os.path.relpath(mesh_filename, dataset_root_dir)
            cleaned_filename = preprocess_obj.generated_cleaned_mesh(
                input_dir=dataset_root_dir,
                output_dir=cleaned_root_dir,
                relative_filepath=rel_mesh_filename,
                skip_exist=skip_exist,
                printout=printout,
            )
        elif ext.lower() in {'.glb', '.gltf'}:
            cleaned_filename = utils.clean_up_glb_write_gltf(
                filename_glb=mesh_filename,
                overwrite_gltf=not skip_exist,
                exists_ok=True,
                image_ext='.png',  # assume the texture maps are of png format
            )
        elif ext.lower() in {'off'}:
            # off meshes do not need cleaning
            cleaned_filename = mesh_filename
        else:
            warnings.warn(f'{mesh_filename} not cleaned')
            cleaned_filename = mesh_filename

        if os.path.exists(cleaned_filename):
            cleaned_mesh_filenames.append(cleaned_filename)
        else:
            warnings.warn(f'{cleaned_filename} not exists')

    return cleaned_mesh_filenames


def get_clean_mesh_filenames(
        mesh_filenames: T.List[str],
        dataset_root_dir: str,
        cleaned_root_dir: str,
) -> T.List[str]:
    """
    Get the clean the meshes' filenames in mesh_filenames.

    Args:
        mesh_filenames:
            a list containing the meshes to be cleaned.
            The mesh_filenames are relative filepath within `dataset_root_dir`.
        cleaned_root_dir:
            only for cleaning obj

    Returns:
        cleaned_mesh_filenames:
            a list containing the cleaned meshes
    """

    if isinstance(mesh_filenames, str):  # str, list, or tuple
        mesh_filenames = [mesh_filenames]

    cleaned_mesh_filenames = []
    for mesh_filename in mesh_filenames:
        name, ext = os.path.splitext(mesh_filename)
        if ext.lower() == '.obj':
            rel_mesh_filename = os.path.relpath(mesh_filename, dataset_root_dir)
            cleaned_filename = os.path.join(
                cleaned_root_dir,
                rel_mesh_filename,
            )
        elif ext.lower() in {'.glb', '.gltf'}:
            name, ext = os.path.splitext(mesh_filename)
            cleaned_filename = f'{name}.gltf'
        else:
            cleaned_filename = mesh_filename

        if os.path.exists(cleaned_filename):
            cleaned_mesh_filenames.append(cleaned_filename)

    return cleaned_mesh_filenames


def gather_and_clean_dataset(
        dataset_name: str,
        dataset_root_dir: str,
        cleaned_root_dir: str,
        clean_mesh: bool,
        train_mesh_filenames: T.List[str],
        test_mesh_filenames: T.List[str] = None,
        rank: int = 0,
        world_size: int = 1,
        printout: bool = True,
) -> T.Dict[str, T.List[str]]:
    """
    This function is called first in the training process.

    Returns:
        train_mesh_filenames:
            list of mesh filenames for training (full path)
        test_mesh_filenames:
            list of mesh filenames for testing (full path) or None
    """

    if printout:
        print(f'dataset_name = {dataset_name}')
        print(f'dataset_root_dir = {dataset_root_dir}')
        print(f'cleaned_root_dir = {cleaned_root_dir}')
        print(f'clean_mesh = {clean_mesh}')
        print(f'train_mesh_filenames:')
        print(train_mesh_filenames)
        print(f'test_mesh_filenames:')
        print(test_mesh_filenames)
        print(f'rank = {rank}', flush=True)

    if train_mesh_filenames is None:
        train_mesh_filenames = []
    if test_mesh_filenames is None:
        test_mesh_filenames = []

    # download ShapeNet dataset
    if rank == 0:

        if dataset_name.lower() in {
                'shapenet', 'shapenet-debug', 'sketchfab',
                'sketchfab-small', 'sketchfab-small-debug',
                'tex-models',
        }:
            assert os.path.exists(dataset_root_dir)

            # compile the mesh_filenames
            train_mesh_filenames = [
                os.path.join(
                    dataset_root_dir,
                    mesh_filename,
                )
                for mesh_filename in train_mesh_filenames
            ]

            if test_mesh_filenames is not None:
                test_mesh_filenames = [
                    os.path.join(
                        dataset_root_dir,
                        mesh_filename,
                    )
                    for mesh_filename in test_mesh_filenames
                ]
            else:
                test_mesh_filenames = None

        elif dataset_name.lower() in {
                'objaverse',
        }:
            raise NotImplementedError
        else:
            raise NotImplementedError

        # clean meshes
        if clean_mesh:
            if printout:
                print(f'clean training meshes', flush=True)
            train_mesh_filenames = clean_meshes(
                mesh_filenames=train_mesh_filenames,
                dataset_root_dir=dataset_root_dir,
                cleaned_root_dir=cleaned_root_dir,
                skip_exist=True,
                printout=printout,
            )

            if printout:
                print(f'cleaned train mesh filenames:', flush=True)
                print(train_mesh_filenames)

            if test_mesh_filenames is not None:
                if printout:
                    print(f'clean test meshes')

                test_mesh_filenames = clean_meshes(
                    mesh_filenames=test_mesh_filenames,
                    dataset_root_dir=dataset_root_dir,
                    cleaned_root_dir=cleaned_root_dir,
                    skip_exist=True,
                    printout=printout,
                )

                if printout:
                    print(f'cleaned test mesh filenames:')
                    print(test_mesh_filenames)

    if world_size > 1:
        torch.distributed.barrier()

    if rank != 0:
        # compile the mesh_filenames
        train_mesh_filenames = [
            os.path.join(
                dataset_root_dir,
                mesh_filename,
            )
            for mesh_filename in train_mesh_filenames
        ]

        if test_mesh_filenames is not None:
            test_mesh_filenames = [
                os.path.join(
                    dataset_root_dir,
                    mesh_filename,
                )
                for mesh_filename in test_mesh_filenames
            ]
        else:
            test_mesh_filenames = None

        # clean meshes
        if clean_mesh:
            train_mesh_filenames = get_clean_mesh_filenames(
                mesh_filenames=train_mesh_filenames,
                dataset_root_dir=dataset_root_dir,
                cleaned_root_dir=cleaned_root_dir,
            )

            if test_mesh_filenames is not None:
                test_mesh_filenames = get_clean_mesh_filenames(
                    mesh_filenames=test_mesh_filenames,
                    dataset_root_dir=dataset_root_dir,
                    cleaned_root_dir=cleaned_root_dir,
                )

    if printout:
        print(f'final train mesh filenames:')
        print(train_mesh_filenames)

        print(f'final test mesh filenames:')
        print(test_mesh_filenames)

    return dict(
        train_mesh_filenames=train_mesh_filenames,
        test_mesh_filenames=test_mesh_filenames,
    )


def load_mesh_filename(
        mesh_filenames: T.Union[str, T.List[str]],
        mesh_scale: float = 1.,
        printout: bool = True,
) -> T.List[structures.Mesh]:
    """
    Load mesh files as o3d meshes.

    Args:
        mesh_filenames:
             list of filenames
    Returns:
        list of o3d meshes
    """

    if isinstance(mesh_filenames, str):  # str, list, or tuple
        mesh_filenames = [mesh_filenames]

    meshes = []
    for i in range(len(mesh_filenames)):
        mesh_filename = mesh_filenames[i]

        # In default, clean_mesh is true, note that even bunny should be cleaned to multiply Kd and texture map
        if printout:
            print(f'Loading {i}/{len(mesh_filenames)} mesh: {mesh_filename}...', flush=True)

        mesh = o3d.io.read_triangle_mesh(mesh_filename, enable_post_processing=True)
        # preprocess the mesh after reading mesh, before building mesh dataset
        mesh = mesh_utils.preprocess_mesh(mesh, scale=mesh_scale)

        # convert o3d mesh to structures.mesh
        mesh = structures.Mesh(
            mesh=mesh,
            scale=mesh_scale,
        )
        meshes.append(mesh)

    return meshes


def load_and_mix_mesh_filename(
        mesh_filenames: T.Union[str, T.List[str]],
        mesh_scale: float = 1.,
        min_num_mesh: int = 1,
        max_num_mesh: int = 2,
        radius_scale: float = 2,
        total_combined: int = None,
        printout: bool = True,
) -> T.List[structures.Mesh]:
    """
    Load mesh files as o3d meshes, randomly combine meshes to form
    more complex meshes.

    Args:
        mesh_filenames:
             list of filenames
    Returns:
        list of o3d meshes
    """

    if isinstance(mesh_filenames, str):  # str, list, or tuple
        mesh_filenames = [mesh_filenames]

    if total_combined is None:
        total_combined = len(mesh_filenames)

    # randomly mix and match meshes
    mixed_mesh_filenames: T.List[T.List[str]] = []
    for i in range(total_combined):
        num = np.random.randint(low=min_num_mesh, high=max_num_mesh+1)
        fns = np.random.choice(mesh_filenames, size=(num,))
        mixed_mesh_filenames.append(fns)

    all_meshes = []
    for i in range(len(mixed_mesh_filenames)):
        sub_mesh_filenames = mixed_mesh_filenames[i]

        if printout:
            print(f'Loading {i}/{len(mixed_mesh_filenames)} meshes: {sub_mesh_filenames}...', flush=True)

        meshes = []
        for j in range(len(sub_mesh_filenames)):
            mesh_filename = sub_mesh_filenames[j]
            mesh = o3d.io.read_triangle_mesh(mesh_filename, enable_post_processing=True)
            # preprocess the mesh after reading mesh, before building mesh dataset
            mesh = mesh_utils.preprocess_mesh(
                mesh,
                scale=mesh_scale,
            )  # mesh_scale, centered
            meshes.append(mesh)

        # rotate and move meshes to different centers
        for j in range(len(meshes)):
            euler_angles = np.random.rand(3) * 2 * np.pi
            R = meshes[j].get_rotation_matrix_from_xyz(euler_angles)
            meshes[j].rotate(R, center=(0, 0, 0))

            c_w = (np.random.rand(3) - 0.5) * 2 * mesh_scale * radius_scale
            # (3,) [-mesh_scale * radius, mesh_scale * radius]
            cs = meshes[j].get_axis_aligned_bounding_box().get_center()
            meshes[j].translate(c_w - cs, relative=True)

        # combine mesh
        mesh = meshes[0]
        for j in range(1, len(meshes)):
            mesh = mesh + meshes[j]

        # translate and sclae the combined mesh
        center_w = np.zeros(3)
        cs = mesh.get_axis_aligned_bounding_box().get_center()
        mesh.translate(center_w - cs, relative=True)
        s = np.max(mesh.get_axis_aligned_bounding_box().get_half_extent())
        mesh.scale(scale=mesh_scale / s, center=np.zeros((3, 1)))

        # convert o3d mesh to structures.mesh
        mesh = structures.Mesh(
            mesh=mesh,
            scale=mesh_scale,
        )
        all_meshes.append(mesh)

    return all_meshes


def get_dataset(
        dataset_name: str,
        dataset_info: T.Dict[str, T.Any],
        input_camera_setting: T.Dict[str, T.Any] = None,
        input_camera_trajectory_params: T.Dict[str, T.Any] = None,
        output_camera_setting: T.Dict[str, T.Any] = None,
        output_camera_trajectory_params: T.Dict[str, T.Any] = None,
        rank: int = 0,
        world_size: int = 1,
        printout: bool = True,
        imagenet_root_dir: str = 'datasets/imagenet',
) -> T.Dict[str, T.Any]:
    mesh_filename_dict = get_dataset_mesh_filename_config()

    if dataset_name not in mesh_filename_dict:
        raise RuntimeError(f'{dataset_name} not in {get_dataset_mesh_filename()}')

    dataset_filename_dict = mesh_filename_dict[dataset_name]

    # download dataset
    filename_dict = gather_and_clean_dataset(
        dataset_name=dataset_name,
        dataset_root_dir=dataset_filename_dict['dataset_root_dir'],
        cleaned_root_dir=dataset_filename_dict.get('cleaned_root_dir', None),
        clean_mesh=dataset_filename_dict.get('clean_mesh', False),
        train_mesh_filenames=dataset_filename_dict['train'],
        test_mesh_filenames=dataset_filename_dict.get('test', None),
        rank=rank,
        world_size=world_size,
        printout=printout,
    )
    train_mesh_filenames = filename_dict['train_mesh_filenames']
    test_mesh_filenames = filename_dict['test_mesh_filenames']

    if world_size > 1:
        torch.distributed.barrier()

    # make sure all files exist
    for filename in train_mesh_filenames:
        assert os.path.exists(filename), f'train: {filename} not exist'
    if test_mesh_filenames is not None:
        for filename in test_mesh_filenames:
            assert os.path.exists(filename), f'test: {filename} not exist'

    # load mesh_filename to structures.Mesh
    if printout:
        print(f'Loading training meshes...', flush=True)

    if dataset_info.get('mix_meshes', False):
        # combine multiple meshes
        meshes: T.List[structures.Mesh] = load_and_mix_mesh_filename(
            mesh_filenames=train_mesh_filenames,
            mesh_scale=dataset_info.get('mesh_scale', 1.),
            min_num_mesh=dataset_info.get('min_num_mesh', 1),
            max_num_mesh=dataset_info.get('max_num_mesh', 2),
            radius_scale=dataset_info.get('radius_scale', 2),
            total_combined=dataset_info.get('total_combined', None),
            printout=printout,
        )
    else:
        meshes: T.List[structures.Mesh] = load_mesh_filename(
            mesh_filenames=train_mesh_filenames,
            mesh_scale=dataset_info.get('mesh_scale', 1.),
            printout=printout,
        )

    if world_size > 1:
        torch.distributed.barrier()

    if printout:
        print(f'number of meshes = {len(meshes)}', flush=True)

    # load replacement texture maps
    texture_mode = dataset_info.get('texture_mode', 'ori')
    if texture_mode is None or texture_mode == 'ori':
        texture_filenames = None
    elif texture_mode == 'files' and dataset_info.get('texture_filenames', None) is not None:
        texture_filenames = dataset_info.get('texture_filenames', None)
    elif texture_mode == 'imagenet':
        assert imagenet_root_dir is not None
        assert os.path.exists(imagenet_root_dir), f'imagenet: {imagenet_root_dir} not exist'
        # gather all textures
        texture_filenames = glob(os.path.join(imagenet_root_dir, '**/*.JPEG'), recursive=True)
    else:
        raise NotImplementedError

    if world_size > 1:
        torch.distributed.barrier()

    datasets = []
    val_datasets = []
    test_datasets = []

    if printout:
        print(f'creating training and validation datasets...', flush=True)
    for i in range(len(meshes)):
        if printout:
            print(f'creating training dataset: {i}/{len(meshes)}', flush=True)

        mesh = meshes[i]
        datasets.append(
            mesh_dataset_v2.MeshDataset(
                mesh=mesh,
                n_target_imgs=dataset_info['n_target_imgs'],
                n_imgs=dataset_info['n_imgs'],
                total=dataset_info['total'],
                input_camera_setting=input_camera_setting,
                input_camera_trajectory_params=input_camera_trajectory_params,
                output_camera_setting=output_camera_setting,
                output_camera_trajectory_params=output_camera_trajectory_params,
                rng_seed=dataset_info.get('dataset_rng_seed', None),
                render_method=dataset_info['render_method'],
                texture_filenames=texture_filenames,
                texture_crop_method=dataset_info.get('texture_crop_method', 'ori'),
                min_subsample=dataset_info.get('min_subsample', 1),
                max_subsample=dataset_info.get('max_subsample', 1),
            )
        )

        # validation datasets are the same meshes but different views
        val_datasets.append(
            mesh_dataset_v2.MeshDataset(
                mesh=mesh,
                n_target_imgs=dataset_info['n_target_imgs'],
                n_imgs=dataset_info['n_imgs'],
                total=dataset_info.get('total_valid', 100),
                input_camera_setting=input_camera_setting,
                input_camera_trajectory_params=input_camera_trajectory_params,
                output_camera_setting=output_camera_setting,
                output_camera_trajectory_params=output_camera_trajectory_params,
                rng_seed=dataset_info.get('dataset_rng_seed', 0) + 1,
                render_method=dataset_info['render_method'],
                texture_filenames=texture_filenames,
                texture_crop_method=dataset_info.get('texture_crop_method', 'ori'),
                min_subsample=dataset_info.get('min_subsample', 1),
                max_subsample=dataset_info.get('max_subsample', 1),
            )
        )

    if printout:
        print(f'number of meshes in datasets = {len(datasets)}')
        print(f'number of meshes in val_datasets = {len(datasets)}')
    dataset = mesh_dataset_v2.MeshConcatDataset(datasets)
    val_dataset = mesh_dataset_v2.MeshConcatDataset(val_datasets)

    if test_mesh_filenames is None or len(test_mesh_filenames) == 0:
        test_dataset = None
    else:
        if printout:
            print(f'Loading test meshes...', flush=True)

        if dataset_info.get('mix_meshes', False):
            # combine multiple meshes
            meshes: T.List[structures.Mesh] = load_and_mix_mesh_filename(
                mesh_filenames=test_mesh_filenames,
                mesh_scale=dataset_info.get('mesh_scale', 1.),
                min_num_mesh=dataset_info.get('min_num_mesh', 1),
                max_num_mesh=dataset_info.get('max_num_mesh', 2),
                radius_scale=dataset_info.get('radius_scale', 2),
                total_combined=dataset_info.get('total_combined', None),
                printout=printout,
            )
        else:
            meshes: T.List[structures.Mesh] = load_mesh_filename(
                mesh_filenames=test_mesh_filenames,
                mesh_scale=dataset_info.get('mesh_scale', 1.),
                printout=printout,
            )

        if printout:
            print(f'creating test datasets...', flush=True)
        for i in range(len(meshes)):
            if printout:
                print(f'creating test dataset: {i}/{len(meshes)}', flush=True)
            mesh = meshes[i]
            test_datasets.append(
                mesh_dataset_v2.MeshDataset(
                    mesh=mesh,
                    n_target_imgs=dataset_info['n_target_imgs'],
                    n_imgs=dataset_info['n_imgs'],
                    total=dataset_info.get('total_test', 100),
                    input_camera_setting=input_camera_setting,
                    input_camera_trajectory_params=input_camera_trajectory_params,
                    output_camera_setting=output_camera_setting,
                    output_camera_trajectory_params=output_camera_trajectory_params,
                    rng_seed=dataset_info.get('dataset_rng_seed', 0) + 1,
                    render_method=dataset_info['render_method'],
                    texture_filenames=texture_filenames,
                    texture_crop_method=dataset_info.get('texture_crop_method', 'ori'),
                    min_subsample=dataset_info.get('min_subsample', 1),
                    max_subsample=dataset_info.get('max_subsample', 1),
                )
            )
        test_dataset = mesh_dataset_v2.MeshConcatDataset(test_datasets)

    return dict(
        dataset=dataset,  # concated dataset
        datasets=datasets,  # list of individual dataset
        val_dataset=val_dataset,
        val_datasets=val_datasets,
        test_dataset=test_dataset,
        test_datasets=test_datasets,
    )
