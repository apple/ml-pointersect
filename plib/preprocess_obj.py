#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
import warnings

# The file generates cleaned obj file readable by open3d.

import numpy as np
import os

import shutil
from PIL import Image
from plib import mesh_utils


def generated_cleaned_mesh(
        input_dir: str,
        output_dir: str,
        relative_filepath: str,
        skip_exist: bool = True,
        printout: bool = True,
) -> str:
    """
    Generate a cleaned mesh in the new directory.

    Args:
        input_dir: input folder of the mesh (obj)
        output_dir: output folder of the mesh (obj)
        relative_filepath: filename / relative path of the mesh (obj)
        skip_exist: if true, skip if the cleaned mesh already exists.

    Returns:
        full path of the output file
    """

    # check whether the cleaned obj already exists
    output_filepath = os.path.join(output_dir, relative_filepath)

    if printout:
        print(f'cleaning {os.path.join(input_dir, relative_filepath)} to {output_filepath}')

    if os.path.exists(output_filepath) and skip_exist:
        if printout:
            print(f'cleaned file {output_filepath} exists, skipped', flush=True)
        return output_filepath

    # Copy the obj and related files to new dir
    output_mtl_filepath = copy_obj_files(
        input_dir=input_dir,
        output_dir=output_dir,
        relative_filepath=relative_filepath,
    )

    # Handle non-textured mesh
    map_kd_value_to_textures(output_mtl_filepath)
    # Remove duplicated mesh with smaller face scalar
    remove_redundant_face(output_filepath)

    return output_filepath


def copy_obj_files(
        input_dir: str,
        output_dir: str,
        relative_filepath: str,
) -> str:
    """
    Copy the obj and all related files into a new directory.

    Args:
        input_dir: input folder of the mesh (obj)
        output_dir: output folder of the mesh (obj)
        relative_filepath: filename / relative path of the mesh (obj)

    Returns:
        full path of the output file
    """
    input_filepath = os.path.join(input_dir, relative_filepath)
    output_filepath = os.path.join(output_dir, relative_filepath)
    input_full_dir, filename = os.path.split(input_filepath)
    output_full_dir, _ = os.path.split(output_filepath)

    # get mtl file
    mtl_line = ''
    with open(input_filepath, 'r') as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            if line.lstrip().startswith('mtllib'):
                mtl_line = line.lstrip()
                mtl_line = mtl_line[6:].lstrip().replace('\n', '')
                break

    if not mtl_line:  # empty
        print('no mtl file found!')
        return ''

    input_mtl_filepath = os.path.join(input_full_dir, mtl_line)
    # force the output name of material file as filename+'.mtl'
    output_mtl_filepath = os.path.join(output_full_dir, filename + '.mtl')
    # make the mtl file is in the same folder with obj file (it should)
    lines[line_id] = f'mtllib ./{filename}.mtl'

    # copy obj file (change the refered mtl file name)
    if not os.path.exists(output_full_dir):
        os.makedirs(output_full_dir)

    with open(output_filepath, 'w') as f:
        for line in lines:
            f.write(line)

    # copy mtl file
    shutil.copy(input_mtl_filepath, output_mtl_filepath)

    # copy all textures
    with open(input_mtl_filepath, 'r') as f:
        lines = f.readlines()
        tex_lines = [line.lstrip() for line in lines if line.lstrip().startswith('map_Kd')]
        tex_path_list = [tex_line[6:].lstrip().replace('\n', '') for tex_line in tex_lines]  # extract texture file path
        for tex_path in tex_path_list:
            # assume the original mtl file is in the same folder with obj file again (it should)
            input_tex_path = os.path.join(input_full_dir, tex_path)
            output_tex_path = os.path.join(output_full_dir, tex_path)
            output_tex_dir, _ = os.path.split(output_tex_path)
            if not os.path.exists(output_tex_dir):
                os.makedirs(output_tex_dir)
            # sometimes the texture file is not existed at all, need to check first
            if os.path.exists(input_tex_path):
                shutil.copy(input_tex_path, output_tex_dir)

    return output_mtl_filepath


def remove_redundant_face(
        filepath: str
):
    """
    Remove repetitive face in a mesh and replace the original mesh.

    Args:
        filepath: path of the mesh (obj)
    """
    warnings.warn(
        'We do not release this part of the code. '
        'If a high level, the function reads a mesh, '
        'computes ambient occlusion on mesh (important),'
        'reorders the faces, and deletes redundant faces. '
        'We use this function only to clean up ShapeNet meshes, '
        'so the results you got on ShapeNet may be different '
        'from ours.'
    )


def map_kd_value_to_textures(mtl_filepath: str):
    """
    For materials without texture map, this function will create texture map based on Kd.
    For materials with texture map, this function will multiply the map with Kd value.

    Args:
        mtl_filepath: the path to obj material. This function will overwrite mtl and texture maps in the destination

    Returns:
    """
    with open(mtl_filepath, 'r+') as f:
        mtl_dir, _ = os.path.split(mtl_filepath)
        mtl_content = f.read().split('newmtl')
        f.seek(0)
        new_texture_id = 0
        for material in mtl_content:
            material_lines = material.splitlines()
            Kd_line_ids = [i for i, x in enumerate(material_lines) if x.lstrip().startswith('Kd')]
            texture_lines = [x for x in material_lines if x.lstrip().startswith('map_Kd')]
            if Kd_line_ids:  # Kd_line is not empty, assume it is a valid material
                material_lines[0] = 'newmtl' + material_lines[0]  # add delimiter back
                Kd_line = material_lines[Kd_line_ids[0]]
                Kd_values = np.asarray(Kd_line.split()[1:], dtype='f4')  # parse three Kd values

                texture_line = None
                if texture_lines:
                    texture_line = texture_lines[0].lstrip()  # ignore spaces begin
                    texture_line = texture_line[6:].replace(" ", "")  # remove map_Kd and space
                    # some files may be missing, check first
                    if not os.path.exists(os.path.join(mtl_dir, texture_line)):
                        # remove the map line
                        material_lines.remove(texture_lines[0])
                        texture_line = None

                if texture_line is None:
                    Kd_values = (Kd_values * 255).astype(np.uint8)
                    Kd_img = np.tile(np.reshape(Kd_values, [1, 1, 3]), (100, 100, 1))
                    #  for some reason, open3d would make texture boundary darker
                    #  thus, use small textures (1x1 image) would make texture color non-uniform
                    #  use 100x100 is more robust
                    Kd_img = Image.fromarray(Kd_img)  # reshape to a 1x1 image
                    new_texture_path = f"new_textures/{new_texture_id}.png"
                    new_texture_dir = os.path.join(mtl_dir, 'new_textures')
                    if not os.path.exists(new_texture_dir):
                        os.makedirs(new_texture_dir)
                    Kd_img.save(os.path.join(mtl_dir, new_texture_path))
                    material_lines[-1] = f"map_Kd ./{new_texture_path}"  # replace last empty line by map_Kd
                    material_lines.append("")
                    new_texture_id += 1
                else:
                    with Image.open(os.path.join(mtl_dir, texture_line)) as Kd_img:
                        Kd_img_array = mesh_utils.clean_texture(np.asarray(Kd_img))
                        Kd_img_array = Kd_img_array * Kd_values
                        Kd_img = Image.fromarray(Kd_img_array.astype(np.uint8))
                        Kd_img.save(os.path.join(mtl_dir, texture_line))
                material_lines[Kd_line_ids[0]] = 'Kd 1 1 1'  # no need for Kd values anymore

            for x in material_lines:
                f.write(x + "\n")
        f.truncate()
