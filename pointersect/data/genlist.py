#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# The script is used to pick mesh files from dataset

import glob
# add path for demo utils functions
import sys
import os
import argparse
import numpy as np
import json

sys.path.append(os.path.abspath(''))


# randomly pick mesh files from datasets

def get_tex_model_list(
        setting: str,
        num_classes: int = 3,
        rnd_seed: int = 0
) -> list:
    """
    given setting (train/test), generate a list of obj file names in the tex dataset
    Args:
        setting: train/test
        num_classes: number of class in training
        rnd_seed: random seed

    Returns:
        a list of filenames

    """
    class_list = ['buddha', 'buddha2', 'bunny', 'cat', 'feline', 'tiger', 'zebra']

    if rnd_seed > 0:  # negative: use sorted class_list
        np.random.seed(rnd_seed)
        class_list = np.random.permutation(class_list).tolist()

    if setting == 'train':
        class_list = class_list[:num_classes]
    elif setting == 'test':
        class_list.reverse()
        class_list = class_list[:num_classes]

    obj_path_list = [c + '.obj' for c in class_list]

    return obj_path_list


def get_ShapeNet_model_list(
        rootpath: str,
        setting: str,
        num_classes: int = 3,
        num_sample_in_class: int = 2,
        rnd_seed: int = 987,
        use_sub_classlist: bool = False,
) -> list:
    """
    Given setting (train/test), generate a list of obj file names in the shapenet dataset.

    Args:
        rootpath: the root directory of the ShapeNet dataset
        setting: train/test
        num_classes: number of class
        num_sample_in_class: number of mesh sample in each class
        rnd_seed: random seed
        use_sub_classlist: use specific class set for train/ test

    Returns:
        a list of filenames
    """
    np.random.seed(rnd_seed)
    img_types = ('*.jpg', '*.png', '*.jpeg', '*.JPG')
    obj_path_list = []

    folder_list = sorted([fname for fname in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, fname))])
    with open(os.path.join(rootpath, 'taxonomy.json')) as f:
        class_json_list = json.load(f)
        all_class_list = [(class_json['name'].split(',')[0].replace(' ', '_'), class_json['synsetId']) for class_json in
                          class_json_list if class_json['synsetId'] in folder_list]
        # all_class_list = [ c[0].split(',')[0].replace(' ','_') for c in all_class_list ]
        class_to_folder = dict(all_class_list)
        """
            class_to_folder = dict(
                airplane='02691156',
                bench='02828884',
                cabinet='02933112',
                car='02958343',
                chair='03001627',
                display='03211117',
                lamp='03636649',
                loudspeaker='03691459',
                rifle='04090263',
                sofa='04256520',
                table='04379243',
                cellular_telephone='04401088',
                vessel='04530566',
            )
        """

        # In other works, the first 3 classes below are split to training set and the other 10 classes are for testing
        fixed_class = ['airplane', 'car', 'chair', 'bench', 'rifle', 'vessel', 'cabinet', 'display', 'lamp',
                       'loudspeaker', 'sofa', 'table', 'cellular_telephone']
        if setting == 'train':
            class_list = fixed_class[:3]
        elif setting == 'test':
            class_list = fixed_class[3:]

        if not use_sub_classlist:  # use all classes
            full_class_list = list(class_to_folder.keys())
            extra_class_list = [c for c in full_class_list if c not in fixed_class]
            extra_class_list = np.random.permutation(extra_class_list).tolist()

            if setting == 'train':
                class_list = class_list + extra_class_list[:27]
            elif setting == 'test':
                class_list = class_list + extra_class_list[27:]

        # Find objs
        for c in class_list[:num_classes]:
            folder = class_to_folder[c]
            folder_path = os.path.join(rootpath, folder)
            obj_list = sorted(os.listdir(folder_path))
            obj_list = np.random.permutation(obj_list).tolist()
            obj_count = 0

            for obj in obj_list:
                has_img = False
                for t in img_types:
                    path_images = os.path.join(folder_path, obj, '**models**', t)
                    has_img = has_img or len(glob.glob(path_images, recursive=True)) != 0
                    path_images = os.path.join(folder_path, obj, '**images**', t)
                    has_img = has_img or len(glob.glob(path_images, recursive=True)) != 0

                if has_img:  # currently only use objs with textures
                    obj_path_list.append(os.path.join(folder, obj, 'models/model_normalized.obj'))
                    obj_count += 1
                if obj_count >= num_sample_in_class:
                    break

    return obj_path_list


def get_Sketchfab_with_license_model_list(
        rootpath: str,
        setting: str,
        num_classes: int = 3,
        rnd_seed: int = 0
) -> list:
    np.random.seed(rnd_seed)
    if setting == 'train':
        gltf_list = sorted(os.listdir(os.path.join(rootpath, 'train')))
    elif setting == 'test':
        gltf_list = sorted(os.listdir(os.path.join(rootpath, 'test')))

    gltf_list = np.random.permutation(gltf_list).tolist()

    gltf_list = [os.path.join(gltf, 'scene.gltf') for gltf in gltf_list[:num_classes]]

    return gltf_list


def get_Sketchfab_model_list(
        rootpath: str,
        setting: str,
        num_classes: int = 3,
        rnd_seed: int = 0
) -> list:
    np.random.seed(rnd_seed)
    if setting == 'train':
        off_list = sorted(os.listdir(os.path.join(rootpath, 'train_mesh')))
    elif setting == 'test':
        off_list = sorted(os.listdir(os.path.join(rootpath, 'test_mesh')))

    off_list = np.random.permutation(off_list).tolist()

    off_list = [setting + '_mesh/' + off for off in off_list[:num_classes]]

    return off_list


# below are only for debug this function individually
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument(
        '--rootpath',
        type=str,
        default='./',
        help='rootpath of the dataset')

    parser.add_argument(
        '--lstfilepath',
        type=str,
        default='./',
        help='path of train.lst and test.lst to save')

    parser.add_argument(
        '--setting',
        type=str,
        help='train or test')

    options = parser.parse_args()
    obj_path_list = get_ShapeNet_model_list(options.rootpath, options.setting)
    file_lst_path = os.path.join(options.lstfilepath, options.setting + '_new.lst')
    with open(file_lst_path, 'w') as f:
        for obj_path in obj_path_list:
            f.write(obj_path + '\n')
