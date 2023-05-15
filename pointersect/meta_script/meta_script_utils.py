#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import os
import typing as T
import yaml


def get_base_dir() -> str:
    filename = os.path.normpath(
        os.path.join(
            os.path.abspath(__file__),
            '../../..',
        ))
    return filename


def read_config(filename: str) -> T.Dict[str, T.Any]:
    with open(filename) as file:
        default_config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return default_config_dict


def write_config_file(filename: str, config_dict: T.Dict[str, T.Any]):
    with open(filename, 'w') as file:
        config_txt = yaml.dump(config_dict, file, default_flow_style=False)
    return config_txt


def remove_bad_char(name: str) -> str:
    bad_chars = [';', ':', '!', "*", ' ', '+']
    for c in bad_chars:
        name = name.replace(c, '_')
    bad_chars = ['=']
    for c in bad_chars:
        name = name.replace(c, '-')
    bad_chars = [',', '[', ']', '(', ')']
    for c in bad_chars:
        name = name.replace(c, '')
    return name


def compile_command(
        script_filename: str,
        num_gpus: int,
        params: T.Dict[str, T.Any],
        switch_names: T.List[str] = tuple(),
        prefix: str = '/venv',
        use_xvfb: bool = False,
) -> str:
    """
    Compose the command to execute on bolt.

    Args:
        script_filename:
            the .py filename.  Note that it is assumed to be executed from the base folder.
        num_gpus:
            number of gpus
        params:
            the command-line arguments. They will be composed as --{key} {val}, except for
            those contained in `switch_names` (which will become --{key} if val = true).
        switch_names:
            a list of names of params to be treated as switches.
        use_xvfb:
            if true, use xvfb to simulate an x window

    Returns:

    """

    xvfb_prefix = 'xvfb-run -a -s "-screen 0 800x600x24" ' if use_xvfb else ''
    if num_gpus > 1:
        command = f'source /miniconda/etc/profile.d/conda.sh; ' \
                  f'conda activate {prefix}; ' \
                  f'CUDA_HOME="{prefix}/pkgs/cuda-toolkit" ' \
                  f'PYTHONPATH="." ' \
                  f'{xvfb_prefix}' \
                  f'torchrun --standalone --nnodes=1 --nproc_per_node={num_gpus} ' \
                  f'{script_filename} '
    else:
        command = f'source /miniconda/etc/profile.d/conda.sh; ' \
                  f'conda activate {prefix}; ' \
                  f'CUDA_HOME="{prefix}/pkgs/cuda-toolkit" ' \
                  f'PYTHONPATH="." ' \
                  f'{xvfb_prefix}' \
                  f'python ' \
                  f'{script_filename} '

    for key in params.keys():
        val = params[key]
        if key in switch_names:
            if val:
                command += '--{} '.format(key)
        else:
            if isinstance(val, (list, tuple)):
                command += '--{} '.format(key)
                for v in val:
                    command += '{} '.format(v)
            else:
                command += '--{} {} '.format(key, val)
    return command

