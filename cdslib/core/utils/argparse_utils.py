#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
import argparse
import json
import os
from pprint import pprint
import shlex
import typing as T
import warnings

import yaml

from cdslib.core.nn import HyperParams


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    elif v.lower() == "none":
        return None
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def read_config_file(
    filename: str,
) -> T.Dict[str, T.Any]:
    """
    Read a config file and return a dict (arg_name -> val).

    Args:
        filename:
            json or yaml filename

    Returns:
        a dictionary maps argument name (str) -> val
    """

    if os.path.exists(filename):
        ext = os.path.splitext(filename)[1]
        if ext == ".json":
            with open(filename, "r") as f:
                config_dict = json.load(f)
        elif ext == ".yaml":
            with open(filename, "r") as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise NotImplementedError(f"{filename} ({ext}) not supported")
    else:
        raise RuntimeError(f"{filename} not exist")

    return config_dict


def compile_argparser_str(
    config_dict: T.Dict[str, T.Any],
    switches: T.List[str] = None,
    printout: bool = False,
) -> T.List[str]:
    """
    Given a config_dict: key -> val, returns a list of
    ["--key", "val"] so that argparse can parse it.

    Args:
        config_dict:
            a dict of key -> val
        switches:
            a list containing the str that should be considered as switch
            when creating commandline parse string
        printout:
            whether to print the parsed results

    Returns:
        a list containing the cmd that can be parsed by `argparse`.
    """

    if switches is None:
        switches = {}

    cmd_str = ""
    for arg_name, val in config_dict.items():
        if arg_name in switches:
            if val:
                cmd_str += f"--{arg_name} "
        else:
            cmd_str += f"--{arg_name} "
            if isinstance(val, (list, tuple)):
                for v in val:
                    cmd_str += f"{v} "
            else:
                cmd_str += f"{val} "

    cmd_list = shlex.split(cmd_str)
    if printout:
        pprint(f"cmd_str: {cmd_str}")
        pprint(f"cmd_list: {cmd_list}")
    return cmd_list


def print_options(
    options,
    parser: argparse.ArgumentParser = None,
    output_file: str = None,
    output_json_file: str = None,
    output_yaml_file: str = None,
):
    """
    Print the options parsed by ArgumentParser.

    Args:
        options:
            parsed options
        parser:
            the parser used to parse `options`.
        output_file:
            if given, will save a stdout into a txt file
        output_json_file:
            if given, will save the options into a json file.
         output_yaml_file:
            if given, will save the options into a yaml file.

    Notes:
        The function will print both current options and
        their default values (if parser is given and if they are different).
    """

    if not isinstance(options, dict):
        options_dict = dict()
        for k, v in sorted(vars(options).items()):
            options_dict[k] = v
    else:
        options_dict = options

    message = ""
    message += "----------------- Options ---------------\n"

    for k, v in sorted(options_dict.items()):
        comment = ""
        if parser is not None:
            default = parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    print(message)

    # save to the disk
    if output_file is not None and options_dict.get("rank", 0) == 0:
        with open(output_file, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    # save to the disk
    if output_json_file is not None and options_dict.get("rank", 0) == 0:
        with open(output_json_file, "w") as opt_file:
            json.dump(options_dict, opt_file)

    # save to the disk
    if output_yaml_file is not None and options_dict.get("rank", 0) == 0:
        with open(output_yaml_file, "w") as opt_file:
            yaml.dump(options_dict, opt_file, default_flow_style=False)


class WarnDict(dict):
    """
    A dictionary that warns when using get and allows access its key/values
    like they are its attributes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, key, default=None):
        if key not in self:
            warnings.warn(f"access {key} not in dict, use {default}")

        return super().get(key, default)

    def __getattr__(self, name: str) -> T.Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: T.Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def recursive_dict_update(
    tgt_dict: T.MutableMapping[T.Any, T.Any],
    src_dict: T.MutableMapping[T.Any, T.Any],
):
    """
    Recursively update the content in `src_dict` into `tgt_dict`.

    For example,
    tgt_dict = {
        'a': 1,
        'b': {'c': 2},
        'd': 3,
    }
    src_dict = {
        'a': 4,
        'b': {'e': 5},
        'f': 6,
    }

    After the function tgt_dict will contain:
    tgt_dict = {
        'a': 4,
        'b': {'c': 2, 'e': 5},
        'd': 3,
        'f': 6,
    }
    Notice that the function does not replace tgt_dict['b'] with the new
    dictionary, but update it.

    Args:
        tgt_dict:
            dictionary to be updated
        src_dict:
            dictionary containing the source content
    """

    if src_dict is None:
        return

    for key, val in src_dict.items():
        if key not in tgt_dict or tgt_dict[key] is None or isinstance(tgt_dict[key], HyperParams.TBD):
            tgt_dict[key] = val
        else:
            if isinstance(val, (dict, HyperParams)):
                assert isinstance(tgt_dict[key], (dict, HyperParams)), f"{key}, {type(tgt_dict[key])}"
                recursive_dict_update(tgt_dict[key], src_dict[key])
            else:
                tgt_dict[key] = val
