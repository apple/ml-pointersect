#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# The file implements util functions to manipulate models.

import os
import typing as T

import torch

from cdslib.core.models.base_model import BaseModel


def load_model(
        filename: str,
        model_names: T.Union[str, T.List[str]],
        model_classes: T.Union[T.Callable, T.List[T.Callable]],
        model_params_names: T.Optional[T.Union[str, T.List[str]]] = None,
        model_patch_params: T.Optional[T.Union[T.Dict[str, T.Any], T.List[T.Dict[str, T.Any]]]] = None,
        device=torch.device("cpu"),
) -> T.Tuple[T.Dict[str, BaseModel], T.Dict[str, T.Any]]:
    """
    Load a model (:py:class:`BaseModel`) to device and as eval mode.

    Args:
        filename:
            filename of the pt file
        model_names:
            a list of name of the model in the pth file.
            In other words, we load `checkpoint[model_name]`.
        model_classes:
            a list of class definition of the model class.
            The model will load the state_dict from `checkpoint[model_name]`.
        model_params_names:
            default model params, ie, if the parameters of the model is not
            saved in checkpoint[model_name], the model will be initialized
            model_params_names.
        model_patch_params:
            manually set the parameters to overwrite the config_dict
            of the model stored in the checkpoint file.
            For example, if checkpoint[model_names[0]]['config_dict']['param1'] = 30,
            but model_patch_params[0]['param1'] = 0, the model will be created with
            `param1 = 0`.
        device:
            device to load the models.

    Returns:
        model_dict:
            a dict containing models (model_name -> model)
        checkpoint:
            checkpoint dict.  The dictionary stored by in the pretrained model .pth file.
    """

    assert os.path.exists(filename)

    if isinstance(model_names, str):
        model_names = [model_names]
    if not isinstance(model_classes, (list, tuple)):
        model_classes = [model_classes]
    if isinstance(model_params_names, str):
        model_params_names = [model_params_names]
    if model_params_names is None:
        model_params_names = [None] * len(model_names)
    assert len(model_names) == len(model_classes)
    assert len(model_names) == len(model_params_names)
    if model_patch_params is None:
        model_patch_params = [dict()] * len(model_names)
    if not isinstance(model_patch_params, (list, tuple)):
        model_patch_params = [model_patch_params]

    # load the model
    checkpoint = torch.load(filename, map_location=torch.device("cpu"))  # dict of [name, data]
    model_dict = dict()
    for i in range(len(model_names)):
        model_name = model_names[i]
        model_class = model_classes[i]
        model_params_name = model_params_names[i]
        model_patch_param = model_patch_params[i]

        model_data_dict = checkpoint[model_name]
        params = model_data_dict.get("config_dict", None)
        if params is None and model_params_name is not None:
            params = checkpoint[model_params_name]

        # create model
        model = model_class(**params, **model_patch_param)

        # load model
        model.load_state_dict(model_data_dict)

        # send model to device and eval mode
        model.to(device)
        model.eval()
        model_dict[model_name] = model

    return model_dict, checkpoint
