#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#

import torch


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device):
    """
    Move optimizer to a specific device.

    Args:
        optimizer:
            the optimzer to move
        device:
            the torch device
    """
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
