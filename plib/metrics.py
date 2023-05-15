#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

import numpy as np
import skimage
import skimage.metrics
import torch
from lpips import LPIPS


def psnr(
        rgb: torch.Tensor,
        gts: torch.Tensor,
) -> float:
    """
    Calculate the PSNR metric. Non-differentiable.

    Args:
        rgb: (h, w, 3), in the range of [0, 1]
        gts: (h, w, 3), in the range of [0, 1]

    Returns:
        psnr value
    """
    assert (rgb.shape[-1] == 3)
    assert (gts.shape[-1] == 3)

    mse = torch.mean((rgb[..., :3] - gts[..., :3]) ** 2).item()
    return 10 * np.log10(1.0 / mse)


def get_lpips_model(device: torch.device('cpu')) -> torch.nn.Module:
    """
    Return lpips model
    """

    lpips_model = LPIPS(net='vgg').to(device=device)
    return lpips_model

def lpips(
        rgb: torch.Tensor,
        gts: torch.Tensor,
        lpips_model: torch.nn.Module = None,
) -> float:
    """
    Convenient function to call lpips library to calculate the LPIPS metric.
    Not differentiable.

    Args:
        rgb: (h, w, 3), in the range of [0, 1]
        gts: (h, w, 3), in the range of [0, 1]

    Returns:
        LPIPS value
    """
    assert (rgb.shape[-1] == 3)
    assert (gts.shape[-1] == 3)

    if lpips_model is None:
        lpips_model = LPIPS(net='vgg').to(device=rgb.device)

    return lpips_model(
        (2.0 * rgb[..., :3] - 1.0).permute(2, 0, 1),
        (2.0 * gts[..., :3] - 1.0).permute(2, 0, 1),
    ).mean().item()


def ssim(
        rgb: torch.Tensor,
        gts: torch.Tensor,
) -> float:
    """
    Convenient function to call skimage's ssim.  Not differentiable.

    Args:
        rgb: (h, w, 3), in the range of [0, 1]
        gts: (h, w, 3), in the range of [0, 1]

    Returns:
        ssim value
    """
    return skimage.metrics.structural_similarity(
        rgb[..., :3].cpu().numpy(),
        gts[..., :3].cpu().numpy(),
        multichannel=True,
        data_range=1,
        gaussian_weights=True,
        sigma=1.5,
    )
