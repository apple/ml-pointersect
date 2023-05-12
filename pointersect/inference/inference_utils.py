#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# This file implements util function for inference.
import os
import sys

sys.path.append('cdslib')
import numpy as np
import imageio
import torch
import typing as T
from pointersect.models.pointersect import SimplePointersect
from cdslib.core.models import model_utils as cds_model_utils
from plib import render
from plib import metrics


def get_pointersect_max_ray_chunk_size(k: int) -> int:
    """
    Returns the max ray chunk size.
    """
    if torch.cuda.is_available():
        mem_free, total_mem = torch.cuda.mem_get_info()  # in bytes
        mem_free_gb = mem_free / (1024 ** 3)
    else:
        raise RuntimeError('need gpu')

    if k <= 20:
        chunk_size = 1_000_000 / 80 * mem_free_gb
    elif k <= 40:
        chunk_size = 400_000 / 80 * mem_free_gb
    elif k <= 100:
        chunk_size = 100_000 / 80 * mem_free_gb
    elif k <= 200:
        chunk_size = 25_000 / 80 * mem_free_gb
    else:
        raise RuntimeError('Does not support k > 200 currently')
    return chunk_size


def load_pointersect(
        filename: str,
        device: torch.device = torch.device('cpu'),
) -> T.Tuple[SimplePointersect, T.Dict[str, T.Any]]:
    """
    load a pretrained pointersect model from bolt.

    Args:
        filename:
            filename of the model checkpoint
        device:
            device to load the model

    Returns:
        pointersect model
    """

    model_dict, checkpoint = cds_model_utils.load_model(
        filename=filename,
        model_names='model',
        model_classes=SimplePointersect,
        model_params_names='model_info',
        device=device,
    )
    # assume load only one model
    assert len(model_dict) == 1
    model_name = 'model'
    if isinstance(model_name, (list, tuple)):
        model_name = model_name[0]
    model = model_dict[model_name]

    model.eval()
    model.to(device=device)

    model_info = dict(
        filename=filename,
    )
    return model, model_info


def save_imgs(
        imgs: torch.Tensor,
        save_npy: bool,
        save_gif: bool,
        save_png: bool,
        output_dir: str,
        overwrite: bool = False,
        gif_fps: int = 10,
):
    """
    Save image tensor as numpy and optionally gif and png
    Args:
        imgs:
            (b, h, w, 3)
        save_gif:
            whether to save gif
        save_png:
            whether to save pngs of each images
        output_dir:
            output folder
        overwrite:
            whether to overwrite the old content
    """
    if os.path.exists(output_dir) and not overwrite:
        raise RuntimeError
    os.makedirs(output_dir, exist_ok=True)

    imgs = imgs.detach().cpu().numpy()

    # save raw npy
    if save_npy:
        filename = os.path.join(output_dir, 'imgs.npy')
        np.save(filename, imgs)

    # gif
    if save_gif:
        filename = os.path.join(output_dir, 'imgs.gif')
        render.create_gif(
            images=imgs,
            filename=filename,
            fps=gif_fps,
        )

    # pngs
    if save_png:
        subdir = os.path.join(output_dir, 'images')
        os.makedirs(subdir, exist_ok=True)
        for i in range(imgs.shape[0]):
            filename = os.path.join(subdir, f'{i}.png')
            imageio.imwrite(filename, (imgs[i] * 255.).astype(np.uint8))


def compute_mse(
        arr: torch.Tensor,
        ref: torch.Tensor,
        ndim_b: int = None,
        valid_mask: torch.Tensor = None,
):
    """
    Compute the mean squared error between arr and ref.
    The average is taken over the d_shape.

    Args:
        arr: (*b_shape, *d_shape)
        ref: (*b_shape, *d_shape)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        valid_mask: (*b_shape, *d_shape)

    Returns:
        mse: (*b_shape,)
    """
    if ndim_b is None:
        ndim_b = 0

    squared_error = (arr - ref) ** 2  # (*b, *d)
    squared_error = squared_error.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
    if valid_mask is None:
        mse = squared_error.mean(dim=-1)  # (*b,)
    else:
        valid_mask = valid_mask.view(
            *(valid_mask.shape), *([1] * (arr.ndim - valid_mask.ndim))).expand_as(arr)
        valid_mask = valid_mask.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        mse = (squared_error * valid_mask).sum(dim=-1) / valid_mask.sum(-1)
    return mse


def compute_rmse(
        arr: torch.Tensor,
        ref: torch.Tensor,
        ndim_b: int = None,
        valid_mask: torch.Tensor = None,
):
    """
    Compute the root mean squared error between arr and ref.
    The average is taken over the d_shape.

    Args:
        arr: (*b_shape, *d_shape)
        ref: (*b_shape, *d_shape)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        valid_mask: (*b_shape, *d_shape)

    Returns:
        mse: (*b_shape,)
    """
    mse = compute_mse(arr=arr, ref=ref, ndim_b=ndim_b, valid_mask=valid_mask)  # (*b,)
    rmse = mse ** 0.5  # (*b,)
    return rmse


def compute_psnr(
        arr: torch.Tensor,
        ref: torch.Tensor,
        ndim_b: int = None,
        max_val: float = 1.,
        valid_mask: torch.Tensor = None,
):
    """
    Compute peak signal to noise ratio
    Args:
        arr: (*b_shape, *d_shape)
        ref: (*b_shape, *d_shape)
        ndim_b:
            number of dimension of b_shape. If None, = 0.

    Returns:
        psnr: (*b_shape,)
    """

    mse = compute_mse(arr=arr, ref=ref, ndim_b=ndim_b, valid_mask=valid_mask)  # (*b,)
    psnr = 10 * torch.log10((max_val * max_val) / mse)  # (*b,)
    return psnr


def compute_ssim(
        arr: torch.Tensor,
        ref: torch.Tensor,
        ndim_b: int = None,
        # valid_mask: torch.Tensor = None,
):
    """
    Compute the ssim.
    Args:
        arr: (*b_shape, *d_shape)
        ref: (*b_shape, *d_shape)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        # valid_mask: (*b_shape, *d_shape)

    Returns:
        ssim_scores: (*b_shape,)

    Note:
        This function is NOT differentiable
    """
    if ndim_b is None:
        ndim_b = 0

    ori_shape = arr.shape
    b_shape = ori_shape[:ndim_b]
    d_shape = ori_shape[ndim_b:]
    arr = arr.reshape(-1, *d_shape)  # (b, *d)
    ref = ref.reshape(-1, *d_shape)  # (b, *d)
    b = arr.size(0)

    assert len(d_shape) == 3
    assert d_shape[-1] >= 3

    ssim_scores = []
    for ib in range(b):
        # if valid_mask is None:
        ssim_score = metrics.ssim(rgb=arr[ib], gts=ref[ib])  # float
        ssim_scores.append(ssim_score)
    ssim_scores = torch.tensor(ssim_scores, dtype=torch.float, device=arr.device)
    ssim_scores = ssim_scores.reshape(*b_shape)
    return ssim_scores


def compute_lpips(
        arr: torch.Tensor,
        ref: torch.Tensor,
        ndim_b: int = None,
        lpips_model: torch.nn.Module = None,
        device: torch.device = torch.device('cuda'),
):
    """
    Compute the LPIPS metric.
    Args:
        arr: (*b_shape, *d_shape)  Assumes the RGB image is in [0,1]
        ref: (*b_shape, *d_shape)  Assumes the RGB image is in [0,1]
        ndim_b:
            number of dimension of b_shape. If None, = 0.

    Returns:
        lpips_score: (*b_shape,)

    Note:
        This function is NOT differentiable
    """
    if ndim_b is None:
        ndim_b = 0

    ori_shape = arr.shape
    b_shape = ori_shape[:ndim_b]
    d_shape = ori_shape[ndim_b:]
    arr = arr.reshape(-1, *d_shape)  # (b, *d)
    ref = ref.reshape(-1, *d_shape)  # (b, *d)
    b = arr.size(0)

    arr_device = arr.device

    arr = arr.to(device=device)
    ref = ref.to(dtype=arr.dtype, device=device)

    assert len(d_shape) == 3
    assert d_shape[-1] >= 3

    scores = []
    for ib in range(b):
        score = metrics.lpips(rgb=arr[ib], gts=ref[ib], lpips_model=lpips_model)  # float
        scores.append(score)
    scores = torch.tensor(scores, dtype=torch.float, device=arr_device)
    scores = scores.reshape(*b_shape)
    return scores.to(device=arr_device)


def compute_l1(
        arr: torch.Tensor,
        ref: torch.Tensor,
        ndim_b: int = None,
        valid_mask: torch.Tensor = None,
):
    """
    Compute average l1 distance between arr and ref.

    Args:
        arr: (*b_shape, *d_shape)
        ref: (*b_shape, *d_shape)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        valid_mask: (*b_shape, *d_shape)

    Returns:
        err: (*b_shape,)
    """
    if ndim_b is None:
        ndim_b = 0

    err = (arr - ref).abs()  # (*b, *d)
    err = err.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
    if valid_mask is None:
        err = err.mean(dim=-1)  # (*b,)
    else:
        valid_mask = valid_mask.view(
            *(valid_mask.shape), *([1] * (arr.ndim - valid_mask.ndim))).expand_as(arr)
        valid_mask = valid_mask.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        err = (err * valid_mask).sum(dim=-1) / valid_mask.sum(-1)

    return err


def compute_area(
        arr: torch.Tensor,
        ref: torch.Tensor,
        ndim_b: int = None,
        normalized: bool = True,
        valid_mask: torch.Tensor = None,
):
    """
    Compute the area spanned by the unit vectors in arr and ref.

    Args:
        arr: (*b_shape, *d_shape, 3)
        ref: (*b_shape, *d_shape, 3)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        normalized:
            whetehr arr and ref are unit vectors
        valid_mask: (*b_shape, *d_shape,)

    Returns:
        err: (*b_shape,)
    """
    if ndim_b is None:
        ndim_b = 0

    if not normalized:
        arr = torch.nn.functional.normalize(arr, p=2, dim=-1)
        ref = torch.nn.functional.normalize(ref, p=2, dim=-1)

    out = torch.linalg.cross(arr, ref, dim=-1)  # (*b, *d, 3)
    area = torch.linalg.vector_norm(out, ord=2, dim=-1)  # (*b, *d,)

    if valid_mask is None:
        area = area.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        area = area.mean(dim=-1)  # (*b,)
    else:
        valid_mask = valid_mask.view(
            *(valid_mask.shape), *([1] * (area.ndim - valid_mask.ndim))).expand_as(area)
        valid_mask = valid_mask.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        area = area.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        area = (area * valid_mask).sum(dim=-1) / valid_mask.sum(-1)

    return area


def compute_diff_angle(
        arr: torch.Tensor,
        ref: torch.Tensor,
        ndim_b: int = None,
        normalized: bool = True,
        valid_mask: torch.Tensor = None,
):
    """
    Compute the angle spanned by the unit vectors in arr and ref.

    Args:
        arr: (*b_shape, *d_shape, 3)
        ref: (*b_shape, *d_shape, 3)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        normalized:
            whetehr arr and ref are unit vectors
        valid_mask: (*b_shape, *d_shape,)

    Returns:
        err: (*b_shape,) angle in degree
    """
    if ndim_b is None:
        ndim_b = 0

    if not normalized:
        arr = torch.nn.functional.normalize(arr, p=2, dim=-1)
        ref = torch.nn.functional.normalize(ref, p=2, dim=-1)

    # make sure arr and ref points to the same direction
    out = torch.sum(arr * ref, dim=-1)  # (*b, *d)
    arr = arr * out.sign().unsqueeze(-1)

    # recompute inner product
    out = torch.sum(arr * ref, dim=-1)  # (*b, *d)

    angle = torch.arccos(out.clamp(min=-1 + 1e-9, max=1 - 1e-9)) * (180. / torch.pi)  # (*b, *d) in degree
    if valid_mask is None:
        angle = angle.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        angle = angle.mean(dim=-1)  # (*b,)
    else:
        valid_mask = valid_mask.view(
            *(valid_mask.shape), *([1] * (angle.ndim - valid_mask.ndim))).expand_as(angle)
        valid_mask = valid_mask.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        angle = angle.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        angle = (angle * valid_mask).sum(dim=-1) / valid_mask.sum(-1)
    return angle


def compute_accuracy(
        arr: torch.Tensor,
        ref: torch.Tensor,
        ndim_b: int = None,
        valid_mask: torch.Tensor = None,
):
    """
    Compute the accuracy.

    Args:
        arr: (*b_shape, *d_shape), binary
        ref: (*b_shape, *d_shape), binary
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        valid_mask: (*b_shape, *d_shape,)

    Returns:
        acc: (*b_shape,)
    """
    if ndim_b is None:
        ndim_b = 0

    same = (arr > 0.5) == (ref > 0.5)  # (*b, *d)
    same = same.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
    if valid_mask is None:
        acc = same.float().mean(dim=-1)  # (*b,)
    else:
        valid_mask = valid_mask.view(
            *(valid_mask.shape), *([1] * (arr.ndim - valid_mask.ndim))).expand_as(arr)
        valid_mask = valid_mask.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        acc = (same.float() * valid_mask).sum(dim=-1) / valid_mask.sum(-1)
    return acc
