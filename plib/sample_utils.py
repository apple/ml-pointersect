#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
from scipy.stats import qmc
import numpy as np
import torch
import typing as T


def get_np_dtype(dtype: T.Union[int, float, torch.dtype, np.dtype]):
    """
    Return the cooresponding numpy dtype.
    """
    if dtype == int:
        return np.int64
    elif dtype == float:
        return np.float64
    elif dtype in {np.float32, np.float64}:
        return dtype
    elif dtype == torch.float32:
        return np.float32
    elif dtype == torch.float64:
        return np.float64
    else:
        raise NotImplementedError


def get_torch_dtype(dtype: T.Union[int, float, torch.dtype, np.dtype]):
    """
    Return the cooresponding numpy dtype.
    """
    if dtype == int:
        return torch.long
    elif dtype == float:
        return torch.float64
    elif dtype in {torch.float32, torch.float64}:
        return dtype
    elif dtype == np.float32:
        return torch.float32
    elif dtype == np.float64:
        return torch.float64
    else:
        raise NotImplementedError


def get_samples(
        total_samples: int,
        d: int,
        method: str = 'random',
        rng=None,
        fast_forward_n: int = None,
        shuffle: bool = False,
) -> np.ndarray:
    """
    Uniformly distribute `total_samples` within [0, 1)^{d}.

    Args:
        total_samples:
            total number of samples
        d:
            number of dimension
        method:
            'random':
            'LatinHypercube':
        fast_forward:
            number of samples already generated
        shuffle:
            whether to shuffle the samples (along total_samples).
            It is needed when combining the samples with other samples.

    Returns:
        (total_samples, d)
    """

    if method == 'random':
        if rng is None:
            return np.random.rand(total_samples, d)
        else:
            return rng.rand(total_samples, d)
    elif method.lower() == 'LatinHypercube'.lower():
        sampler = qmc.LatinHypercube(d=d, seed=rng)
        if fast_forward_n is not None and fast_forward_n > 0:
            sampler = sampler.fast_forward(fast_forward_n)
        samples = sampler.random(n=total_samples)  # (total_samples, d)
        if shuffle:
            samples = shuffle_along_axis(arr=samples, axis=0, rng=rng)
        return samples
    else:
        raise NotImplementedError


def shuffle_along_axis(arr: np.ndarray, axis: int, rng=None) -> np.ndarray:
    """
    Shuffle `arr` along `axis`.
    Args:
        arr:
            (*,)
        axis:

    Returns:
        (*,)
    """
    if rng is None:
        rng = np.random

    idx = rng.rand(*arr.shape).argsort(axis=axis)
    return np.take_along_axis(arr, idx, axis=axis)
