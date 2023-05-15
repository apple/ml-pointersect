#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#

import typing as T

import torch
from torch.nn import functional as F


def upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    r"""Pad, upsample, FIR filter, and downsample a batch of 2D images.

    Accepts a batch of 2D images of the shape `[batch, inC, inH, inW]`
    and performs the following operations for each image (batch * inC).

    1. Upsample the image by inserting the zeros after each pixel (`upx`, `upy`).

    2. Pad the image with zeros by the specified number of pixels on each side
       (`padx0`, `padx1`, `pady0`, `pady1`). Specifying a negative value
       corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`k`) with `valid` mode.

    4. Subsample (downsample) the image by (`downx`, `downy`).

    Args:
        x (batch, c, h, w):
            Input tensor of the shape `[Batch, C, inH, inW]`.
        k (kh, kw):
            2D FIR filter of the shape `[firH, firW]`.
        up_x (int):
            Integer upsampling factor along the X-axis (default: 1).
        up_y (int):
            Integer upsampling factor along the Y-axis (default: 1).
        down_x (int):
            Integer downsampling factor along the X-axis (default: 1).
        down_y (int):
            Integer downsampling factor along the Y-axis (default: 1).
        pad_x0 (int):
            Number of pixels to pad on the left side (default: 0). After upsampling.
        pad_x1 (int):
            Number of pixels to pad on the right side (default: 0). After upsampling.
        pad_y0 (int):
            Number of pixels to pad on the top side (default: 0). After upsampling.
        pad_y1 (int):
            Number of pixels to pad on the bottom side (default: 0). After upsampling.

    Returns:
        Tensor of the shape `[batch, inC, outH, outW]`, and same datatype as `x`.

        .. math::
            out_h = (in_h * up_y + pad_{y0} + pad_{y1} - kernel_h) // down_y + 1

        .. math::
            out_w = (in_w * up_x + pad_{x0} + pad_{x1} - kernel_w) // down_x + 1
    """

    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w)  # (bc, h, w)

    _, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input  # (bc, h, w)

    if up_x > 1 or up_y > 1:
        out = out.view(-1, in_h, 1, in_w, 1)
        # upsample (insert 0)
        out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1])  # last dim's -> first dim's
        out = out.view(-1, in_h * up_y, in_w * up_x)  # (bc, h*up, w*up)

    # pad if pad >= 0
    out = F.pad(
        out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )  # (bc, h + pad_y0 + pad_y1, w + pad_x0 + pad_x1)
    # remove if pad < 0
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
    ]  # (bc, h + pad_y0 + pad_y1, w + pad_x0 + pad_x1)

    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])  # (bc, 1, h*up, w*up)
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)  # make sure it is convolution
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )  # (bc, h*up, w*up)

    if down_x > 1 or down_y > 1:
        out = out[:, ::down_y, ::down_x]  # downsample by subsampling

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)  # (b, c, h, w)


def upfirdn1d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    up: int = 1,
    down: int = 1,
    pad: T.Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    r"""One-dimensional upsample, fir filtering, downsample.

    Args:
        input (batch, cin, seq_len):
            input tensor
        kernel (kernel_size,):
            1D FIR filter of the shape.
        up (int):
            upsample factor
        down (int):
            downsample factor
        pad (list of int):
            (left_padding, right_padding)

    Returns:
        output (batch, cin, seq_len_out):

            .. math::
                \text{seq_len_out} = (\text{seq_len_in} * up + pad0 + pad1 - k) // down + 1

    Note that if up = 2 and down = 2, out_h = in_h if pad0 + pad1 = k-1.
    See upfirdn2d for detailed documentation.
    """
    if kernel.dim() == 1:
        # make blur_kernel (h, 1)
        kernel = kernel.unsqueeze(1)  # (h, 1)
    elif kernel.dim() == 1:
        assert kernel.size(1) == 1
    else:
        raise RuntimeError(f"1D blur_kernel wrong shape {kernel.shape}")

    # reshape input to (batch, feature_dim, seq_len, 1)
    input = input.unsqueeze(-1)  # (batch, cin, seq_len, 1)

    # mimic 1D upfirdn1d with upfirdn2d
    up_x = 1  # no upsample
    down_x = 1  # no downsample
    pad_x0 = 0
    pad_x1 = 0

    # use upfirdn2d to compute
    out = upfirdn2d(input, kernel, up_x, up, down_x, down, pad_x0, pad_x1, pad[0], pad[1])

    return out.squeeze(-1)  # (batch, cin, seq_len_out)
