#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#

import unittest
import torch
import torch.nn.functional as F
from cdslib.core.nn.functional import upfirdn2d, upfirdn1d


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    r"""Pad, upsample, FIR filter, and downsample a batch of 2D images.

    Accepts a batch of 2D images of the shape `[majorDim, minorDim(inC), inH, inW]`
    and performs the following operations for each image, batched across
    `majorDim` and `minorDim`:

    1. Upsample the image by inserting the zeros after each pixel (`upx`, `upy`).

    2. Pad the image with zeros by the specified number of pixels on each side
       (`padx0`, `padx1`, `pady0`, `pady1`). Specifying a negative value
       corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`k`), shrinking the
       image so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by throwing away pixels (`downx`, `downy`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x:      Input tensor of the shape `[majorDim, minorDim, inH, inW]`.
        k:      2D FIR filter of the shape `[firH, firW]`.
        upx:    Integer upsampling factor along the X-axis (default: 1).
        upy:    Integer upsampling factor along the Y-axis (default: 1).
        downx:  Integer downsampling factor along the X-axis (default: 1).
        downy:  Integer downsampling factor along the Y-axis (default: 1).
        padx0:  Number of pixels to pad on the left side (default: 0). After upsampling.
        padx1:  Number of pixels to pad on the right side (default: 0). After upsampling.
        pady0:  Number of pixels to pad on the top side (default: 0). After upsampling.
        pady1:  Number of pixels to pad on the bottom side (default: 0). After upsampling.
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the shape `[majorDim, minorDim, outH, outW]`, and same datatype as `x`.
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    """

    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)  # (bc, h, w)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    # use padding to upsample (insert 0)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])  # last dim's (left, right) -> first dim's (left, right)
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)  # bc, h*up, w*up, minor

    # pad if pad >= 0
    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    # remove if pad < 0
    out = out[
          :,
          max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
          max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
          :,
          ]

    out = out.permute(0, 3, 1, 2)  # bc, minor, h*up, w*up
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]  # bc*minor, 1, h*up, w*up
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)  # make sure it is convolution
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)  # bc, h, w, minor
    out = out[:, ::down_y, ::down_x, :]  # downsample by direct subsampling

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)  # b, c, h, w


class TestUpfirdn2D(unittest.TestCase):
    def _test(self, batch, c, h, w, wh, ww,
              up_x, up_y, down_x, down_y,
              pad_x0, pad_x1, pad_y0, pad_y1):
        input = torch.randn(batch, c, h, w)
        kernel = torch.randn(wh, ww)
        out_rick = upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y,
            pad_x0, pad_x1, pad_y0, pad_y1)
        out_native = upfirdn2d_native(
            input, kernel, up_x, up_y, down_x, down_y,
            pad_x0, pad_x1, pad_y0, pad_y1)
        assert torch.allclose(out_rick, out_native)

        input_rick = input.clone()
        kernel_rick = kernel.clone()
        input_rick.requires_grad = True
        kernel_rick.requires_grad = True
        out_rick = upfirdn2d(
            input_rick, kernel_rick,
            up_x, up_y, down_x, down_y,
            pad_x0, pad_x1, pad_y0, pad_y1)
        loss = out_rick.mean()
        loss.backward()

        input_native = input.clone()
        kernel_native = kernel.clone()
        input_native.requires_grad = True
        kernel_native.requires_grad = True
        out_native = upfirdn2d(
            input_native, kernel_native,
            up_x, up_y, down_x, down_y,
            pad_x0, pad_x1, pad_y0, pad_y1)
        loss = out_native.mean()
        loss.backward()
        assert torch.allclose(input_rick.grad, input_native.grad)
        assert torch.allclose(kernel_rick.grad, kernel_native.grad)

    def test(self):
        self._test(
            batch=5, c=10,
            h=32, w=33,
            wh=4, ww=8,
            up_x=1, up_y=1,
            down_x=1, down_y=1,
            pad_x0=3, pad_x1=2,
            pad_y0=5, pad_y1=1,
        )

    def test1(self):
        self._test(
            batch=5, c=10,
            h=1, w=30,
            wh=3, ww=4,
            up_x=2, up_y=3,
            down_x=2, down_y=2,
            pad_x0=5, pad_x1=8,
            pad_y0=3, pad_y1=6,
        )

    def test2(self):
        self._test(
            batch=5, c=10,
            h=13, w=30,
            wh=3, ww=4,
            up_x=2, up_y=3,
            down_x=2, down_y=2,
            pad_x0=-1, pad_x1=3,
            pad_y0=-2, pad_y1=6,
        )


class TestUpfirdn1D(unittest.TestCase):
    def _test(self, batch, c, h, wh,
              up, down, pad_0, pad_1):
        input = torch.randn(batch, c, h)
        kernel = torch.randn(wh)

        out_rick = upfirdn1d(
            input, kernel, up, down, (pad_0, pad_1))

        up_x, up_y, down_x, down_y = 1, up, 1, down
        pad_x0, pad_x1, pad_y0, pad_y1 = 0, 0, pad_0, pad_1
        input_native = input.unsqueeze(-1)
        kernel_native = kernel.unsqueeze(-1)
        out_native = upfirdn2d_native(
            input_native, kernel_native, up_x, up_y, down_x, down_y,
            pad_x0, pad_x1, pad_y0, pad_y1).squeeze(-1)
        assert torch.allclose(out_rick, out_native)

        input_rick = input.clone()
        kernel_rick = kernel.clone()
        input_rick.requires_grad = True
        kernel_rick.requires_grad = True
        out_rick = upfirdn1d(
            input_rick, kernel_rick,
            up, down, (pad_0, pad_1))
        loss = out_rick.mean()
        loss.backward()

        input_native = input_native.clone()
        kernel_native = kernel_native.clone()
        input_native.requires_grad = True
        kernel_native.requires_grad = True
        out_native = upfirdn2d(
            input_native, kernel_native,
            up_x, up_y, down_x, down_y,
            pad_x0, pad_x1, pad_y0, pad_y1).squeeze(-1)
        loss = out_native.mean()
        loss.backward()
        assert torch.allclose(input_rick.grad, input_native.grad.squeeze(-1))
        assert torch.allclose(kernel_rick.grad, kernel_native.grad.squeeze(-1))

    def test(self):
        self._test(
            batch=5, c=10,
            h=32, wh=4,
            up=1, down=1,
            pad_0=5, pad_1=2,
        )

    def test2(self):
        self._test(
            batch=5, c=20,
            h=103, wh=5,
            up=2, down=2,
            pad_0=0, pad_1=1,
        )

    def test3(self):
        self._test(
            batch=5, c=20,
            h=100, wh=40,
            up=1, down=1,
            pad_0=10, pad_1=10,
        )


if __name__ == '__main__':
    unittest.main()
