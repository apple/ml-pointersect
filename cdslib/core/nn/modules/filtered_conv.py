#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
# Author: Rick Chang
# This file implements a stacked convolution network that applies
# a low-pass filter before sub-sampling or up-sampling.

import math
import typing as T
import warnings

import torch
import torch.nn.functional as F

from cdslib.core.nn.functional import upfirdn1d

from .conv import Conv1DLayer
from .conv import ConvTranspose1DLayer
from .linear import LinearLayer


class StackedFilteredConv1dLayers(torch.nn.Module):
    """
    Convenient helper nn.Module to create a stack of filtered conv1D layers.
    """

    def __init__(
        self,
        in_channels: int,
        layer_configs: T.Sequence[T.Dict[str, T.Any]],
        blur_kernel=(1, 3, 3, 1),
    ):
        """
        Construct multiple stacked conv1D layers.
        It applies a low-pass filtering for every upsampling/downsampling to avoid aliasing.

        Overview of the layers:
            If upsample:
                blur -> conv -> normalization -> nonlinear -> dropout
            Else if downsample:
                conv -> blur -> normalization -> nonlinear -> dropout
            Else:
                conv -> normalization -> nonlinear -> dropout

        Args:
            in_channels (int):
                Number of input channels.
            layer_configs:
                A list of dict (one for each layer) containing the parameters to the :py:class:`ModulatedConv1d` layer:

                The key-value pairs include:
                    'out_channels' (required, int):
                        output channel of the current layer
                    'kernel_size' (required, int):
                        size of the 1D convolution kernel
                    'type' (str):
                        ``'same'``, ``'upsample'`` (by 2), ``'downsample'`` (by 2).
                        Default: ``'same'``.
                    'pad_type' (str):
                        ``'same'``, ``'valid'``. Default: ``'same'``.
                    'padding_mode' (str):
                        ``'zeros'``, ``'reflect'``, ``'replicate'``, ``'circular'``.
                        Default: 'zeros'.
                    'dilation' (int):
                        dilation of the conv layer. Default: ``1``.
                    'groups' (int):
                        number of groups from input channels to output channels.
                        (has to be 1 if demodulate is used). Default: ``1``.
                    'bias':
                        bool, whether to add a learnable bias to the output. Default: ``True``.
                    'dropout' (float):
                        dropout rate. Default: ``0``.
                    'nonlinearity' (str):
                        ``'none'``, ``'leaky_relu'``, ``'relu'``, ``'tanh'``,
                        ``'sigmoid'``, ``'silu'``. Default: ``'relu'``.
                    'norm_layer' (str):
                        'none', 'demodulate', 'layernorm' 'batchnorm', 'instancenorm'. Default: 'none'.
            blur_kernel (tuple of int):
                The blur blur_kernel used to remove aliasing caused by up/down-sampling
        """
        super().__init__()
        self.in_channels = in_channels
        self.layer_configs = layer_configs
        self.num_layers = len(self.layer_configs)

        # adding default layer configs
        for layer_config in self.layer_configs:
            assert "out_channels" in layer_config
            assert "kernel_size" in layer_config
            if "type" not in layer_config:
                layer_config["type"] = "same"
            if "pad_type" not in layer_config:
                layer_config["pad_type"] = "same"
            if "padding_mode" not in layer_config:
                layer_config["padding_mode"] = "zeros"
            if "dilation" not in layer_config:
                layer_config["dilation"] = 1
            if "groups" not in layer_config:
                layer_config["groups"] = 1
            if "bias" not in layer_config:
                layer_config["bias"] = True
            if "dropout" not in layer_config:
                layer_config["dropout"] = 0
            if "nonlinearity" not in layer_config:
                layer_config["nonlinearity"] = "relu"
            if "norm_layer" not in layer_config:
                layer_config["norm_layer"] = "none"

            if layer_config["norm_layer"] == "demodulate":
                assert layer_config["groups"] == 1

        # construct the layers
        self.pres = torch.nn.ModuleList()  # blur layer if downsample
        self.main = torch.nn.ModuleList()  # the main conv layer
        self.posts = torch.nn.ModuleList()  # blur layer if upsample
        self.addons = torch.nn.ModuleList()  # normalization -> nonlinear

        dh = self.in_channels
        for layer_idx in range(self.num_layers):
            layer_config = layer_configs[layer_idx]

            if layer_config["norm_layer"] == "demodulate":
                self.main.append(
                    ModulatedConv1d(
                        in_channels=dh,
                        out_channels=layer_config["out_channels"],
                        kernel_size=layer_config["kernel_size"],
                        style_dim=0,
                        demodulate=True,
                        dilation=layer_config["dilation"],
                        type=layer_config["type"],  # upsample/downsample/same
                        pad_type=layer_config["pad_type"],
                        blur_kernel=blur_kernel,
                    )
                )
                self.pres.append(None)
                self.posts.append(None)
            else:
                # up/down-sampling
                if layer_config["type"] == "upsample":
                    # construct a FIR filter that is applied after upsampling to avoid aliasing
                    if layer_config["pad_type"] == "same":
                        s = 2
                        p = len(blur_kernel) - 2 + s - layer_config["dilation"] * (layer_config["kernel_size"] - 1)
                        pad0 = (p + 1) // 2
                        pad1 = p // 2
                    elif layer_config["pad_type"] == "valid":
                        pad0 = 0
                        pad1 = 0
                    else:
                        raise NotImplementedError
                    blur = Blur1D(blur_kernel, pad=(pad0, pad1), kernel_scale=2)
                    self.pres.append(None)
                    self.posts.append(blur)
                elif layer_config["type"] == "downsample":
                    # construct a FIR filter that is applied before downsampling to avoid aliasing
                    if layer_config["pad_type"] == "same":
                        s = 2
                        p = len(blur_kernel) - s + layer_config["dilation"] * (layer_config["kernel_size"] - 1)
                        pad0 = (p + 1) // 2
                        pad1 = p // 2
                    elif layer_config["pad_type"] == "valid":
                        pad0 = 0
                        pad1 = 0
                    else:
                        raise NotImplementedError
                    blur = Blur1D(blur_kernel, pad=(pad0, pad1), kernel_scale=1)
                    self.pres.append(blur)
                    self.posts.append(None)
                elif layer_config["type"] == "same":
                    self.pres.append(None)
                    self.posts.append(None)
                else:
                    raise NotImplementedError

                # conv layers
                if layer_config["type"] == "upsample":
                    conv = ConvTranspose1DLayer(
                        in_channels=dh,
                        out_channels=layer_config["out_channels"],
                        kernel_size=layer_config["kernel_size"],
                        stride=2,
                        padding=0,
                        output_padding=0,
                        dilation=layer_config["dilation"],
                        groups=layer_config["groups"],
                        bias=layer_config["bias"],
                        padding_mode=layer_config["padding_mode"],
                        w_init_gain=layer_config["nonlinearity"],
                    )
                elif layer_config["type"] == "downsample":
                    conv = Conv1DLayer(
                        in_channels=dh,
                        out_channels=layer_config["out_channels"],
                        kernel_size=layer_config["kernel_size"],
                        stride=2,
                        padding=0,
                        dilation=layer_config["dilation"],
                        groups=layer_config["groups"],
                        bias=layer_config["bias"],
                        padding_mode=layer_config["padding_mode"],
                        w_init_gain=layer_config["nonlinearity"],
                    )
                elif layer_config["type"] == "same":
                    if layer_config["pad_type"] == "same":
                        padding = (layer_config["dilation"] * (layer_config["kernel_size"] - 1)) // 2
                    elif layer_config["pad_type"] == "valid":
                        padding = 0
                    else:
                        raise NotImplementedError
                    conv = Conv1DLayer(
                        in_channels=dh,
                        out_channels=layer_config["out_channels"],
                        kernel_size=layer_config["kernel_size"],
                        stride=1,
                        padding=padding,
                        dilation=layer_config["dilation"],
                        groups=layer_config["groups"],
                        bias=layer_config["bias"],
                        padding_mode=layer_config["padding_mode"],
                        w_init_gain=layer_config["nonlinearity"],
                    )
                else:
                    raise NotImplementedError
                self.main.append(conv)

            # update channel dimension
            dh = self.layer_configs[layer_idx]["out_channels"]

            # addons
            additional_layers = []
            if layer_config["norm_layer"] in {"none", "demodulate"}:
                layer_config["permute_for_norm"] = False
            else:
                if layer_config["norm_layer"] == "layernorm":
                    layer = torch.nn.LayerNorm(dh)
                    layer_config["permute_for_norm"] = True
                elif layer_config["norm_layer"] == "batchnorm":
                    layer = torch.nn.BatchNorm1d(dh)
                    layer_config["permute_for_norm"] = False
                elif layer_config["norm_layer"] == "instancenorm":
                    layer = torch.nn.InstanceNorm1d(dh)
                    layer_config["permute_for_norm"] = False
                else:
                    raise NotImplementedError
                additional_layers.append(layer)

            # nonlinear
            if layer_config["nonlinearity"] == "none":
                layer = None
            elif layer_config["nonlinearity"] == "leaky_relu":
                layer = torch.nn.LeakyReLU(inplace=False, negative_slope=0.2)
            elif layer_config["nonlinearity"] == "relu":
                layer = torch.nn.ReLU(inplace=False)
            elif layer_config["nonlinearity"] == "tanh":
                layer = torch.nn.Tanh()
            elif layer_config["nonlinearity"] == "sigmoid":
                layer = torch.nn.Sigmoid()
            elif layer_config["nonlinearity"] == "silu":
                if torch.__version__ >= "1.7.0":
                    layer = torch.nn.SiLU(inplace=False)
                else:
                    raise NotImplementedError
                    warnings.warn("SiLu is only available in pytorch >= 1.7.0. Replace with relu.")
                    layer = torch.nn.ReLU(inplace=False)
            else:
                raise ValueError("unsupported nonlinearity")
            if layer is not None:
                additional_layers.append(layer)

            if layer_config["dropout"] > 0:
                additional_layers.append(torch.nn.Dropout(p=layer_config["dropout"], inplace=False))

            if len(additional_layers) > 0:
                self.addons.append(torch.nn.Sequential(*additional_layers))
            else:
                self.addons.append(None)

        # count total downsampling layers
        self.num_downsampling_layers = 0
        for layer_idx in range(self.num_layers):
            layer_config = self.layer_configs[layer_idx]
            if layer_config["type"] == "downsample":
                self.num_downsampling_layers += 1

    def forward(self, x, batch_first=True):
        r"""

        Args:
            x (batch, dim_input, seq_len) or (seq_len, batch, dim_input):
                the input sequence.
            batch_first (bool):
                whether the batch dimension is the first dimension or the second.

        Returns:
            output: (batch, dim_output, seq_len_out) or (seq_len, batch, dim_input)

        How to compute output size
            The ooutput sequence length depends on the type of the layer used.

            Let stride = s, dilation = d, kernel_size = k, blur_kernel_size = kb

            upsample layer:
                .. math::

                    \text{output seq_len} & = s * Lin - s + d * (\text{kernel_size}-1) + 2 + pad0 + pad1 - kb \\
                    & = s * Lin, \text{if pad_type = same} \\
                    & = s * Lin - s + d * (\text{kernel_size}-1) + 2 - kb, \text{if pad_type = valid}

            downsample layer:
                .. math::
                    \text{output seq_len} & = ( Lin + pad0 + pad1 - kb - d * (k-1) + s ) // s \\
                    & = Lin // s, \text{if pad_type = same} \\
                    & = ( Lin - kb - d * (k-1) + s ) // s, \text{if pad_type = valid}

            same layer:
                .. math::
                    \text{output seq_len} & = Lin + 2p - d * (k-1) \\
                    & = Lin \text{ (or} Lin -1 \text{)},  \text{if pad_type = same and d * (k-1) is even (odd)} \\
                    & = Lin - d * (k-1),  \text{if pad_type = valid}

        """
        # make sure x is (batch, cin, seq_len)
        if not batch_first:
            # convert (seq_len, batch, dim_input) to (batch, dim_input, seq_len)
            x = x.permute(1, 2, 0)

        # check if the input sequence length is enough
        for i in range(len(self.main)):
            # pre
            if self.pres[i] is not None:
                x = self.pres[i](x)
            # main
            if self.main[i] is not None:
                x = self.main[i](x)
            # post
            if self.posts[i] is not None:
                x = self.posts[i](x)
            # addon
            if self.addons[i] is not None:
                if self.layer_configs[i]["permute_for_norm"]:
                    x = x.permute(0, 2, 1)
                x = self.addons[i](x)
                if self.layer_configs[i]["permute_for_norm"]:
                    x = x.permute(0, 2, 1)

        # revert back to the original order
        if not batch_first:
            # convert (batch, dim, seq_len) to (seq_len, batch, dim)
            x = x.permute(2, 0, 1)

        return x

    def compute_output_seq_len(self, seq_len_in: int, pad_type=None) -> T.List[int]:
        """Compute the output sequence length.

        Args:
            pad_type:
                ``None``: use the original padding config
                ``'same'``: pad such that output has the same length as input
                ``'valid'``: use no padding

        Returns:
            a list of seq_len, one of each layer output
        """
        assert pad_type is None or pad_type == "valid"
        Louts = []
        Lin = seq_len_in

        for i in range(self.num_layers):
            # pre
            if self.pres[i] is not None:
                Lin = self.pres[i].compute_output_seq_len(seq_len_in=Lin, pad_type=pad_type)
            # main
            if self.main[i] is not None:
                Lin = self.main[i].compute_output_seq_len(seq_len_in=Lin, pad_type=pad_type)
            # post
            if self.posts[i] is not None:
                Lin = self.posts[i].compute_output_seq_len(seq_len_in=Lin, pad_type=pad_type)
            Louts.append(Lin)
        return Louts

    def compute_seq_len_for_same(self, seq_len_in: int):
        """
        If we want to maintain the seq_len during downsampling,
        every input seq_len to a downsampling layer need to be an even number.

        Suppose we have N downsampling layer and input sequence length = L,
        we should pad the input so that its length becomes
        2^N * ceil(L / 2^N).
        """
        # when self.num_downsampling_layers == 0, it returns seq_len_in.
        ll = 2 ** self.num_downsampling_layers
        return ll * math.ceil(seq_len_in / ll)


class Blur1D(torch.nn.Module):
    """
    Upsample, finite-impulse-response filtering, downsample.
    """

    def __init__(
        self,
        kernel: T.Sequence[float],
        pad: T.Tuple[int, int],
        kernel_scale: float = 1,
    ):
        r"""
        Args:
            kernel (list of float):
                1D blur_kernel.
            pad (tuple of int):
                (pad_left, pad_right)
            kernel_scale (float):
                a scale factor multiplied to the blur_kernel's values
                to maintain constant average energy.

        Note:
            .. math:: \text{seq_len_out} = (\text{seq_len_in} * up + pad0 + pad1 - \text{kernel_size}) // down + 1
        """
        super().__init__()

        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel /= kernel.sum()  # normalize sum = 1
        kernel = kernel * kernel_scale
        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, input):
        r"""
        Args:
            input (batch, cin, seq_len):
                input sequence

        Returns:
            output (batch, cin, seq_len_out):

                .. math:: \text{seq_len_out} = (\text{seq_len_in} * up + pad_0 + pad_1 - \text{kernel_size}) // down + 1
        """
        out = upfirdn1d(input, self.kernel.to(dtype=input.dtype), pad=self.pad)
        return out

    def compute_output_seq_len(self, seq_len_in: int, pad_type=None):
        assert pad_type is None or pad_type == "valid"
        up, down = 1, 1
        if pad_type is None:
            pad = self.pad
        elif pad_type == "valid":
            pad = [0, 0]
        else:
            raise NotImplementedError
        # seq_len_out = (seq_len_in * up + pad[0] + pad[1] - len(self.kernel)) // down + 1
        seq_len_out = (
            torch.div(
                seq_len_in * up + pad[0] + pad[1] - len(self.kernel),
                down,
                rounding_mode="floor",
            )
            + 1
        )
        return seq_len_out


class ModulatedConv1d(torch.nn.Module):
    r"""
    The layer implements style modulated 1D convolution.

        :math:`y = mod(s,k) * x`,

        where :math:`*` is the convolusion,
        :math:`mod(s,k)` first multiplies i-th dimension of :math:`s` to i-th channel in :math:`k`,
        then it normalizes the weights so that the output dimensions of the weight have unit norm.

    Ref: https://arxiv.org/pdf/1912.04958.pdf  Equation 1-3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        dilation: int = 1,
        type: str = "same",
        pad_type: str = "same",
        bias: bool = True,
        modulation_bias: bool = True,
        fixed_modulation_bias: float = None,
        blur_kernel=(1, 3, 3, 1),
    ):
        r"""
        Args:
            in_channels (int):
                number of input channels of the blur_kernel
            out_channels (int):
                number of output channels of the blur_kernel
            kernel_size (int):
                size of the blur_kernel
            style_dim (int):
                dimension of the style.  If <= 0: no style modulation.
            demodulate (bool):
                whether to demodulate the weights so that the output weights have unit norm.
            dilation (int):
                dilation of the kernel.
            type (str):
                ``'same'``, ``'upsample'``,  ``'downsample'``.
                Whether to keep dimension, upsample (by 2), or downsample (by 2), respectively.
            pad_type (str):
                ``'same'``, ``'valid'``.
            bias (bool):
                whether to learn the bias
            modulation_bias (bool):
                whether to learn the modulation linear layer's bias.
            fixed_modulation_bias (float):
                a fixed (the modulation linear layer's) bias to add after linear ouptut. None: no fixed bias
            blur_kernel (tuple of float):
                the FIR filter kernel to avoid aliasing.

        Returns:
            Output sequence length:

                let stride = s, dilation = d, kernel_size = k.

                upsampling:
                    .. math::

                        \text{output seq_len} & = s * Lin - s + d * (k-1) + 2 + (pad0 + pad1 - kb) \\
                        & = s * Lin, \text{ if }  pad0 + pad1 = kb + s - 2 - d * (k - 1) \\
                        & = s * Lin, \text{ if }  pad0 = (a + 1) // 2, pad1 = a // 2, \text{where} a = kb + s - 2 - d * (k - 1)

                downsampling:
                    .. math::

                        \text{output seq_len} & = ( Lin + pad0 + pad1 - kb - d * (k-1) + s ) // s \\
                        & = Lin // s, \text{ if }  pad0 + pad1 = kb - s + d * (k - 1) \\
                        & = Lin // s, \text{ if } pad0 = (a + 1) // 2, pad1 = a // 2, \text{where} a = kb - s + d * (k - 1)

                same:
                    .. math::

                        \text{output seq_len} & = Lin + 2p - d * (k-1) \\
                        & = Lin, \text{ if } d * (k-1) \text{ is even} \\
                        & = Lin - 1, \text{ if } d (k-1) \text{ is odd}
        """
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.type = type
        self.style_dim = style_dim
        self.pad_type = pad_type
        self.dilation = dilation
        self.modulation_bias = modulation_bias
        self.fixed_modulation_bias = fixed_modulation_bias
        self.blur_kernel = blur_kernel

        if self.type == "upsample":
            # construct a FIR filter that is applied after upsampling to avoid aliasing
            if self.pad_type == "same":
                s = 2
                p = len(blur_kernel) + s - 2 - self.dilation * (kernel_size - 1)
                pad0 = (p + 1) // 2
                pad1 = p // 2
            elif self.pad_type == "valid":
                pad0 = 0
                pad1 = 0
            else:
                raise NotImplementedError
            self.blur = Blur1D(blur_kernel, pad=(pad0, pad1), kernel_scale=2)
        elif self.type == "downsample":
            # construct a FIR filter that is applied before downsampling to avoid aliasing
            if self.pad_type == "same":
                s = 2
                p = len(blur_kernel) - s + self.dilation * (kernel_size - 1)
                pad0 = (p + 1) // 2
                pad1 = p // 2
            elif self.pad_type == "valid":
                pad0 = 0
                pad1 = 0
            else:
                raise NotImplementedError
            self.blur = Blur1D(blur_kernel, pad=(pad0, pad1), kernel_scale=1)
        elif self.type == "same":
            pass
        else:
            raise NotImplementedError

        fan_in = in_channels * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        if pad_type == "same":
            self.padding = (self.dilation * (kernel_size - 1)) // 2
        elif pad_type == "valid":
            self.padding = 0
        else:
            raise NotImplementedError

        self.weight = torch.nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        if self.style_dim > 0:
            self.modulation = LinearLayer(
                in_features=style_dim,
                out_features=in_channels,
                bias=modulation_bias,
                w_init_gain="linear",
                bias_init_val=1.0,
                fixed_bias=fixed_modulation_bias,
            )

        self.demodulate = demodulate

    def forward(self, input: torch.Tensor, style: torch.Tensor = None):
        """
        Args:
            input: (batch, cin, seq_len)
            style: (batch, dim_style)

        Returns:
            (batch, cout, seq_len_out)
        """
        batch, in_channel, height = input.shape
        weight = self.scale * self.weight  # (1, cout, cin, kernel_size)

        if self.style_dim > 0 and style is not None:
            style = self.modulation(style).view(
                batch,
                1,
                in_channel,
                1,
            )  # (batch, 1, cin, 1,)
            weight = weight * style  # (batch, cout, cin, kernel_size)
        else:
            weight = weight.expand(batch, -1, -1, -1)

        if self.bias is not None:
            bias = self.bias.repeat(batch)  # (batch*cout,), same as bias.expand(batch,-1).reshape(-1)
        else:
            bias = None

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3]) + self.eps)
            weight = weight * demod.view(
                batch,
                self.out_channels,
                1,
                1,
            )  # (batch, cout, cin, kernel_size)

        if self.type == "upsample":
            input = input.view(1, batch * in_channel, height)  # (1, b*cin, h) to use group_convolution
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel,
                self.out_channels,
                self.kernel_size,
            )  # (b*cin, cout, k)
            out = F.conv_transpose1d(
                input,
                weight,
                bias=bias,
                padding=0,
                stride=2,
                groups=batch,
                dilation=self.dilation,
            )  # (1, b*cout, hout), seq_len -> 2*seq_len
            _, _, height = out.shape
            out = out.view(batch, self.out_channels, height)
            out = self.blur(out)  # seq_len -> seq_len
        elif self.type == "downsample":
            weight = weight.view(
                batch * self.out_channels,
                in_channel,
                self.kernel_size,
            )
            input = self.blur(input)
            (
                _,
                _,
                height,
            ) = input.shape
            input = input.view(1, batch * in_channel, height)
            out = F.conv1d(
                input,
                weight,
                bias=bias,
                padding=0,
                stride=2,
                groups=batch,
                dilation=self.dilation,
            )
            _, _, height = out.shape
            out = out.view(batch, self.out_channels, height)
        elif self.type == "same":
            input = input.view(1, batch * in_channel, height)
            weight = weight.view(
                batch * self.out_channels,
                in_channel,
                self.kernel_size,
            )
            out = F.conv1d(
                input,
                weight,
                bias=bias,
                padding=self.padding,
                groups=batch,
                dilation=self.dilation,
            )
            _, _, height = out.shape
            out = out.view(batch, self.out_channels, height)
        else:
            raise NotImplementedError

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, {self.kernel_size}, "
            f"type={self.type}, pad_type={self.pad_type}"
        )

    def compute_output_seq_len(self, seq_len_in: int, pad_type=None) -> int:
        """Compute the output sequence length."""

        if pad_type is None:
            # use the current pad_type setting
            pad_type = self.pad_type

        s = 2
        d = self.dilation
        k = self.kernel_size
        kb = len(self.blur_kernel)

        if self.type == "upsample":
            if pad_type == "same":
                seq_len_out = s * seq_len_in
            elif pad_type == "valid":
                seq_len_out = s * seq_len_in - s + d * (k - 1) + 2 - kb
            else:
                raise NotImplementedError
        elif self.type == "downsample":
            if pad_type == "same":
                seq_len_out = seq_len_in // s
            elif pad_type == "valid":
                seq_len_out = (seq_len_in - kb - d * (k - 1) + s) // s
            else:
                raise NotImplementedError
        elif self.type == "same":
            if pad_type == "same":
                if d * (k - 1) % 2 == 0:
                    seq_len_out = seq_len_in
                else:
                    seq_len_out = seq_len_in - 1
            elif pad_type == "valid":
                seq_len_out = seq_len_in - d * (k - 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return seq_len_out
