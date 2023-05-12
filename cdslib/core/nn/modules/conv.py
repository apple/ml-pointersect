#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
# Author: Rick Chang
# This file implements various convolution layers with customized initialization.

import math
import typing as T
import warnings

import torch
import torch.nn as nn

from .. import nn_utils


class Conv1DLayer(nn.Conv1d):
    """
    A helper class that wraps around the typical nn.Conv1D to help initialize it.
    """

    _FLOAT_MODULE = nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        w_init_gain: str = "linear",
        init_method: str = "xavier_normal",
        lrelu_nslope: float = 0.2,
        kaiming_fan_mode: str = "fan_in",
        batch_first: bool = True,
    ):
        """
        Create a one-dimensional convolution layer whose weights are initialized with a chosen method.

        The usage of the layer is the same as :py:class:`torch.nn.Conv1D`.

        Args:
            in_channels (int):
                Number of channels in the input sequence
            out_channels (int):
                Number of channels produced by the convolution
            kernel_size (int or tuple):
                Size of the convolving kernel
            stride (int or tuple, optional):
                Stride of the convolution. Default: 1
            padding (int, tuple or str, optional):
                Padding added to both sides of the input. Default: 0
            padding_mode (string, optional):
                ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
                Default: ``'zeros'``
            dilation (int or tuple, optional):
                Spacing between kernel elements. Default: 1
            groups (int, optional):
                Number of blocked connections from input channels
                to output channels. Default: 1
            bias (bool, optional):
                If ``True``, adds a learnable bias to the output. Default: ``True``
            w_init_gain:
                The nonlinearity that is designed to be added after the layer.
                Check :py:func:`cdslib.nn.init_weight` for supported nonlinearities.
                Note that the layer does not add nonlinearity.
            init_method:
                The initialization method. Check :py:func:`cdslib.nn.init_weight`.
            lrelu_nslope:
                The negative slope of leaky-relu if leaky-relu is used. Check :py:func:`cdslib.nn.init_weight`.
            kaiming_fan_mode:
                The fan mode if kaiming_* init method is used. Check :py:func:`cdslib.nn.init_weight`.
            batch_first:
                Input dimension order. True: (batch, dim, seq_len),  False: (seq_len, batch, dim)
        """
        nn.Conv1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        nn_utils.init_weight(
            self.weight,
            w_init_gain=w_init_gain,  # leaky_relu, relu, tanh, sigmoid, ...
            init_method=init_method,
            lrelu_nslope=lrelu_nslope,
            kaiming_fan_mode=kaiming_fan_mode,
        )

        self.batch_first = batch_first

    def compute_output_seq_len(self, seq_len_in: int, pad_type=None):
        """Compute the output sequence length.

        Args:
            pad_type:
                ``None``: use the original padding config
                ``'same'``: use the padding config for 'same'
                ``'valid'``: use no padding
                int: directly set the (symmetric) padding value.
        """

        s = self.stride if not isinstance(self.stride, (list, tuple)) else self.stride[0]
        d = self.dilation if not isinstance(self.dilation, (list, tuple)) else self.dilation[0]
        k = self.kernel_size if not isinstance(self.kernel_size, (list, tuple)) else self.kernel_size[0]

        if pad_type is None:
            p = self.padding if not isinstance(self.padding, (list, tuple)) else self.padding[0]
        elif pad_type == "same":
            p = (d * (k - 1)) // 2
        elif pad_type == "valid":
            p = 0
        elif isinstance(pad_type, int):
            p = pad_type
        else:
            raise NotImplementedError

        seq_len_out = math.floor((seq_len_in + 2 * p - d * (k - 1) - 1) / float(s) + 1)
        return seq_len_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return super().forward(x)
        else:
            if x.ndim == 3:
                # (seq_len, batch, dim) -> (batch, dim, seq_len)
                x = x.permute(1, 2, 0)
                y = super().forward(x)  # (batch, dim, seq_len)
                return y.permute(2, 0, 1)
            elif x.ndim == 2:
                # (seq_len, dim) -> (dim, seq_len)
                x = x.t()
                y = super().forward(x)
                return y.t()
            else:
                raise NotImplementedError


class ConvTranspose1DLayer(nn.ConvTranspose1d):
    """
    A helper class that wraps around the typical nn.ConvTranspose1D to help initialize it.
    """

    _FLOAT_MODULE = nn.ConvTranspose1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        w_init_gain: str = "linear",
        init_method: str = "xavier_normal",
        lrelu_nslope: float = 0.2,
        kaiming_fan_mode: str = "fan_in",
        batch_first: bool = True,
    ):
        """
        Create a one-dimensional transposed convolution layer whose weights
        are initialized with a chosen method.

        The usage of the layer is the same as :py:class:`torch.nn.ConvTranspose1d`.

        Args:
            in_channels (int):
                Number of channels in the input sequence
            out_channels (int):
                Number of channels produced by the convolution
            kernel_size (int or tuple):
                Size of the convolving kernel
            stride (int or tuple, optional):
                Stride of the convolution. Default: 1
            padding (int, tuple or str, optional):
                Padding added to both sides of the input. Default: 0
            output_padding (int or tuple, optional):
                Additional size added to one side
            padding_mode (string, optional):
                ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
                Default: ``'zeros'``
            dilation (int or tuple, optional):
                Spacing between kernel elements. Default: 1
            groups (int, optional):
                Number of blocked connections from input channels
                to output channels. Default: 1
            bias (bool, optional):
                If ``True``, adds a learnable bias to the output. Default: ``True``
            w_init_gain:
                The nonlinearity that is designed to be added after the layer.
                Check :py:func:`cdslib.nn.init_weight` for supported nonlinearities.
                Note that the layer does not add nonlinearity.
            init_method:
                The initialization method. Check :py:func:`cdslib.nn.init_weight`.
            lrelu_nslope:
                The negative slope of leaky-relu if leaky-relu is used. Check :py:func:`cdslib.nn.init_weight`.
            kaiming_fan_mode:
                The fan mode if kaiming_* init method is used. Check :py:func:`cdslib.nn.init_weight`.
            batch_first:
                Input dimension order. True: (batch, dim, seq_len),  False: (seq_len, batch, dim)
        """
        nn.ConvTranspose1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        with torch.no_grad():
            self.weight.transpose_(0, 1)
        nn_utils.init_weight(
            self.weight,
            w_init_gain=w_init_gain,  # leaky_relu, relu, tanh, sigmoid, ...
            init_method=init_method,
            lrelu_nslope=lrelu_nslope,
            kaiming_fan_mode=kaiming_fan_mode,
        )
        with torch.no_grad():
            self.weight.transpose_(0, 1)

        self.batch_first = batch_first

    def compute_output_seq_len(self, seq_len_in: int, pad_type=None):
        """Compute the output sequence length.

        Args:
            pad_type:
                ``None``: use the original padding config
                ``'same'``: use the padding config for 'same'
                ``'valid'``: use no padding
                int: directly set the (symmetric) padding value.
        """
        s = self.stride if not isinstance(self.stride, (list, tuple)) else self.stride[0]
        d = self.dilation if not isinstance(self.dilation, (list, tuple)) else self.dilation[0]
        k = self.kernel_size if not isinstance(self.kernel_size, (list, tuple)) else self.kernel_size[0]

        if pad_type is None:
            p = self.padding if not isinstance(self.padding, (list, tuple)) else self.padding[0]
            ou = self.output_padding if not isinstance(self.output_padding, (list, tuple)) else self.output_padding[0]
        elif pad_type == "same":
            p = (d * (k - 1)) // 2
            ou = 0
        elif pad_type == "valid":
            p = 0
            ou = 0
        elif isinstance(pad_type, int):
            p = pad_type
            ou = 0
        else:
            raise NotImplementedError

        seq_len_out = (seq_len_in - 1) * s - 2 * p + d * (k - 1) + ou + 1
        return seq_len_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return super().forward(x)
        else:
            if x.ndim == 3:
                # (seq_len, batch, dim) -> (batch, dim, seq_len)
                x = x.permute(1, 2, 0)
                y = super().forward(x)  # (batch, dim, seq_len)
                return y.permute(2, 0, 1)
            elif x.ndim == 2:
                # (seq_len, dim) -> (dim, seq_len)
                x = x.t()
                y = super().forward(x)
                return y.t()
            else:
                raise NotImplementedError


class StackedConv1DLayers(nn.Module):
    """
    Convenient helper nn.Module to create a stack of conv1D layers.
    """

    def __init__(
        self,
        num_layers: int,
        dim_input: int,
        dim_output: int,
        dim_features: T.Union[T.List[int], int],
        kernel_sizes: T.Union[T.List[int], int],
        strides: T.Union[T.List[int], int] = 1,
        paddings: T.Union[T.List[int], int] = 0,
        dilations: T.Union[T.List[int], int] = 1,
        groups: T.Union[T.List[int], int] = 1,
        padding_modes: T.Union[T.List[str], str] = "zeros",
        nonlinearity: str = "leaky_relu",
        add_norm_layer: bool = True,
        norm_fun=nn.LayerNorm,
        dropout_prob: float = 0.0,
        output_add_nonlinearity: bool = False,
    ):
        """
        Multiple Conv1D layers with normalization and nonlinearity.

        Args:
            num_layers (int):
                total number of layers of the network
            dim_input (int):
                input dimension
            dim_features (int or list of int):
                an integer if all layers share the same dim_feature,
                or list of length num_layer-1 (one for each layer except the last layer).
            kernel_sizes (int or list of int):
                an integer if all layers share the same kernel_size,
                or list of length num_layer (one for each layer).
            strides (int or list of int):
                an integer if all layers share the same stride,
                or list of length num_layer (one for each layer).
            paddings (int or list of int):
                an integer if all layers share the same padding,
                or list of length num_layer (one for each layer).
            dilations (int or list of int):
                an integer if all layers share the same dilation,
                or list of length num_layer (one for each layer).
            groups (int or list of int):
                an integer if all layers share the same group,
                or list of length num_layer (one for each layer).
            padding_modes (str or list of str):
                a str if all layers share the same padding_mode,
                or list of length num_layer (one for each layer)
            nonlinearity (str):
                nonlinearity used in-between layers:
                ``'leaky_relu'``, ``'relu'``, ``'tanh'``, ``'sigmoid'``, ``'silu'`` (torch >= 1.7.0)
            add_norm_layer (bool):
                whether to add normalization layers in between linear layers.
            norm_fun:
                function of the normalization
                (should be a function that takes only the dim_feature,
                pass lambda function if want to change default parameters)
                ex: :code:`lambda x: torch.nn.LayerNorm(x, eps=1e-5, elementwise_affine=False)`
            dropout_prob (float):
                dropout probability of the in-between layer activations
            output_add_nonlinearity (bool):
                Whether to add nonlinearity (norm_layer, and dropout) at the last layer.
        """
        super().__init__()
        self.dim_input = dim_input
        self.num_layers = num_layers
        self.dim_output = dim_output
        self.add_norm_layer = add_norm_layer
        self.linear_bias = not self.add_norm_layer  # if added normalization layer, no need to learn bias
        self.norm_fun = norm_fun
        self.dropout_prob = dropout_prob
        self.output_add_nonlinearity = output_add_nonlinearity

        # check dim_features
        if isinstance(dim_features, int):
            self.dim_features = [dim_features for _ in range(self.num_layers - 1)]
        elif isinstance(dim_features, (list, tuple)):
            if len(dim_features) == 1:
                self.dim_features = [dim_features[0] for _ in range(self.num_layers - 1)]
            else:
                self.dim_features = dim_features
        else:
            raise ValueError("wrong type dim_features")
        if len(self.dim_features) != self.num_layers - 1 and self.num_layers != 1:
            raise ValueError("wrong length of dim_features (should be num_layers-1)")

        # check kernel_sizes
        if isinstance(kernel_sizes, int):
            self.kernel_sizes = [kernel_sizes for _ in range(self.num_layers)]
        elif isinstance(kernel_sizes, (list, tuple)):
            if len(kernel_sizes) == 1:
                self.kernel_sizes = [kernel_sizes[0] for _ in range(self.num_layers)]
            else:
                self.kernel_sizes = kernel_sizes
        else:
            raise ValueError("wrong type kernel_sizes")
        if len(self.kernel_sizes) != self.num_layers:
            raise ValueError("wrong length of kernel_sizes (should be num_layers)")

        # check strides
        if isinstance(strides, int):
            self.strides = [strides for _ in range(self.num_layers)]
        elif isinstance(strides, (list, tuple)):
            if len(strides) == 1:
                self.strides = [strides[0] for _ in range(self.num_layers)]
            else:
                self.strides = strides
        else:
            raise ValueError("wrong type strides")
        if len(self.strides) != self.num_layers:
            raise ValueError("wrong length of strides (should be num_layers)")

        # check paddings
        if isinstance(paddings, int):
            self.paddings = [paddings for _ in range(self.num_layers)]
        elif isinstance(paddings, (list, tuple)):
            if len(paddings) == 1:
                self.paddings = [paddings[0] for _ in range(self.num_layers)]
            else:
                self.paddings = paddings
        else:
            raise ValueError("wrong type paddings")
        if len(self.paddings) != self.num_layers:
            raise ValueError("wrong length of paddings (should be num_layers)")

        # check dilations
        if isinstance(dilations, int):
            self.dilations = [dilations for _ in range(self.num_layers)]
        elif isinstance(dilations, (list, tuple)):
            if len(dilations) == 1:
                self.dilations = [dilations[0] for _ in range(self.num_layers)]
            else:
                self.dilations = dilations
        else:
            raise ValueError("wrong type dilations")
        if len(self.dilations) != self.num_layers:
            raise ValueError("wrong length of dilations (should be num_layers)")

        # check groups
        if isinstance(groups, int):
            self.groups = [groups for _ in range(self.num_layers)]
        elif isinstance(groups, (list, tuple)):
            if len(groups) == 1:
                self.groups = [groups[0] for _ in range(self.num_layers)]
            else:
                self.groups = groups
        else:
            raise ValueError("wrong type dilations")
        if len(self.groups) != self.num_layers:
            raise ValueError("wrong length of groups (should be num_layers)")

        # check padding_modes
        if isinstance(padding_modes, str):
            self.padding_modes = [padding_modes for _ in range(self.num_layers)]
        elif isinstance(padding_modes, (list, tuple)):
            if len(padding_modes) == 1:
                self.padding_modes = [padding_modes[0] for _ in range(self.num_layers)]
            else:
                self.padding_modes = padding_modes
        else:
            raise ValueError("wrong type padding_modes")
        if len(self.padding_modes) != self.num_layers:
            raise ValueError("wrong length of padding_modes (should be num_layers)")

        # check dropout probability
        if self.dropout_prob > 0 and self.num_layers == 1:
            self.dropout_prob = 0
            warnings.warn(
                "dropout option adds dropout after all but the last layer."
                "so non-zero dropout expects "
                "num_layers greater than 1, but got dropout={} and "
                "num_layers={}".format(self.dropout_prob, self.num_layers)
            )

        # check nonlinearity
        self.nonlinearity = nonlinearity
        if self.nonlinearity == "leaky_relu":
            nonlinearity_fun = lambda: nn.LeakyReLU(inplace=False)
        elif self.nonlinearity == "relu":
            nonlinearity_fun = lambda: nn.ReLU(inplace=False)
        elif self.nonlinearity == "tanh":
            nonlinearity_fun = nn.Tanh
        elif self.nonlinearity == "sigmoid":
            nonlinearity_fun = nn.Sigmoid
        elif self.nonlinearity in "silu":
            if torch.__version__ >= "1.7.0":
                nonlinearity_fun = lambda: nn.SiLU(inplace=False)
            else:
                raise NotImplementedError
                warnings.warn("SiLu is only available in pytorch >= 1.7.0. Replace with relu.")
                nonlinearity_fun = lambda: nn.ReLU(inplace=False)
        else:
            raise ValueError("unsupported nonlinearity")

        self.main = nn.ModuleList()
        self.layernorms = nn.ModuleList()

        # batchnorm and instance norm takes (batch, feature, seq_len)
        # layernorm uses (batch, seq_len, feature)
        self.permute_for_norm = False

        dh = self.dim_input
        for layer_idx in range(self.num_layers - 1):
            self.main.append(
                Conv1DLayer(
                    in_channels=dh,
                    out_channels=self.dim_features[layer_idx],
                    kernel_size=self.kernel_sizes[layer_idx],
                    stride=self.strides[layer_idx],
                    padding=self.paddings[layer_idx],
                    dilation=self.dilations[layer_idx],
                    groups=self.groups[layer_idx],
                    padding_mode=self.padding_modes[layer_idx],
                    bias=self.linear_bias,
                    w_init_gain=self.nonlinearity,
                )
            )
            dh = self.dim_features[layer_idx]

            additional_layers = []
            if self.add_norm_layer:
                layer = self.norm_fun(dh)
                additional_layers.append(layer)
                if isinstance(layer, nn.LayerNorm):
                    self.permute_for_norm = True
                elif isinstance(layer, (nn.BatchNorm1d, nn.InstanceNorm1d)):
                    pass
                else:
                    warnings.warn(
                        "Unrecognized normalization function {}, "
                        "make sure you have checked we do not "
                        "need to permute the dimensions.".format(layer)
                    )

            additional_layers.append(nonlinearity_fun())
            if self.dropout_prob > 0:
                additional_layers.append(nn.Dropout(p=self.dropout_prob, inplace=False))

            self.layernorms.append(nn.Sequential(*additional_layers))

        # output linear layer
        self.main.append(
            Conv1DLayer(
                in_channels=dh,
                out_channels=self.dim_output,
                kernel_size=self.kernel_sizes[-1],
                stride=self.strides[-1],
                padding=self.paddings[-1],
                dilation=self.dilations[-1],
                groups=self.groups[-1],
                padding_mode=self.padding_modes[-1],
                bias=self.linear_bias,
                w_init_gain="linear" if not self.output_add_nonlinearity else self.nonlinearity,
            )
        )

        if self.output_add_nonlinearity:
            dh = self.dim_output
            additional_layers = []
            if self.add_norm_layer:
                layer = self.norm_fun(dh)
                additional_layers.append(layer)
                if isinstance(layer, nn.LayerNorm):
                    self.permute_for_norm = True
                elif isinstance(layer, (nn.BatchNorm1d, nn.InstanceNorm1d)):
                    pass
                else:
                    warnings.warn(
                        "Unrecognized normalization function {}, "
                        "make sure you have checked we do not "
                        "need to permute the dimensions.".format(layer)
                    )

            additional_layers.append(nonlinearity_fun())
            if self.dropout_prob > 0:
                additional_layers.append(nn.Dropout(p=self.dropout_prob, inplace=False))
            self.layernorms.append(nn.Sequential(*additional_layers))

        # compute the minimum sequence length
        self.min_seq_len = self.compute_min_seq_len()

    def compute_layer_lengths(self, seq_len_in: int) -> T.List[int]:
        """
        Compute the sequence length of the output of each layers.

        Args:
            seq_len_in (int):
                input sequence length

        Returns:
            list of the length of each layer's output.
        """
        Louts = []
        Lin = seq_len_in
        for layer_idx in range(self.num_layers):
            Lout = math.floor(
                (
                    (
                        Lin
                        + 2 * self.paddings[layer_idx]
                        - self.dilations[layer_idx] * (self.kernel_sizes[layer_idx] - 1)
                        - 1
                    )
                    / float(self.strides[layer_idx])
                )
                + 1
            )  # int (even if seq_len_in is a tensor)
            Louts.append(Lout)
            Lin = Lout
        return Louts

    def compute_layer_valid_lengths(self, seq_len_in: int) -> T.List[int]:
        """
        Compute the sequence length of the output of each layers, with no padding (valid).

        Args:
            seq_len_in (int):
                input sequence length.

        Returns:
            list of the length of each layer's output.
        """
        Louts = []
        Lin = seq_len_in
        for layer_idx in range(self.num_layers):
            Lout = math.floor(
                (
                    (Lin - self.dilations[layer_idx] * (self.kernel_sizes[layer_idx] - 1) - 1)
                    / float(self.strides[layer_idx])
                )
                + 1
            )
            Louts.append(Lout)
            Lin = Lout
        return Louts

    def compute_receptive_fields(self) -> T.List[int]:
        """
        Compute the receptive fields of each layers.

        Returns:
            list of the length of each layer's receptive field.
        """
        rs = [1]  # input
        jumps = [1]
        for layer_idx in range(self.num_layers):
            dilated_kernel_size = (self.kernel_sizes[layer_idx] - 1) * self.dilations[layer_idx] + 1
            r = rs[-1] + (dilated_kernel_size - 1) * jumps[-1]
            jump = jumps[-1] * self.strides[layer_idx]
            rs.append(r)
            jumps.append(jump)

        return rs[1:]

    def compute_layer_start_idxs(self) -> T.List[int]:
        """
        Compute the center coordinate of the first feature of each layer.

        Note that the function assumes symmetric padding: for example, input starts at 0.5.

        Returns:
            list of the center coordinates of each layer's output.
        """
        start_idxs = [0.5]  # input
        jumps = [1]
        for layer_idx in range(self.num_layers):
            dilated_kernel_size = (self.kernel_sizes[layer_idx] - 1) * self.dilations[layer_idx] + 1
            sidx = start_idxs[-1] + ((dilated_kernel_size - 1.0) / 2.0 - self.paddings[layer_idx]) * jumps[-1]
            jump = jumps[-1] * self.strides[layer_idx]
            start_idxs.append(sidx)
            jumps.append(jump)
        return start_idxs[1:]

    def compute_min_seq_len(self) -> int:
        """
        Compute the minimum input sequence length.

        Returns:
            mim input sequence length.
        """
        return self.compute_receptive_fields()[-1]

    def forward(self, x, batch_first=True):
        """
        Args:
            x (batch, dim_input, seq_len) or (seq_len, batch, dim_input):
                input sequence
            batch_first (bool):
                whether the batch dimension is at the first (or the second dimension).

        Returns:
            (batch, dim_output, seq_len_out) or (seq_len, batch, dim_input)
        """

        if not batch_first:
            # convert (seq_len, batch, dim_input) to (batch, dim_input, seq_len)
            x = x.permute(1, 2, 0)

        # check if the input sequence length is enough
        for i in range(len(self.main) - 1):
            x = self.main[i](x)
            if self.permute_for_norm:
                x = x.permute(0, 2, 1)
            x = self.layernorms[i](x)
            if self.permute_for_norm:
                x = x.permute(0, 2, 1)
        x = self.main[-1](x)  # (batch, dim_output, seq_len_out)
        if self.output_add_nonlinearity:
            if self.permute_for_norm:
                x = x.permute(0, 2, 1)
            x = self.layernorms[-1](x)
            if self.permute_for_norm:
                x = x.permute(0, 2, 1)

        if not batch_first:
            # convert (batch, dim, seq_len) to (seq_len, batch, dim)
            x = x.permute(2, 0, 1)

        return x
