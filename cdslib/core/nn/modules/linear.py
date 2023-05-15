#
# Copyright (C) 2021 Apple Inc. All rights reserved.
# Author: Rick Chang
#
# This file implements various useful building-block linear layers.

import math
import typing as T
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import nn_utils


class LinearLayer(nn.Linear):
    """
    A helper class that wraps around the typical nn.Linear to help initialize it.
    """

    _FLOAT_MODULE = nn.Linear

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_init_gain: str = "linear",
        init_method: str = "xavier_normal",
        lrelu_nslope: float = 0.01,
        kaiming_fan_mode: str = "fan_in",
        bias_init_val: float = 0.0,
        lr_multiplier: float = 1.0,
        fixed_bias: float = None,
    ):
        """
        Create a linear layer whose weights are initialized with a chosen method.
        The usage of the layer is the same as torch.nn.Linear.

        Args:
            in_features:
                Size of the last dimension of the input tensor.
            out_features:
                Size of the last dimension of the output tensor.
            bias:
                Whether to add a learnable bias term, one for each out_features.
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
            bias_init_val:
                Initialization value of the bias.
            lr_multiplier:
                A scalar to be multiplied with the learning rate.
                For example, if lr_multiplier = 0.1, both the weight and the bias will update at 1/10 of the rate.
            fixed_bias (float):
                A fixed bias value to add to the output. If `None`, nothing is added.
        """
        nn.Linear.__init__(self, in_features, out_features, bias)
        nn_utils.init_weight(
            self.weight,
            w_init_gain=w_init_gain,  # leaky_relu, relu, tanh, sigmoid, ...
            init_method=init_method,
            lrelu_nslope=lrelu_nslope,
            kaiming_fan_mode=kaiming_fan_mode,
        )
        # init bias
        if self.bias is not None:
            nn.init.constant_(self.bias, bias_init_val)

        # learning rate multiplier (make the layer update slower or faster)
        self.lr_multiplier = lr_multiplier

        # add constant at the end
        self.fixed_bias = fixed_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input:
                :math:`(*, C_{in})`, input tensor.

        Returns:
            :math:`(*, C_{out})`, output tensor.
        """
        if self.bias is not None:
            out = F.linear(input, self.weight * self.lr_multiplier, self.bias * self.lr_multiplier)
        else:
            out = F.linear(input, self.weight * self.lr_multiplier)

        if self.fixed_bias is not None:
            out = out + self.fixed_bias

        return out


class StackedLinearLayers(nn.Module):
    """
    Convenient helper nn.Module to create a stack of linear layers.
    """

    def __init__(
        self,
        num_layers: int,
        dim_input: int,
        dim_output: int,
        dim_features: T.Union[T.Sequence[int], int],
        nonlinearity: str = "leaky_relu",
        add_norm_layer: bool = False,
        norm_fun: T.Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
        output_add_nonlinearity: bool = False,
    ):
        """
        Convenient helper nn.Module to create a stack of linear layers.

        Args:
            num_layers:
                Total number of linear layers to create.
            dim_input:
                Feature dimension of the input tensor, which is :math:`(*, C_{in})`.
            dim_output:
                Feature dimension of the output tensor, which is :math:`(*, C_{out})`.
            dim_features:
                An integer if all layers share the same feature dimension,
                or a list of num_layer-1 integers, one for each layer except the last layer.
            nonlinearity:
                Nonlinearity used after each linear layer (except the last layer if output_add_nonlinearity is False).
                Choose from: `leaky_relu`, `relu`, `tanh`, `sigmoid`, `silu`, `swish`
                Note that silu (swish) is supported in pytorch version >= 1.7.0.
            add_norm_layer:
                Whether to add normalization layers between linear layers
            norm_fun:
                Callable function used to normalize the output of linear layer (before nonlinearity).
                It should be a function that takes dim_feature as input.
                For example, you can pass `nn.LayerNorm`.
                If you want to control additional functionality like the eps and elementwise_affine of nn.LayerNorm,
                you can pass a lambda function:
                lambda dim: torch.nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
            dropout_prob:
                Dropout probability added after nonlinearity. If 0, no dropout layer is added.
            output_add_nonlinearity:
                Whether to add nonlinearity (norm_layer, and dropout) at the last layer

        Note that the order of the layers is:
            Linear -> normalization (if add_norm_layer) -> nonlinearity -> dropout.

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
        elif isinstance(dim_features, T.Sequence):
            if len(dim_features) == 1:
                self.dim_features = [dim_features[0] for _ in range(self.num_layers - 1)]
            else:
                self.dim_features = dim_features
        else:
            raise ValueError("wrong type dim_features")
        if len(self.dim_features) != self.num_layers - 1 and self.num_layers != 1:
            raise ValueError("wrong length of dim_features (should be num_layers-1)")

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
            nonlinearity_fun = lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False)
        elif self.nonlinearity == "relu":
            nonlinearity_fun = lambda: nn.ReLU(inplace=False)
        elif self.nonlinearity == "tanh":
            nonlinearity_fun = nn.Tanh
        elif self.nonlinearity == "sigmoid":
            nonlinearity_fun = nn.Sigmoid
        elif self.nonlinearity in {"silu", "swish"}:
            nonlinearity_fun = lambda: nn.SiLU(inplace=False)
        else:
            raise ValueError("unsupported nonlinearity")

        self.main = nn.ModuleList()
        self.addons = nn.ModuleList()  # normalization, nonlinearity
        # self.dropouts = nn.ModuleList() if self.dropout_prob > 0 else None  # dropout layers

        # batchnorm and instance norm takes (batch, feature, seq_len) instead of (batch, seq_len, feature)
        self.permute_for_norm = False

        dh = self.dim_input
        for layer_idx in range(self.num_layers - 1):
            self.main.append(
                LinearLayer(
                    in_features=dh,
                    out_features=self.dim_features[layer_idx],
                    bias=self.linear_bias,
                    w_init_gain=self.nonlinearity,
                )
            )
            dh = self.dim_features[layer_idx]

            additional_layers = []
            if self.add_norm_layer:
                layer = self.norm_fun(dh)
                additional_layers.append(layer)
                if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.InstanceNorm1d):
                    self.permute_for_norm = True
                elif isinstance(layer, nn.LayerNorm):
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

            self.addons.append(nn.Sequential(*additional_layers))

        # output linear layer
        self.main.append(
            LinearLayer(
                in_features=dh,
                out_features=self.dim_output,
                bias=True,
                w_init_gain="linear",
            )
        )

        if self.output_add_nonlinearity:
            dh = self.dim_output
            additional_layers = []
            if self.add_norm_layer:
                layer = self.norm_fun(dh)
                additional_layers.append(layer)
                if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.InstanceNorm1d):
                    self.permute_for_norm = True
                elif isinstance(layer, nn.LayerNorm):
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
            self.addons.append(nn.Sequential(*additional_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x:
                :math:`(*, C_{in})`, input tensor.

        Returns:
            :math:`(*, C_{out})`, output tensor.
        """
        for i in range(len(self.main) - 1):
            x = self.main[i](x)
            if self.permute_for_norm and len(x.shape) == 3:
                x = x.permute(0, 2, 1)
            x = self.addons[i](x)
            if self.permute_for_norm and len(x.shape) == 3:
                x = x.permute(0, 2, 1)
        x = self.main[-1](x)
        if self.output_add_nonlinearity:
            if self.permute_for_norm and len(x.shape) == 3:
                x = x.permute(0, 2, 1)
            x = self.addons[-1](x)
            if self.permute_for_norm and len(x.shape) == 3:
                x = x.permute(0, 2, 1)
        return x


class ShiftedLinearLayer(nn.Module):
    r"""
    The layer implements a typical linear layer
    but allows the input x to be shifted by dx.

    .. math::
        & y = W * (x + dx) + b + b0, \text{ where} \\
        & \quad * \text{ is matrix-vector multiplication} \\
        & \quad x \in R^{cin}, \\
        & \quad dx \in R^{cin}, \\
        & \quad W \in R^{cout \times cin},
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        fixed_bias: float = None,
        bias_init_val: float = 0.0,
        lr_multiplier: float = 1.0,
        demodulate=False,
    ):
        r"""
        Args:
            in_features (int):
                input feature dimension
            out_features (int):
                output feature dimension
            bias (bool):
                whether to learn bias :math:`b`
            fixed_bias (float):
                a fixed bias b0 added after Wx + b + b0.
            lr_multiplier (float):
                a factor controls the learning rate of the layer.
            demodulate (bool):
                whether to normalize the row of W.
        """
        super().__init__()

        self.eps = 1e-8
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_bias = fixed_bias
        self.demodulate = demodulate
        self.lr_multiplier = lr_multiplier
        self.scale = 1 / math.sqrt(in_features)

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.ones(out_features) * bias_init_val)
        else:
            self.bias = None

    def __repr__(self):
        return (
            f"ShiftedLinearLayer({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, fixed_bias={self.fixed_bias})"
        )

    def forward(self, x: torch.Tensor, dx: torch.Tensor = None):
        r"""
        Args:
            x (\*, in_features):
                input tensor
            dx (\*, in_features):
                same shape as x, or can be broadcast.
                if None, dx = 0.

        Returns:
            y (* out_features):
                :math:`y = W (x + dx) + b + b0`

        """
        weight = self.scale * self.weight  # (cout, cin)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=1, keepdim=True) + self.eps)  # (cout, 1)
            weight = weight * demod

        y = F.linear(
            input=x + dx if dx is not None else x,
            weight=self.lr_multiplier * weight,
            bias=self.lr_multiplier * self.bias if self.bias is not None else None,
        )

        if self.fixed_bias is not None:
            y = y + self.fixed_bias

        return y
