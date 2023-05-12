#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
# Author: Rick Chang
# This file implements various classes that operates on the
# linear subspace of the weights of a linear layer.

import math
import typing as T
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import ShiftedLinearLayer


class Subspace(nn.Module):
    """
    The layer implements a matrix A and its transpose A^T.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        normalize: bool = False,
        orthogonalize: bool = False,
        init_norm: float = 1.0,
    ):
        """
        Args:
            in_features (int):
                input feature dimension (number of columns)
            out_features (int):
                output feature dimension (number of rows)
            normalize (bool):
                whether to normalize the basis vectors (columns of weights).
            orthogonalize (bool):
                whether to make orthogonal the basis vectors
                (columns of weights) by qr decomposition.
                Note that it is recommended to turn off (set to False).
            init_norm (float):
                the l2 norm of the columns

        Notes:
            If orthogonalize is `True`, the weight will orthogonalized by QR decomposition,
            and thus the shape will change (min of in_feature and out_feature).
        """
        super().__init__()

        self.eps = 1e-8
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        self.orthogonalize = orthogonalize
        self.scale = 1 / math.sqrt(in_features)

        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # initialize weights
        nn.init.orthogonal_(self.weight, gain=init_norm)

    def __repr__(self):
        return (
            f"Subspace({self.in_features}, {self.out_features}, "
            f"normalize={self.normalize}, orthogonalize={self.orthogonalize})"
        )

    def forward(
        self,
        input: torch.Tensor,
        mode: str = "A",
        input_left: torch.Tensor = None,
    ):
        r"""
        Args:
            input:
                - For A: `(*, in_features)`
                - For AT: `(*, out_features)`
                - For ATA: `(*, in_features)`
            mode (str):
                - ``'A'``: :math:`A(x) = y = A * x`
                - ``'AT'``: :math:`AT(y) = x = A^T * y`
                - ``'ATA'``: :math:`ATA(x_r, x_l) = x_l^T * A^T * A * x_r`
                - ``'raw'``: returns the weight
            input_left `(*, in_features)`:
                only used when the mode is ``'ATA'``: `(*, in_features)`

        Returns:
            - For A: a tuple [`(*, out_features)` and weight `(out_features, in_features)`]
            - For AT: a tuple [`(*, in_features)` and weight `(out_features, in_features)`]
            - For ATA: a tuple [if input_left is `None`, `(*, in_features)` else `(*,)`, and weight `(out_features, in_features)`]
            - For raw: weight `(out_features, in_features)`

        """

        if mode == "A":
            return self.A(input)
        elif mode == "AT":
            return self.AT(input)
        elif mode == "ATA":
            return self.ATA(xr=input, xl=input_left)
        elif mode == "ATAATA":
            return self.ATAATA(xr=input, xl=input_left)
        elif mode == "raw":
            return self._get_weight()
        else:
            raise NotImplementedError

    def _get_weight(self):
        weight = self.scale * self.weight  # (cout, cin)
        if self.orthogonalize:
            # qr decomposition also normalize the bases
            q, r = torch.qr(weight)
            d = torch.diag(r, 0)
            ph = d.sign()  # +1 or -1   # loss gradient to r

            if self.normalize:
                weight = q * ph  # (cout, cin)
            else:
                ori_norm = weight.norm(p=2, dim=0, keepdim=True)  # (1, cin)
                weight = q * ph  # (cout, cin)
                weight = weight * ori_norm

        elif self.normalize:
            demod = torch.rsqrt(weight.pow(2).sum(dim=0, keepdim=True) + self.eps)  # (1, cin)
            weight = weight * demod

        return weight

    def A(self, x):
        weight = self._get_weight()
        y = F.linear(input=x, weight=weight, bias=None)
        return y, weight

    def AT(self, y):
        weight = self._get_weight()
        x = F.linear(input=y, weight=weight.t(), bias=None)
        return x, weight

    def ATA(self, xr=None, xl=None):
        # xl^T * A^T * A * xr
        assert xr is not None or xl is not None
        weight = self._get_weight()
        if xr is not None:
            Axr = F.linear(input=xr, weight=weight, bias=None)  # (*, cout)
            ATAxr = F.linear(input=Axr, weight=weight.t(), bias=None)  # (*, cin)
            if xl is not None:
                xlATAxr = torch.einsum("...i,...i->...", xl, ATAxr)  # (*,)
                return xlATAxr, weight
            else:
                return ATAxr, weight
        else:
            assert xl is not None
            Axl = F.linear(input=xl, weight=weight, bias=None)  # (*, cout)
            ATAxl = F.linear(input=Axl, weight=weight.t(), bias=None)  # (*, cin)
            return ATAxl, weight

    def ATAATA(self, xr=None, xl=None):
        # xl^T * A^T * A * A^T * A * xr
        assert xr is not None or xl is not None
        weight = self._get_weight()
        if xr is not None:
            # A
            y = F.linear(input=xr, weight=weight, bias=None)  # (*, cout)
            # AT
            y = F.linear(input=y, weight=weight.t(), bias=None)  # (*, cin)
            # A
            y = F.linear(input=y, weight=weight, bias=None)  # (*, cout)
            # AT
            y = F.linear(input=y, weight=weight.t(), bias=None)  # (*, cin)
            if xl is not None:
                y = torch.einsum("...i,...i->...", xl, y)  # (*,)
                return y, weight
            else:
                return y, weight
        else:
            assert xl is not None
            # compute (xl^T * A^T * A * A^T * A)^T = At * A * At * A * xl
            # A
            y = F.linear(input=xl, weight=weight, bias=None)  # (*, cout)
            # AT
            y = F.linear(input=y, weight=weight.t(), bias=None)  # (*, cin)
            # A
            y = F.linear(input=y, weight=weight, bias=None)  # (*, cout)
            # AT
            y = F.linear(input=y, weight=weight.t(), bias=None)  # (*, cin)
            return y, weight


class ModulatedSubspace(nn.Module):
    r"""
    The layer implements modulated subspace layer.

    .. math::
        &  y = W * (x + V * s + x0 + x1) + b + b0, \text{ where} \\
        &\quad  x \in R^{cin}, \\
        &\quad  s \in R^{k}, \\
        &\quad  dx \in R^{cin}, \\
        &\quad  W \in R^{cout \times cin}, \\
        &\quad  V \in R^{cin \times k}, \\
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        style_features: int,
        bias: bool = True,
        fixed_bias: float = None,
        input_bias: bool = True,
        fixed_input_bias: float = None,
        bias_init_val: float = 0.0,
        input_bias_init_val: float = 0.0,
        lr_multiplier: float = 1.0,
        demodulate: bool = False,
        normalize_basis: bool = True,
        orthogonalize_basis: bool = False,
    ):
        r"""
        Args:
            in_features (int):
                input feature dimension
            out_features (int):
                output feature dimension
            style_features (int):
                style feature dimension, i.e., dimension of `s`.
            bias (bool):
                whether to learn bias :math:`b`
            fixed_bias (float):
                a fixed bias b0 added after Wx + b + b0.
            input_bias (bool):
                whether to learn x0
            fixed_input_bias (float):
                a fixed bias x1 added after x0
            lr_multiplier (float):
                a factor controls the learning rate of the layer.
            demodulate (bool):
                whether to normalize the row of W.
            normalize_basis (bool):
                whether to normalize the basis to have unit l2 norm
            orthogonalize_basis (bool):
                whether to orthogonalize the basis
        """
        super().__init__()

        self.eps = 1e-8
        self.in_features = in_features
        self.out_features = out_features
        self.style_features = style_features
        self.fixed_bias = fixed_bias
        self.input_bias = input_bias
        self.fixed_input_bias = fixed_input_bias
        self.demodulate = demodulate
        self.lr_multiplier = lr_multiplier
        self.normalize_basis = normalize_basis
        self.orthogonalize_basis = orthogonalize_basis

        # W
        self.linear = ShiftedLinearLayer(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            fixed_bias=fixed_bias,
            bias_init_val=bias_init_val,
            lr_multiplier=lr_multiplier,
            demodulate=demodulate,
        )
        # V
        self.subspace = Subspace(
            in_features=style_features,
            out_features=in_features,
            normalize=normalize_basis,
            orthogonalize=orthogonalize_basis,
        )
        # input bias
        if input_bias:
            self.input_bias = nn.Parameter(torch.ones(in_features) * input_bias_init_val)
        else:
            self.input_bias = None

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        r"""
        Args:
            x `(*, in_features)`:
                input tensor
            s `(*, style_features)`:
                style tensor or those can be broadcast
        Returns:
            y `(*, out_features)`:

        """
        dx, _ = self.subspace(input=s, mode="A", input_left=None)  # (*, cin) or (batch, cin)

        # add input bias
        ndim = x.ndim
        if self.input_bias is not None:
            dx = dx + self.input_bias.view(*([1] * (ndim - 1) + [self.in_features]))
        if self.fixed_input_bias is not None:
            dx = dx + self.fixed_input_bias

        y = self.linear(x, dx)
        return y

    def subspace_fun(self, xl=None, xr=None, mode="A"):
        """
        This function exposes the underlying subspace operators.
        See :py:class:`Subspace` for detailed documentation.
        """
        return self.subspace(input=xr, input_left=xl, mode=mode)


class StackedModulatedSubspace(nn.Module):
    """
    Convenient helper nn.Module to create a stack of modulated subspace layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_configs: T.Sequence[T.Dict[str, T.Any]],
    ):
        r"""Construct multiple stacked modulated subspace layers
        sandwiched between nonlinearity.

        Args:
            in_features (int):
                number of input channels
            out_features (int):
                number of output channels in the output system
            layer_configs (list of T.Dict[str, T.Any]):
                a list of dict (one for each layer) containing the parameters
                to :py:class:`ModulatedSubspace`
                and parameters for nonlinearity, and dropout:

                - out_features (int, required):
                    output feature dimension
                - style_features (int, required):
                    style feature dimension
                - bias (bool):
                    whether to learn bias :math:`b`. Default: `True`.
                - fixed_bias (float):
                    a fixed bias b0 added after :math:`Wx + b + b0`. Default: `0`.
                - input_bias (bool):
                    whether to learn x0. Default: `True`.
                - fixed_input_bias (float):
                    a fixed bias x1 added after x0. Default: `0`.
                - lr_multiplier (float):
                    a factor controls the learning rate of the layer. Default: `1`.
                - demodulate (bool):
                    whether to normalize the row of W. Default: `True`.
                - normalize_basis (bool):
                    whether to normalize the basis to have unit l2 norm. Default: `True`.
                - orthogonalize_basis (bool):
                    whether to orthogonalize the basis. Default: `False`.
                    (recommended: False)
                - dropout (float):
                    the dropout rate. Default: `0`.
                - nonlinearity (str):
                    ``'none'``, ``'leaky_relu'``, ``'relu'``, ``'tanh'``,
                    ``'sigmoid'``, ``'silu'``. Default: ``'relu'``.

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_configs = layer_configs
        self.num_layers = len(self.layer_configs)
        assert self.out_features == self.layer_configs[-1]["out_features"]

        # adding default layer configs
        for layer_config in self.layer_configs:
            assert "out_features" in layer_config
            assert "style_features" in layer_config

            if "bias" not in layer_config:
                layer_config["bias"] = True
            if "fixed_bias" not in layer_config:
                layer_config["fixed_bias"] = None
            if "input_bias" not in layer_config:
                layer_config["input_bias"] = True
            if "fixed_input_bias" not in layer_config:
                layer_config["fixed_input_bias"] = None
            if "bias_init_val" not in layer_config:
                layer_config["bias_init_val"] = 0
            if "input_bias_init_val" not in layer_config:
                layer_config["input_bias_init_val"] = 0
            if "lr_multiplier" not in layer_config:
                layer_config["lr_multiplier"] = 1
            if "demodulate" not in layer_config:
                layer_config["demodulate"] = True
            if "normalize_basis" not in layer_config:
                layer_config["normalize_basis"] = True
            if "orthogonalize_basis" not in layer_config:
                layer_config["orthogonalize_basis"] = False
            if "dropout" not in layer_config:
                layer_config["dropout"] = 0
            if "nonlinearity" not in layer_config:
                layer_config["nonlinearity"] = "relu"

        # construct the layers
        self.main = nn.ModuleList()
        self.addons = nn.ModuleList()

        dh = self.in_features
        for layer_idx in range(self.num_layers):
            layer_config = self.layer_configs[layer_idx]
            # conv layers
            self.main.append(
                ModulatedSubspace(
                    in_features=dh,
                    out_features=layer_config["out_features"],
                    style_features=layer_config["style_features"],
                    bias=layer_config["bias"],
                    fixed_bias=layer_config["fixed_bias"],
                    input_bias=layer_config["input_bias"],
                    fixed_input_bias=layer_config["fixed_input_bias"],
                    bias_init_val=layer_config["bias_init_val"],
                    input_bias_init_val=layer_config["input_bias_init_val"],
                    lr_multiplier=layer_config["lr_multiplier"],
                    demodulate=layer_config["demodulate"],
                    normalize_basis=layer_config["normalize_basis"],
                    orthogonalize_basis=layer_config["orthogonalize_basis"],
                )
            )
            dh = layer_config["out_features"]

            # addons
            additional_layers = []
            # nonlinear
            if layer_config["nonlinearity"] == "none":
                layer = None
            elif layer_config["nonlinearity"] == "leaky_relu":
                layer = nn.LeakyReLU(inplace=False, negative_slope=0.2)
            elif layer_config["nonlinearity"] == "relu":
                layer = nn.ReLU(inplace=False)
            elif layer_config["nonlinearity"] == "tanh":
                layer = nn.Tanh()
            elif layer_config["nonlinearity"] == "sigmoid":
                layer = nn.Sigmoid()
            elif layer_config["nonlinearity"] == "silu":
                if torch.__version__ >= "1.7.0":
                    layer = nn.SiLU(inplace=False)
                else:
                    raise NotImplementedError
                    warnings.warn("SiLu is only available in pytorch >= 1.7.0. Replace with relu.")
                    layer = nn.ReLU(inplace=False)
            else:
                raise ValueError("unsupported nonlinearity")
            if layer is not None:
                additional_layers.append(layer)

            if layer_config["dropout"] > 0:
                additional_layers.append(nn.Dropout(p=layer_config["dropout"], inplace=False))

            if len(additional_layers) > 0:
                self.addons.append(nn.Sequential(*additional_layers))
            else:
                self.addons.append(None)

    def forward(
        self,
        x: torch.Tensor,
        styles: T.Union[T.Sequence[torch.Tensor], torch.Tensor],
        use_subspace_fun: bool = False,
        xls: T.Union[T.Sequence[torch.Tensor], torch.Tensor] = None,
        xrs: T.Union[T.Sequence[torch.Tensor], torch.Tensor] = None,
        mode: str = "A",
    ):
        r"""
        Args:
            x `(*, in_features)`:
                input tensor
            styles `(*, style_features)` or list of tensors `(*, style_features)` broadcastable to x:
                style tensor

        Returns:
            `(*, out_features)`
        """

        if use_subspace_fun:
            return self.subspace_fun(xls=xls, xrs=xrs, mode=mode)

        else:
            if isinstance(styles, torch.Tensor):
                styles = [styles] * self.num_layers
            assert len(styles) == self.num_layers
            for i in range(self.num_layers):
                assert styles[i].ndim == x.ndim

            for layer_idx in range(self.num_layers):
                x = self.main[layer_idx](x, styles[layer_idx])
                x = self.addons[layer_idx](x)

            return x

    def subspace_fun(
        self,
        xls: T.Union[T.Sequence[torch.Tensor], torch.Tensor] = None,
        xrs: T.Union[T.Sequence[torch.Tensor], torch.Tensor] = None,
        mode: str = "A",
    ):
        """
        Args:
            xls (*, style_dim) or list of (*, style_dim) or None:
                input tensor on the right hand side
            xrs (*, style_dim) or list of (*, style_dim) or None:
                input tensor on the left hand side
            mode (str):
                mode of the transform
        Returns:
            list of results, one for each layer
        """

        if isinstance(xls, torch.Tensor) or xls is None:
            xls = [xls] * self.num_layers
        assert len(xls) == self.num_layers

        if isinstance(xrs, torch.Tensor) or xrs is None:
            xrs = [xrs] * self.num_layers
        assert len(xrs) == self.num_layers

        out_list = []
        for layer_idx in range(self.num_layers):
            out = self.main[layer_idx].subspace_fun(xl=xls[layer_idx], xr=xrs[layer_idx], mode=mode)
            out_list.append(out)

        return out_list
