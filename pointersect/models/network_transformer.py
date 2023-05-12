#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# This file implements the transformer layers (and its variants).
# It modifies pytorch transformer.

import copy
import typing as T
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import ModuleList
from torch.nn import MultiheadAttention


class TransformerEncoderLayer(torch.nn.Module):
    r"""
    TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Note::
        This class is modified from the pytorch implementation, with a switch
        to turn on/off layer normalization.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: T.Callable = F.relu,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            device: torch.device = None,
            dtype: torch.dtype = None,
            use_layer_norm: bool = True,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first,
            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.use_layer_norm = use_layer_norm
        self.norm_first = norm_first
        if self.use_layer_norm:
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = None
            self.norm2 = None
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> T.Tuple[Tensor, T.Union[Tensor, None]]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.use_layer_norm:
            if self.norm_first:
                # x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x_attn, x_attn_weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + x_attn
                x = x + self._ff_block(self.norm2(x))
            else:
                # x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x_attn, x_attn_weights = self._sa_block(x, src_mask, src_key_padding_mask)
                x = self.norm1(x + x_attn)
                x = self.norm2(x + self._ff_block(x))
        else:
            # x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x_attn, x_attn_weights = self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + x_attn
            x = x + self._ff_block(x)

        return x, x_attn_weights

    # self-attention block
    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
    ) -> T.Tuple[Tensor, T.Union[Tensor, None]]:

        # ori
        x, x_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True)

        # # just to see how slow the self attn is
        # if self.self_attn.batch_first:
        #     b, l, dim = x.shape
        # else:
        #     l, b, dim = x.shape
        # x_weights = torch.rand(b, l, l, device=x.device)
        # x_weights = x_weights / x_weights.sum(dim=-1, keepdim=True)
        # # end slowness check

        # return self.dropout1(x), x_weights[..., 0]  # attention to the learned token
        return self.dropout1(x), x_weights[..., 0, :]  # attention to the learned token

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(torch.nn.Module):
    """TransformerEncoder is a stack of N encoder layers
    Directly coppied from torch.nn.TransformerEncoder, but now it will give output weights
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
    ) -> T.Union[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        output_weights_list = []
        for mod in self.layers:
            output, output_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            output_weights_list.append(output_weights)  # ()

        if self.norm is not None:
            output = self.norm(output)

        # Rick: potential improvement, not used during training, saved as list of tensor
        output_weights = torch.stack(output_weights_list, dim=-1)

        return output, output_weights


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
