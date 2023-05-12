#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
import inspect
import typing as T

import torch.nn as nn

from cdslib.core.nn.modules.conv import StackedConv1DLayers
from cdslib.core.nn.nn_utils import HyperParams


class TacotronContentEncoder(nn.Module):
    """
    - a few conv1D layers
    - Bidirectional LSTM
    """

    def __init__(
        self,
        dim_input: int,
        num_conv_layers: int,
        kernel_size: T.Union[int, T.Sequence[int]],
        dim_conv_layers: T.Union[int, T.Sequence[int]],
        num_lstm_layers: int,
        dim_embed: int,
        p_dropout=0.5,
        nonlinearity="silu",
        add_layer_norm=False,
    ):
        """
        Content encoder: some conv1D -> bidirectional LSTM

        Args:
            dim_input:
                dimension of the input
            num_conv_layers:
                number of conv1D layers
            kernel_size:
                kernel size of the conv layers (int)  odd int
            dim_conv_layers:
                feature dimension of the conv layers (len = num_conv_layers) or int
            num_lstm_layers:
                number of bidirectional lstm layers
            dim_embed:
                final dimension of the content embedding
            p_dropout:
                dropout probability of conv layers
            nonlinearity:
                nonlinearity used in conv layers
            add_layer_norm:
                add layer norm

        """
        super().__init__()
        self.p_dropout = p_dropout

        if isinstance(dim_conv_layers, int):
            dim_conv_layers = [dim_conv_layers]
        if isinstance(dim_conv_layers, T.Sequence) and len(dim_conv_layers) == 1:
            dim_conv_layers = [dim_conv_layers[0]] * num_conv_layers
        assert len(dim_conv_layers) == num_conv_layers
        assert kernel_size % 2 == 1
        assert dim_embed % 2 == 0
        # conv1D layers
        self.conv_layers = StackedConv1DLayers(
            num_layers=num_conv_layers,
            dim_input=dim_input,
            dim_output=dim_conv_layers[-1],
            dim_features=dim_conv_layers[:-1],
            kernel_sizes=kernel_size,
            strides=1,
            paddings=(kernel_size - 1) // 2,
            dilations=1,
            groups=1,
            padding_modes="zeros",
            nonlinearity=nonlinearity,
            add_norm_layer=add_layer_norm,
            dropout_prob=self.p_dropout,
            output_add_nonlinearity=True,
        )

        self.lstm = nn.LSTM(
            input_size=dim_conv_layers[-1],
            hidden_size=dim_embed // 2,
            num_layers=num_lstm_layers,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=True,
        )

    def forward(self, x, valid_lens):
        """
        Args:
            x: (num_char, batch, dim_input)
                symbol embedding
            valid_lens: (batch,)
                valid seq_len of x
        Returns:
             (seq_len, batch, dim_embed), same seq_len as x
        """
        # print(f'x.shape {x.shape}')
        x = self.conv_layers(x, batch_first=False)  # same seq_len as x
        # print(f'after conv x.shape {x.shape}')

        # pack the padded sequence for bidirectional
        x = nn.utils.rnn.pack_padded_sequence(x, valid_lens.detach().cpu(), batch_first=False, enforce_sorted=False)

        outputs, _ = self.lstm(x)  #  (seq_len, batch, dim)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)  # (seq_len, batch, dim)

        return outputs


class ParamTacotronContentEncoder(HyperParams):
    """
    A dictionary that contains the hyper-parameters of :py:class:`TacotronContentEncoder`.
    """

    def __init__(
        self,
        dim_input: int = HyperParams.TBD.INT,
        num_conv_layers: int = 3,
        kernel_size: T.Union[int, T.Sequence[int]] = 5,
        dim_conv_layers: T.Union[int, T.Sequence[int]] = 256,
        num_lstm_layers: int = 1,
        dim_embed: int = 512,
        p_dropout=0.0,
        nonlinearity="silu",
        add_layer_norm=False,
    ):
        """
        See :py:class:`TacotronContentEncoder` for details.
        """
        arg_names = inspect.getfullargspec(self.__init__).args[1:]
        sig = locals()
        arg_dict = {name: sig[name] for name in arg_names}
        super().__init__(**arg_dict)
