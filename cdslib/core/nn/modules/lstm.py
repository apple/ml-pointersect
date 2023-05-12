#
# Copyright (C) 2021 Apple Inc. All rights reserved.
# Author: Rick Chang
#
# This file implements various useful building-block rnn layers.

import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import nn_utils
from .linear import StackedLinearLayers


class LSTMCellLayers(nn.Module):
    """
    Convenient helper nn.Module to create a stack of RNNCells.

    Note that nn.LSTMCell is preferred over nn.LSTM when
    for loop needs to be called during training.
    """

    def __init__(
        self,
        num_layers: int,
        dim_input: int,
        dim_hidden: T.Union[T.Sequence[int], int],
        bias: bool = True,
        append_inputs: bool = False,
        lstm_cell_fn: T.Callable = nn.LSTMCell,
        dropout_prob: float = 0,
    ):
        """
        A multi-layer RNN composed of LSTMCells (or the provided lstm_cell_fn).

        Args:
            num_layers:
                Number of RNN layers to create.
            dim_input:
                Feature dimension of the input.
            dim_hidden:
                Dimension of the hidden states. If an integer is provided, it will be used for all layers.
                Otherwise, provide a list of integer, one for each layer.
            bias:
                Whether to learn bias at each layer.
            append_inputs:
                Whether to concatenate all previous layers' inputs to every layer's input.
                Note that the input to i-th layer is the output hidden state of (i-1)-th layer (plus the
                input to all previous layers if append_inputs is True.)
            lstm_cell_fn:
                RNNCell function to use. Can be one of
                :py:func:`torch.nn.RNNCell`,
                :py:func:`torch.nn.LSTMCell`,
                :py:func:`torch.nn.GRUCell`.
            dropout_prob:
                Dropout probability on the hidden states. If non-zero, introduces a Dropout layer on the outputs of
                each LSTM layer except the last layer.

        """
        super().__init__()
        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.lstm_cell_fn = lstm_cell_fn
        self.dropout_prob = dropout_prob
        if isinstance(self.dim_hidden, int):
            self.dim_hidden = [self.dim_hidden for _ in range(self.num_layers)]
        elif len(self.dim_hidden) == 1:
            self.dim_hidden = [self.dim_hidden[0] for _ in range(self.num_layers)]
        self.append_inputs = append_inputs
        self.cell_list = nn.ModuleList()
        din = dim_input
        for layer_idx in range(num_layers):
            cell = self.lstm_cell_fn(din, self.dim_hidden[layer_idx], bias=bias)
            self.cell_list.append(cell)
            if self.append_inputs:
                din = self.dim_hidden[layer_idx] + din
            else:
                din = self.dim_hidden[layer_idx]

    def get_zero_hidden_states(
        self,
        batch_size: int,
        device=torch.device("cpu"),
    ) -> T.List[T.Any]:
        """
        Get a all-zero hidden state that can be used by the module.

        Args:
            batch_size:
                Batch size of the hidden state.
            device:
                torch.device to put the hidden states.

        Returns:
            A list of hidden states, one for a layer.
        """
        if self.lstm_cell_fn == nn.LSTMCell:
            hidden_states = [
                [
                    torch.zeros(batch_size, self.dim_hidden[layer_idx], device=device),
                    torch.zeros(batch_size, self.dim_hidden[layer_idx], device=device),
                ]
                for layer_idx in range(self.num_layers)
            ]
        elif self.lstm_cell_fn in {nn.RNNCell, nn.GRUCell}:
            hidden_states = [
                torch.zeros(batch_size, self.dim_hidden[layer_idx], device=device)
                for layer_idx in range(self.num_layers)
            ]
        else:
            raise NotImplementedError
        return hidden_states

    def forward(self, x, h=None) -> T.Tuple[torch.Tensor, T.List[T.Any]]:
        """
        Args:
            x:
                (batch, dim_input)
            h:
                current/initial hidden state. If None, a all-zero hidden state will be used.

        Returns:
            y:
                (batch, dim_hidden[-1]). output hidden state of the last layer.
            h:
                A list containing the hidden states, one for each layer.  It can be passed to the next forward call.
        """
        batch_size = x.size(0)
        if h is None:
            h = self.get_zero_hidden_states(batch_size, device=x.device)

        new_h: T.List[T.Union[None, torch.Tensor, T.List[torch.Tensor]]] = [None for _ in range(self.num_layers)]
        for layer_idx in range(self.num_layers):
            new_h[layer_idx] = self.cell_list[layer_idx](x, h[layer_idx])
            # prepare input to the next layer
            if layer_idx < self.num_layers - 1:
                if self.append_inputs:
                    if isinstance(new_h[layer_idx], (list, tuple)):
                        x = torch.cat((new_h[layer_idx][0], x), dim=1)
                    else:
                        x = torch.cat((new_h[layer_idx], x), dim=1)
                else:
                    if isinstance(new_h[layer_idx], (list, tuple)):
                        x = new_h[layer_idx][0]
                    else:
                        x = new_h[layer_idx]

                # dropout
                if self.dropout_prob > 0:
                    x = F.dropout(x, self.dropout_prob, self.training)

        if isinstance(new_h[-1], (list, tuple)):
            y = new_h[-1][0]
        else:
            y = new_h[-1]

        return y, new_h


class LSTMLinear(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        num_rnn_layers: int,
        rnn_feature_dim: int,
        num_linear_layers: int,
        dim_linear_features: int,
        linear_add_layer_norm: bool = False,
        output_mode: str = "last_valid",
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        """
        A LSTM with additional linear layers at the top.

        Args:
            dim_input:
                input dimension of the lstm
            dim_output:
                output dimension of the linear layer
            num_rnn_layers: int
                number of lstm layers
            rnn_feature_dim: int
                feature dimension of the lstm. See :py:class:`nn.LSTM`.
            num_linear_layers: int
                number of linear layers
            dim_linear_features:
                list of feature dimensions of the linear layers. length: num_linear_layers-1
            linear_add_layer_norm:
                whether to add layer norm in the linear layers.
            output_mode:
                ["all" | "last_valid" | "last" | "max_valid" | "max" | "avg_valid" | "avg"]
            dropout:
                dropout probability on both lstm and linear layers
        """
        super().__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_rnn_layers = num_rnn_layers
        self.rnn_feature_dim = rnn_feature_dim
        self.num_linear_layers = num_linear_layers
        self.dim_linear_features = dim_linear_features
        self.output_mode = output_mode
        self.linear_add_layer_norm = linear_add_layer_norm
        self.bidirectional = bidirectional
        assert self.output_mode in {
            "all",
            "last_valid",
            "last",
            "max_valid",
            "max",
            "avg_valid",
            "avg",
        }

        self.rnn = nn.LSTM(
            input_size=self.dim_input,
            hidden_size=self.rnn_feature_dim,
            num_layers=self.num_rnn_layers,
            batch_first=False,
            bidirectional=self.bidirectional,
            dropout=dropout,
        )

        if self.num_linear_layers > 0:
            self.linear_layers = StackedLinearLayers(
                num_layers=self.num_linear_layers,
                dim_input=self.rnn_feature_dim if not self.bidirectional else 2 * self.rnn_feature_dim,
                dim_output=self.dim_output,
                dim_features=self.dim_linear_features,
                add_norm_layer=self.linear_add_layer_norm,
                dropout_prob=dropout,
            )
        else:
            self.linear_layers = None
            dim_rnn = self.rnn_feature_dim if not self.bidirectional else 2 * self.rnn_feature_dim
            assert dim_output == dim_rnn

    def get_init_h(self, batch_size=1, device=torch.device("cpu")):
        return nn_utils.get_constant_rnn_hidden_states(
            "lstm",
            self.num_rnn_layers,
            self.rnn_feature_dim,
            batch_size,
            bidirectional=self.bidirectional,
            const=0.0,
            device=device,
        )

    def forward(self, x, h0=None, valid_seq_lens=None):
        """
        Args:
            x: (seq_len, batch_size, dim_input)
            h0:
                hidden state of the lstm. See :py:class:`nn.LSTM`.
        Returns:
            y:
                (seq_len, batch_size, dim) or (batch_size, dim) depending on `output_mode`
            h:
                final hidden state of the lstm. See :py:class:`nn.LSTM`.

        Note when we output_final_only, i.e., y is the shape of (batch_size, dim_output)
            if valid_seq_lens is None:
                return the output at seq_len-1 (final time step)
            else:
                return the output of valid_seq_lens-1
        """
        seq_len = x.size(0)
        if valid_seq_lens is not None:
            if isinstance(valid_seq_lens, torch.Tensor):
                valid_seq_lens = valid_seq_lens.detach().cpu()
            x = nn.utils.rnn.pack_padded_sequence(x, valid_seq_lens, batch_first=False, enforce_sorted=False)

        # run rnn on x
        out, h = self.rnn(x, h0)

        if isinstance(out, torch.nn.utils.rnn.PackedSequence):
            assert valid_seq_lens is not None
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out,
                batch_first=False,
                padding_value=0.0,
                total_length=seq_len,
            )

        if self.output_mode == "last_valid":
            assert valid_seq_lens is not None
            linear_inputs = torch.zeros(out.size(1), self.rnn_feature_dim, device=out.device)
            for b in range(out.size(1)):
                linear_inputs[b] = out[valid_seq_lens[b] - 1, b]
        elif self.output_mode == "all":
            linear_inputs = out
        elif self.output_mode == "last":
            linear_inputs = out[-1]
        elif self.output_mode == "max_valid":
            assert valid_seq_lens is not None
            linear_inputs = torch.zeros(out.size(1), self.rnn_feature_dim, device=out.device)
            for b in range(out.size(1)):
                linear_inputs[b], _ = torch.max(out[: valid_seq_lens[b], b], dim=0)
        elif self.output_mode == "max":
            linear_inputs, _ = torch.max(out, dim=0)
        elif self.output_mode == "avg_valid":
            assert valid_seq_lens is not None
            linear_inputs = torch.zeros(out.size(1), self.rnn_feature_dim, device=out.device)
            for b in range(out.size(1)):
                linear_inputs[b] = torch.mean(out[: valid_seq_lens[b], b], dim=0)
        elif self.output_mode == "avg":
            linear_inputs = torch.mean(out, dim=0)
        else:
            raise NotImplementedError

        if self.num_linear_layers > 0:
            y = self.linear_layers(linear_inputs)  # (batch,dim_output) or (seq_len, batch, dim_output)
        else:
            y = linear_inputs

        return y, h
