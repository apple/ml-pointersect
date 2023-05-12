#
# Copyright (C) 2021 Apple Inc. All rights reserved.
# Author: Rick Chang
#

import unittest
import torch
from torch import nn
from cdslib.core import nn as cdsnn


class TestLSTMCellLayers(unittest.TestCase):

    def _build_model(self, num_layers, dim_input, dim_hidden, bias, append_inputs, lstm_cell_fn):
        net = cdsnn.LSTMCellLayers(
            num_layers=num_layers, dim_input=dim_input, dim_hidden=dim_hidden,
            bias=bias, append_inputs=append_inputs, lstm_cell_fn=lstm_cell_fn,
        )

        # create a random input sequence
        seq_len = 10
        batch_size = 5
        xs = torch.randn(seq_len, batch_size, dim_input)
        h = None

        # compute output
        ys = []
        for t in range(seq_len):
            y, h = net(xs[t], h)
            ys.append(y)

        # check output dimension
        if isinstance(dim_hidden, list):
            dim_output = dim_hidden[-1]
        else:
            dim_output = dim_hidden

        ys = torch.stack(ys, dim=0)
        assert ys.size(0) == seq_len, f'ys.size(0) {ys.size(0)} != seq_len {seq_len}'
        assert ys.size(1) == batch_size, f'ys.size(1) {ys.size(1)} != batch_size {batch_size}'
        assert ys.size(2) == dim_output, f'ys.size(2) {ys.size(2)} != dim_output {dim_output}'

        # check if it can back-propagate
        loss = ys.sum()
        loss.backward()

    def test_lstm(self):
        self._build_model(
            num_layers=3,
            dim_input=10,
            dim_hidden=20,
            bias=True,
            append_inputs=True,
            lstm_cell_fn=nn.LSTMCell,
        )

    def test_rnn(self):
        self._build_model(
            num_layers=1,
            dim_input=10,
            dim_hidden=13,
            bias=True,
            append_inputs=True,
            lstm_cell_fn=nn.RNNCell,
        )

    def test_gru(self):
        self._build_model(
            num_layers=3,
            dim_input=10,
            dim_hidden=[6, 8, 3],
            bias=True,
            append_inputs=True,
            lstm_cell_fn=nn.GRUCell,
        )

    def test_lstm2(self):
        self._build_model(
            num_layers=4,
            dim_input=10,
            dim_hidden=[6, 8, 7, 8],
            bias=True,
            append_inputs=False,
            lstm_cell_fn=nn.LSTMCell,
        )

if __name__ == '__main__':
    unittest.main()
