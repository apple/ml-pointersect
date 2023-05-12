#
# Copyright (C) 2021 Apple Inc. All rights reserved.
# Author: Rick Chang
#

import unittest
import torch
from torch import nn
from cdslib.core import nn as cdsnn


class TestSomeLinearLayers(unittest.TestCase):
    def test_build_model1(self):
        self._build_model(num_layer=1, dim_input=10, dim_output=5, dim_features=16,
                          nonlinearity='leaky_relu', add_norm_layer=True,
                          norm_fun=nn.LayerNorm, dropout_prob=0.)

    def test_build_model2(self):
        self._build_model(num_layer=2, dim_input=10, dim_output=5, dim_features=16,
                          nonlinearity='leaky_relu', add_norm_layer=True,
                          norm_fun=nn.LayerNorm, dropout_prob=0.)

    def test_build_model3(self):
        self._build_model(num_layer=2, dim_input=10, dim_output=5, dim_features=16,
                          nonlinearity='relu', add_norm_layer=True,
                          norm_fun=lambda x: torch.nn.BatchNorm1d(x, eps=1e-1, momentum=0.2, affine=True, track_running_stats=True),
                          dropout_prob=0.)

    def test_build_model4(self):
        self._build_model(num_layer=3, dim_input=10, dim_output=5, dim_features=[16,8],
                          nonlinearity='tanh', add_norm_layer=True,
                          norm_fun=lambda x: torch.nn.InstanceNorm1d(x),
                          dropout_prob=0.5)

    def _build_model(self, num_layer, dim_input, dim_output, dim_features,
                    nonlinearity='leaky_relu', add_norm_layer=True,
                    norm_fun=nn.LayerNorm, dropout_prob=0.):
        net = cdsnn.StackedLinearLayers(
            num_layers=num_layer, dim_input=dim_input, dim_output=dim_output,
            dim_features=dim_features, nonlinearity=nonlinearity,
            add_norm_layer=add_norm_layer, norm_fun=norm_fun,
            dropout_prob=dropout_prob)

        # print(f'num_layer: {num_layer}, dim_input: {dim_input}, dim_output: {dim_output}, '
        #       f'dim_features: {dim_features}, nonlinearity: {nonlinearity}, '
        #       f'add_norm_layer: {add_norm_layer}, norm_fun: {norm_fun}, dropout_prob: {dropout_prob}')
        # print(net)

        seq_len = 10
        batch_size = 5
        # construct input
        x = torch.randn(seq_len, batch_size, dim_input)
        # compute output
        y = net(x)
        assert y.size(0) == seq_len, f'y.size(0) {y.size(0)} != seq_len {seq_len}'
        assert y.size(1) == batch_size, f'y.size(1) {y.size(1)} != batch_size {batch_size}'
        assert y.size(2) == dim_output, f'y.size(2) {y.size(2)} != dim_output {dim_output}'

        # check if it can back-propagate
        loss = y.sum()
        loss.backward()



if __name__ == '__main__':
    unittest.main()
