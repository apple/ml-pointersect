#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#


import unittest
import torch
from torch import nn
from cdslib.core import nn as cdsnn


class TestStackedConv1DLayers(unittest.TestCase):

    def test_build_model1(self):
        self.build_model(num_layer=1, dim_input=10, dim_output=5, dim_features=16,
                         kernel_sizes=3, strides=1, paddings=0, dilations=1,
                         nonlinearity='leaky_relu', add_norm_layer=True,
                         norm_fun=nn.LayerNorm, dropout_prob=0.)

    def test_build_model2(self):
        self.build_model(num_layer=4, dim_input=10, dim_output=5, dim_features=16,
                         kernel_sizes=3, strides=2, paddings=0, dilations=1,
                         nonlinearity='leaky_relu', add_norm_layer=True,
                         norm_fun=nn.LayerNorm, dropout_prob=0.)

    def test_build_model3(self):
        self.build_model(num_layer=4, dim_input=10, dim_output=5, dim_features=16,
                         kernel_sizes=3, strides=2, paddings=0, dilations=2,
                         nonlinearity='leaky_relu', add_norm_layer=True,
                         norm_fun=nn.LayerNorm, dropout_prob=0.)

    def test_build_model4(self):
        self.build_model(num_layer=4, dim_input=10, dim_output=5, dim_features=16,
                         kernel_sizes=3, strides=2, paddings=1, dilations=2,
                         nonlinearity='leaky_relu', add_norm_layer=True,
                         norm_fun=nn.LayerNorm, dropout_prob=0.)

    def build_model(self, num_layer, dim_input, dim_output, dim_features,
                    kernel_sizes, strides, paddings, dilations,
                    padding_modes='zeros',
                    nonlinearity='leaky_relu', add_norm_layer=True,
                    norm_fun=nn.LayerNorm, dropout_prob=0.):
        net = cdsnn.StackedConv1DLayers(
            num_layers=num_layer, dim_input=dim_input, dim_output=dim_output,
            dim_features=dim_features, kernel_sizes=kernel_sizes, strides=strides,
            paddings=paddings, dilations=dilations,
            nonlinearity=nonlinearity, padding_modes=padding_modes,
            add_norm_layer=add_norm_layer, norm_fun=norm_fun,
            dropout_prob=dropout_prob)

        # print(f'num_layer: {num_layer}, dim_input: {dim_input}, dim_output: {dim_output}, '
        #       f'dim_features: {dim_features}, '
        #       f'kernel_sizes: {kernel_sizes}, '
        #       f'strides: {strides}, '
        #       f'paddings: {paddings}, '
        #       f'dilations: {dilations}, '
        #       f'padding_modes: {padding_modes}, '
        #       f'nonlinearity: {nonlinearity}, '
        #       f'add_norm_layer: {add_norm_layer}, norm_fun: {norm_fun}, dropout_prob: {dropout_prob}')
        # print(net)

        seq_len = 100
        batch_size = 2
        # construct input
        x = torch.randn(seq_len, batch_size, dim_input)
        # compute output
        y = net(x, batch_first=False)

        # compute seq_len_out
        # seq_len_out = net.compute_layer_lengths(seq_len)
        seq_len_out = net.compute_layer_lengths(torch.ones(1, dtype=torch.long) * seq_len)
        # print(f'seq len outs: {seq_len_out}')

        # compute receptive fields
        r_fields = net.compute_receptive_fields()
        # print(f'receptive fields: {r_fields}')

        assert y.size(0) == seq_len_out[-1], f'y.size(0) {y.size(0)} != seq_len {seq_len_out[-1]}'
        assert y.size(1) == batch_size, f'y.size(1) {y.size(1)} != batch_size {batch_size}'
        assert y.size(2) == dim_output, f'y.size(2) {y.size(2)} != dim_output {dim_output}'

        # check if it can back-propagate
        loss = y.sum()
        loss.backward()


if __name__ == '__main__':
    unittest.main()
