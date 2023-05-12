#
# Copyright (C) 2020 Apple Inc. All rights reserved.
#

import unittest
import torch
from torch import nn
from cdslib.core import nn as cdsnn


class TestModulatedConv1D(unittest.TestCase):
    def _test_modulated_conv1d(self, in_channels, out_channels, kernel_size, style_dim,
                               demodulate, dilation, type, pad_type, blur_kernel,
                               batch_size, seq_len_in):
        # construct
        net = cdsnn.ModulatedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            dilation=dilation,
            type=type,
            pad_type=pad_type,
            blur_kernel=blur_kernel
        )

        # create random input
        x = torch.randn(batch_size, in_channels, seq_len_in)
        x.requires_grad = True

        if style_dim > 0:
            s = torch.randn(batch_size, style_dim)
        else:
            s = None

        # forward
        out = net(x, s)

        # compute seq_len_out
        seq_len_out = net.compute_output_seq_len(seq_len_in=seq_len_in)

        assert len(out.shape) == 3
        assert out.shape[0] == batch_size
        assert out.shape[1] == out_channels
        assert out.shape[2] == seq_len_out

        if pad_type == 'same':
            if type == 'same':
                if dilation * (kernel_size - 1) % 2 == 0:
                    assert out.shape[2] == seq_len_in
                else:
                    assert out.shape[2] == seq_len_in - 1
            elif type == 'downsample':
                assert out.shape[2] == seq_len_in // 2
            elif type == 'upsample':
                assert out.shape[2] == seq_len_in * 2

    def test_pad_type(self):

        kernel_size = 3
        demodulate = True
        dilation = 1
        batch_size = 5
        type = 'same'

        # same
        pad_type = 'same'
        seq_len_in = 10
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        seq_len_in = 11
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        pad_type = 'valid'
        seq_len_in = 10
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        seq_len_in = 11
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        ## upsample
        kernel_size = 3
        demodulate = True
        dilation = 1
        batch_size = 5
        type = 'upsample'
        pad_type = 'same'
        seq_len_in = 10
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        seq_len_in = 11
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        pad_type = 'valid'
        seq_len_in = 10
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        seq_len_in = 11
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        kernel_size = 4
        demodulate = True
        dilation = 2
        batch_size = 5
        type = 'upsample'
        pad_type = 'same'
        seq_len_in = 10
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        seq_len_in = 11
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        pad_type = 'valid'
        seq_len_in = 10
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        seq_len_in = 11
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=(1, 3, 3, 1),
            batch_size=batch_size, seq_len_in=seq_len_in)

        kernel_size = 11
        demodulate = True
        dilation = 3
        batch_size = 5
        blur_kernel = (1, 2, 1)
        type = 'upsample'
        pad_type = 'same'
        seq_len_in = 10
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=blur_kernel,
            batch_size=batch_size, seq_len_in=seq_len_in)

        seq_len_in = 11
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=blur_kernel,
            batch_size=batch_size, seq_len_in=seq_len_in)

        pad_type = 'valid'
        seq_len_in = 10
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=blur_kernel,
            batch_size=batch_size, seq_len_in=seq_len_in)

        seq_len_in = 11
        self._test_modulated_conv1d(
            in_channels=3, out_channels=4, kernel_size=kernel_size, style_dim=10,
            demodulate=demodulate, dilation=dilation, type=type, pad_type=pad_type, blur_kernel=blur_kernel,
            batch_size=batch_size, seq_len_in=seq_len_in)


class TestStackedFilteredConv1D(unittest.TestCase):
    def _test(self,
              in_channels, layer_configs, blur_kernel,
              batch_size, seq_len_in):
        # construct
        net = cdsnn.StackedFilteredConv1dLayers(
            in_channels=in_channels,
            layer_configs=layer_configs,
            blur_kernel=blur_kernel,
        )

        # create random input
        x = torch.randn(batch_size, in_channels, seq_len_in)
        x.requires_grad = True

        # forward
        out = net(x)

        loss = out.sum()
        loss.backward()

        assert len(out.shape) == 3
        assert out.shape[0] == batch_size
        assert out.shape[1] == layer_configs[-1]['out_channels']
        assert out.shape[2] == net.compute_output_seq_len(seq_len_in=seq_len_in, pad_type=None)[-1]

    def test1(self):
        in_channel = 6
        type = 'same'
        pad_type = 'same'
        norm_layer = 'demodulate'
        layer_configs = []
        for i in range(5):
            layer_configs.append({
                'out_channels': 3,
                'type': type,
                'kernel_size': 3,
                'pad_type': pad_type,
                'dilation': 1,
                'dropout': 0.1,
                'nonlinearity': 'relu',
                'norm_layer': norm_layer,
            })

        self._test(in_channels=in_channel, layer_configs=layer_configs, batch_size=5, seq_len_in=20,
                   blur_kernel=(1, 3, 3, 1))

    def test2(self):
        in_channel = 6
        pad_type = 'same'
        types = ['same', 'upsample', 'downsample', 'upsample', 'downsample']
        out_channels = [5, 7, 9, 11, 13]
        kernel_sizes = [3, 4, 5, 6, 7]
        norm_layers = ['none', 'demodulate', 'layernorm', 'batchnorm', 'instancenorm']
        layer_configs = []
        for i in range(len(kernel_sizes)):
            layer_configs.append({
                'out_channels': out_channels[i],
                'type': types[i],
                'kernel_size': kernel_sizes[i],
                'pad_type': pad_type,
                'dilation': 1,
                'dropout': 0.1,
                'nonlinearity': 'relu',
                'norm_layer': norm_layers[i],
            })

        self._test(in_channels=in_channel, layer_configs=layer_configs, batch_size=5, seq_len_in=20,
                   blur_kernel=(1, 3, 3, 1))

    def test3(self):
        in_channel = 6
        pad_type = 'valid'
        types = ['same', 'upsample', 'downsample', 'upsample', 'downsample']
        out_channels = [5, 7, 9, 11, 13]
        kernel_sizes = [3, 4, 5, 6, 7]
        norm_layers = ['none', 'demodulate', 'layernorm', 'batchnorm', 'instancenorm']
        layer_configs = []
        for i in range(len(kernel_sizes)):
            layer_configs.append({
                'out_channels': out_channels[i],
                'type': types[i],
                'kernel_size': kernel_sizes[i],
                'pad_type': pad_type,
                'dilation': 1,
                'dropout': 0.1,
                'nonlinearity': 'relu',
                'norm_layer': norm_layers[i],
            })

        self._test(in_channels=in_channel, layer_configs=layer_configs, batch_size=5, seq_len_in=20,
                   blur_kernel=(1, 3, 3, 1))



if __name__ == '__main__':
    unittest.main()
    # test = MyTestSomeFilteredConv1D()
    # test.test2()
