#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#

import unittest
import cdslib.core.nn as cdsnn
import torch


class TestSubspace(unittest.TestCase):
    def _test(self, in_features, out_features, normalize, orthogonalize, init_norm):
        net = cdsnn.Subspace(
            in_features=in_features,
            out_features=out_features,
            normalize=normalize,
            orthogonalize=orthogonalize,
            init_norm=init_norm,
        )

        A = net(None, 'raw').clone()
        xr = torch.randn(in_features)
        xr2 = torch.randn(in_features)
        xl = torch.randn(out_features)

        # A * xr
        gt = A @ xr
        y, A2 = net(xr, 'A')
        assert torch.allclose(gt, y)
        assert torch.allclose(A, A2)

        # A^T * xl
        gt = A.t() @ xl
        y, A2 = net(xl, 'AT')
        assert torch.allclose(gt, y)
        assert torch.allclose(A, A2)

        # x^l * A^T * A * xr
        gt = (xr2.t() @ A.t()) @ (A @ xr)
        y, A2 = net(xr, 'ATA', xr2)
        assert torch.allclose(gt, y)
        assert torch.allclose(A, A2)

        # xr^T * A^T * A * A^T * A * xr
        gt = ((xr2.t() @ A.t()) @ A) @ (A.t() @ A @ xr)
        y, A2 = net(xr, 'ATAATA', xr2)
        assert torch.allclose(gt, y)
        assert torch.allclose(A, A2)

    def test_1(self):
        self._test(
            in_features=5,
            out_features=3,
            normalize=False,
            orthogonalize=False,
            init_norm=1,
        )

    def test_2(self):
        self._test(
            in_features=25,
            out_features=30,
            normalize=True,
            orthogonalize=False,
            init_norm=1,
        )

    def test_3(self):
        self._test(
            in_features=17,
            out_features=5,
            normalize=True,
            orthogonalize=False,
            init_norm=10,
        )


class TestStackedModulatedSubspace(unittest.TestCase):
    def _test(
            self,
            in_features,
            out_features,
            layer_configs,
            batch_size,
    ):
        # construct
        net = cdsnn.StackedModulatedSubspace(
            in_features=in_features,
            out_features=out_features,
            layer_configs=layer_configs,
        )

        # create random input
        x = torch.randn(batch_size, in_features)
        x.requires_grad = True

        # create random style
        s = [torch.randn(batch_size, layer_config['style_features'],
                         requires_grad=True) for layer_config in layer_configs]

        # forward
        out = net(x, s)

        # backward
        loss = out.sum()
        loss.backward()

        assert len(out.shape) == 2
        assert out.shape[0] == batch_size
        assert out.shape[1] == out_features

    def test1(self):
        in_features = 6
        out_features = 7

        all_out_features = [5, 4, 3, out_features]
        all_style_features = [4, 3, 2, 1]

        layer_configs = []
        for i in range(len(all_out_features)):
            layer_configs.append({
                'out_features': all_out_features[i],
                'style_features': all_style_features[i],
                'dropout': 0.1,
                'nonlinearity': 'relu',
            })

        self._test(
            in_features=in_features,
            out_features=out_features,
            layer_configs=layer_configs,
            batch_size=5,
        )

    def test2(self):
        in_features = 62
        out_features = 79

        all_out_features = [54, 42, 33, out_features]
        all_style_features = [44, 300, 2, 100]

        layer_configs = []
        for i in range(len(all_out_features)):
            layer_configs.append({
                'out_features': all_out_features[i],
                'style_features': all_style_features[i],
                'dropout': 0.1,
                'nonlinearity': 'tanh',
            })

        self._test(
            in_features=in_features,
            out_features=out_features,
            layer_configs=layer_configs,
            batch_size=5,
        )


if __name__ == '__main__':
    unittest.main()
