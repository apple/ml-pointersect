#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#

import unittest
from cdslib.core.optim import TFOptimizer
import torch
import torch.nn as nn


class TestTFOptimizer(unittest.TestCase):
    def _test(self, use_amp):
        batch, in_features, out_features = 10, 20, 30
        net = nn.Linear(in_features, out_features, bias=True)
        if use_amp:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        net.to(device=device)
        parameters = net.parameters()

        # Create a pytorch optimizer
        # (it will handle everything other than learning rate,
        # e.g., normalization, momentum, etc)
        _optimizer = torch.optim.Adam(
            parameters,  # parameters to optimize
            lr=1e-3,  # this will be overwritten by our optimzier
            betas=(0.9, 0.98),
            eps=1e-9)

        # wrap our learning rate scheduler
        optimizer = TFOptimizer(
            optimizer=_optimizer,
            model_size=30,
            factor=1.0,
            warmup=4000,
            init_step=0)

        if use_amp:
            scaler = torch.cuda.amp.GradScaler()  # only if using amp
        else:
            scaler = None

        for iter in range(10):

            # create a random input
            x = torch.randn(batch, in_features).to(device=device)

            with torch.cuda.amp.autocast(enabled=use_amp):  # only if using amp
                loss = net(x).sum()

            # zero grad
            optimizer.zero_grad()

            # compute gradient
            if use_amp:
                scaler.scale(loss).backward()
                optimizer.unscale(scaler=scaler)  #
            else:
                loss.backward()

            # if gradient clipping
            nn.utils.clip_grad_norm_(parameters, 1e-3)

            # update parameters
            optimizer.step(scaler=scaler)

            # update scaler
            if use_amp:
                scaler.update()

    def test1(self):
        if torch.__version__ >= "1.7.0" and torch.cuda.is_available():
            # AMP is better supported only on cuda
            self._test(use_amp=True)

    def test2(self):
        self._test(use_amp=False)


if __name__ == '__main__':
    unittest.main()
