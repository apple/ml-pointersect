#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#

import torch
import typing as T
import numpy as np


class NestedDropout(torch.nn.Module):
    """
    Nested dropout layer proposed by Rippel et al. [2014].

    Compared to typical dropout, which independently masks variables,
    the nested dropout masks i+1 to the end, if i is chosen.

    Ref: https://arxiv.org/abs/1402.0915
    """

    def __init__(self, probs: T.Sequence[float]):
        """
        Construct nested dropout layer, which drops the last dimension.
        Note that it creates a mask of shape (B, C), so if the input x
        has a dimension larger than 2, the first dimensions will share the
        same mask, and different instances in the batch uses different masks.

        Args:
            probs:
                the probablity of the index to be chosen. If None, uniform probability.

        Input:
            x: (*, B, C)

        Output:
            y: (*, B, C)
        """
        super().__init__()
        self.probs = probs
        self.rng = np.random.default_rng()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x:
                `(*, B, C)`

        Returns:
            y:
                `(*, B, C)`
        """
        if not self.training:
            return x  # no masking during inference

        ori_x_shape = x.shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)  # (1, C)
        batch_size = x.size(-2)
        dim = x.size(-1)
        x = x.reshape(-1, batch_size, dim)

        # create mask
        chosen_idxs = self.rng.choice(
            np.arange(dim),
            size=[batch_size],
            replace=True,
            p=self.probs,
        )  # (batch,) int

        idxs = torch.arange(0, dim, device=x.device)  # (dim,)
        mask = idxs <= (torch.from_numpy(chosen_idxs).unsqueeze(1).to(device=idxs.device))  # (batch, dim)

        y = x * mask.unsqueeze(0)  # (-1, batch, dim)
        return y.reshape(*ori_x_shape)
