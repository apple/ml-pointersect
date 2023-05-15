#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# This file implements the spherical gaussian distribution

import torch
import typing as T
import math
from . import rigid_motion


class SphericalGaussian:
    r"""
    This class implements the spherical Gaussian distribution (i.e., the von Mises-Fisher distribution in 3D).

    Let :math:`w` in :math:`\mathbb{S}^2` and :math:`k \ge 0`, the spherical Gaussian distribution is defined as

    .. math::
        f(w) & = \frac{1}{4 \pi},  if k = 0   \\
             & = \frac{k}{2 \pi (1 - exp(-2 k))} exp(k (u^T w - 1)),  k > 0

    Ref: http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    """

    def __init__(
            self,
            u: torch.Tensor,
            k: T.Union[torch.Tensor, None],
            log_k: torch.Tensor = None,
            normalize_u: bool = False,
    ):
        """
        Args:
            u: (*, 3)
                the mean direction.
                (*) is the dimension of independent components`.
                We are going to use M to represent it for simplicity.
            k: (*,) or (,)
                the concentration on the sphere  (large k -> more concentrated)
            log_k: (*,) or (,)
                log of k, can be None. If given, k will be ignored.
        """
        *m_shape, dim = u.shape
        self.m_shape = m_shape
        self.normalize_u = normalize_u

        assert u.size(-1) == 3
        if self.normalize_u:
            u = torch.nn.functional.normalize(u, dim=-1)

        self.u = u

        if isinstance(k, (float, int)):
            k = torch.ones(*m_shape, device=u.device) * k

        # assert k.shape == m_shape, f'{k.shape} {m_shape}'
        self.k = k
        if log_k is None:
            self.log_k = self.k.log()
        else:
            self.log_k = log_k
            self.k = self.log_k.exp()

        self.log_2pi = math.log(2 * math.pi)

    def compute_neg_log_likelihoods(
            self,
            samples: torch.Tensor,
            normalize: bool = True,
            min_k: float = 1.0e-8,
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood given the samples.
        Args:
            samples: (*m_shape, 3)
                one for each independent component
            normalize:
                whether we want to compute the normalization term as well
        Returns:
            nlls: (*m_shape,)
                the negative log likelihood for sample
        """
        ks = self.k.clamp(min=min_k)
        log_ks = self.log_k.clamp(min=math.log(min_k))
        if normalize:
            nor = self.log_2pi - log_ks + torch.log(1 - torch.exp(-0.5 * ks))  # (*m_shape,) or (,)
        else:
            nor = 0.

        nlls = nor + ks * (1 - (self.u * samples).sum(-1))  # (*m_shape,)

        return nlls

    def sample(self, sample_shape: T.List[int]):
        """
        Sample spherical gaussian distributions.

        Args:
            sample_shape: (*sample_shape).  Can be an empty list.

        Returns:
            samples: (*sample_shape, *m_shape, 3)

        Sampling strategy:
            We are going to first sample assuming u = (0,0,1) for all components
            but with the actual k. Then we will rotate the samples by a rotation
            matrix.
        """

        # sample assuming u = (0,0,1)
        thetas = torch.rand(*sample_shape, *self.m_shape) * 2 * torch.pi  # (S, M)
        vs = torch.stack((torch.cos(thetas), torch.sin(thetas)), dim=-1)  # (S, M, 2)

        etas = torch.rand(*sample_shape, *self.m_shape)  # (S, M)
        ks = self.k.expand(*sample_shape, *self.m_shape)  # (S, M)
        ws = 1 + ks.pow(-1) * torch.log(etas + (1 - etas) * torch.exp(-2 * ks))  # (S, M)
        ws = ws.unsqueeze(-1)  # (S, M, 1)

        samples = torch.cat(
            (
                (1 - ws.pow(2)).sqrt() * vs,
                ws,
            ), dim=-1)  # (S, M, 3)

        # now that we have samples for u = (0,0,1), we will rotate the samples
        # ori_us = torch.zeros(*samples, *self.m_shape, 3)
        # ori_us[..., 2] = 1
        # new_us = self.u.expand(*samples, *self.m_shape, 3)  # (S, M, 3)
        # Rs = rigid_motion.get_min_R(
        #     ori_us,
        #     new_us,
        # )  # (S, M, 3, 3)  new_us = Rs @ ori_us

        # we construct a rotation matrix that rotates (0,0,1) to u.
        # it might not be the geodestic rotation matrix, but it is ok
        # since spherical gaussian is symmetric around the mean direction.
        ys = torch.zeros(*sample_shape, *self.m_shape, 3)
        ys[..., 1] = 1
        Rs = rigid_motion.construct_coord_frame(
            z=self.u.expand(*sample_shape, *self.m_shape, 3),  # (S, M, 3)
            y=ys,  # (S, M, 3)
        )  # (S, M, 3, 3)

        samples = (Rs @ samples.unsqueeze(-1)).squeeze(-1)  # (S, M, 3)

        return samples
