#
# Copyright (C) 2021 Apple Inc. All rights reserved.
# Author: Rick Chang
# This file implements various attention layers.

import typing as T

import torch
from torch import nn

from .linear import StackedLinearLayers


class GaussianSlidingWindows(nn.Module):
    """
    Attention with mixture of gaussian windows.

    Ref: https://arxiv.org/abs/1308.0850 page 26
    """

    def __init__(
            self,
            num_mixtures: int,
            dim_input: int,
            num_layers: int,
            dim_features: T.Union[int, T.Sequence[int]],
            pos_fun: str = "softplus",
            normalized_weights=False,
    ):
        """
        Given input, the layer computes a mixture of 1D gaussians as the attention weights.

        Args:
            num_mixtures:
                how many gaussian to use
            dim_input:
                dimension of the input (ex: hidden state of bottom lstm)
            num_layers:
                number of layers
            dim_features:
                feature dimension of the linear layers.
                Can be an integer so every layer shares the same dimension
                or a list of (num_layers-1) integers, one for each layer except the last layer.
            pos_fun:
                name of the function to learn positive variance.
                Choices: [`exp` | `softplus`]
            normalized_weights:
                whether to add a softmax to gaussian weights
        """

        super().__init__()
        self.num_mixtures = num_mixtures
        self.dim_input = dim_input
        self.dim_features = dim_features
        self.num_layers = num_layers
        self.dim_linear_output = self.num_mixtures * 3  # mean, std, and weight for each gaussian
        if pos_fun == "exp":
            self.make_pos_fun = lambda x: torch.exp(x)
        elif pos_fun == "softplus":
            self.softplus = nn.Softplus()
            self.make_pos_fun = lambda x: self.softplus(x)
        else:
            raise ValueError("wrong")

        self.normalized_weights = normalized_weights
        if self.normalized_weights:
            self.log_softmax = nn.LogSoftmax(dim=2)

        self.main = StackedLinearLayers(
            num_layers=self.num_layers,
            dim_input=self.dim_input,
            dim_output=self.dim_linear_output,
            dim_features=self.dim_features,
            nonlinearity="leaky_relu",
            add_norm_layer=False,
            norm_fun=nn.LayerNorm,
            dropout_prob=0.0,
        )

    def get_init_means(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Return the initial mean of each gaussian

        Args:
            batch_size:
                Batch size
            device:
                torch.device to place the init means

        Returns:
            (batch_size, num_mixtures)
        """
        return torch.zeros(batch_size, self.num_mixtures, device=device)

    def forward(
            self,
            inputs: torch.Tensor,
            context_vectors: torch.Tensor,
            init_means: torch.Tensor = None,
            extra_chars: int = 0,
    ):
        """
        Args:
            inputs: (seq_len, batch_size, dim_input)
                input used to compute attention
            context_vectors: (num_chars, batch_size, dim_context)
                can be onehot encoding or embeddings
            init_means: (batch_size, num_mixtures) or (1, batch_size, num_mixtures)
                current means of the gaussians (None: all zeros)
            extra_chars:
                how many extra char_idx to calculate

        Returns:
            attn_contexts (seq_len, batch_size, dim_context)
                one for each time step
            attn_weights (seq_len, batch_size, total_char)
                one for each time step, total_char = num_char + extra_chars
            means (seq_len, batch_size, num_mixtures)
                mean of each gaussian window
            vars (seq_len, batch_size, num_mixtures)
                variance of each gaussian window
            weights (seq_len, batch_size, num_mixtures)
                weights of each gaussian window
        """
        seq_len = inputs.size(0)
        batch_size = inputs.size(1)
        num_char = context_vectors.size(0)
        dim_context = context_vectors.size(2)

        total_char = num_char + extra_chars

        attn_weight_dict = self.compute_attn_weights(
            inputs=inputs,
            total_char=total_char,
            init_means=init_means,
        )

        attn_contexts = self.compute_attn_contexts(
            attn_weights=attn_weight_dict['attn_weights'],
            context_vectors=context_vectors,
        )

        return (
            attn_contexts,
            attn_weight_dict['attn_weights'],
            attn_weight_dict['means'],
            attn_weight_dict['vars'],
            attn_weight_dict['weights'],
        )

        # if init_means is None:
        #     init_means = self.get_init_means(batch_size, device=inputs.device)  # (batch_size,num_mixtures)
        #
        # # get the weight, mean, and std for each time step
        # outs = self.main(inputs).unsqueeze(3)  # (seq_len, batch_size, num_mixtures*3,1)
        # # make positive
        # outs = self.make_pos_fun(outs)
        # delta_means, vars, weights = torch.chunk(outs, chunks=3, dim=2)  # each is (seq_len, batch_size, num_mixtures,1)
        #
        # # small tweaks
        # delta_means = delta_means / 25.0
        # vars = torch.clamp(vars, min=0.01, max=100.0)
        # # end small tweaks
        #
        # means = delta_means.cumsum(dim=0) + init_means.view(
        #     1, batch_size, self.num_mixtures, 1
        # )  # (seq_len, batch_size, num_mixtures, 1)
        #
        # # compute attn_weights
        # cidxs = torch.arange(total_char, dtype=torch.float, device=inputs.device).view(
        #     1, 1, 1, total_char
        # )  # (1,1,1,total_char)
        # dists = -1.0 * (means - cidxs).pow(2) * vars  # (seq_len, batch_size, num_mixtures, total_char)
        # # attn_weights = torch.sum(weights * torch.exp_(dists), dim=2) # seq_len, batch_size, total_char
        #
        # if self.normalized_weights:
        #     log_weights = self.log_softmax(weights)
        #     attn_weights = torch.logsumexp(log_weights + dists, dim=2).exp()  # (seq_len, batch_size, total_char)
        # else:
        #     log_weights = None
        #     attn_weights = torch.logsumexp(weights.log() + dists, dim=2).exp()  # (seq_len, batch_size, total_char)
        #
        # # # DEBUG
        # if torch.isnan(attn_weights).any():
        #     print(f"inputs: nan: {torch.isnan(inputs).any()}, inf: {torch.isinf(inputs).any()}")
        #     print(f"outs: nan: {torch.isnan(outs).any()}, inf: {torch.isinf(outs).any()}")
        #     print(f"cidxs: nan: {torch.isnan(cidxs).any()}, inf: {torch.isinf(cidxs).any()}")
        #     print(f"dists: nan: {torch.isnan(dists).any()}, inf: {torch.isinf(dists).any()}")
        #     if log_weights is not None:
        #         print(f"log_weights: nan: {torch.isnan(log_weights).any()}, inf: {torch.isinf(log_weights).any()}")
        #     raise RuntimeError("attn_weights become nan")
        # if torch.isinf(attn_weights).any():
        #     print(f"inputs: nan: {torch.isnan(inputs).any()}, inf: {torch.isinf(inputs).any()}")
        #     print(f"outs: nan: {torch.isnan(outs).any()}, inf: {torch.isinf(outs).any()}")
        #     print(f"cidxs: nan: {torch.isnan(cidxs).any()}, inf: {torch.isinf(cidxs).any()}")
        #     print(f"dists: nan: {torch.isnan(dists).any()}, inf: {torch.isinf(dists).any()}")
        #     if log_weights is not None:
        #         print(f"log_weights: nan: {torch.isnan(log_weights).any()}, inf: {torch.isinf(log_weights).any()}")
        #     raise RuntimeError("attn_weights become inf")
        #
        # # ((dc, nc) * (nc, 1))^T
        # # # reshape context_vector from (nc, batch, dc) to (batch, nc, dc)
        # # context_vectors_reshaped = context_vectors.permute(1,0,2).unsqueeze(0).expand(seq_len,-1,-1,-1)
        # attn_contexts = torch.bmm(
        #     attn_weights.view(seq_len * batch_size, 1, total_char)[:, :, :num_char],  # (seq_len*batch, 1, num_chars)
        #     context_vectors.permute(1, 0, 2)  # (batch_size, num_chars, dim_context)
        #         .unsqueeze(0)  # (1, batch_size, num_chars, dim_context)
        #         .expand(seq_len, -1, -1, -1)  # (seq_len, batch_size, num_chars, dim_context)
        #         .reshape(seq_len * batch_size, num_char, dim_context),  # (seq_len*batch_size, num_chars, dim_context)
        # )  # (seq_len*batch_size, 1, dim_context)
        # # reshape attn_contexts (seq_len*batch_size,1,dim_context) -> (seq_len, batch_size, dim_context)
        # attn_contexts = attn_contexts.view(seq_len, batch_size, dim_context)
        #
        # return (
        #     attn_contexts,
        #     attn_weights,
        #     means.squeeze(3),
        #     vars.squeeze(3),
        #     weights.squeeze(3),
        # )

    @staticmethod
    def compute_attn_contexts(
            attn_weights: torch.Tensor,
            context_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the attended context given attention weights
        with matrix multiplication.

        Args:
            attn_weights: (seq_len, batch_size, total_char)
                one for each time step, total_char = num_char + extra_chars
            context_vectors: (num_chars, batch_size, dim_context)
                can be onehot encoding or embeddings

        Returns:
            attn_contexts: (seq_len, batch_size, dim_context)
                one for each time step
        """

        seq_len, batch_size, total_char = attn_weights.shape
        num_char = context_vectors.size(0)
        dim_context = context_vectors.size(-1)

        attn_contexts = torch.bmm(
            attn_weights.view(seq_len * batch_size, 1, total_char)[:, :, :num_char],  # (seq_len*batch, 1, num_chars)
            context_vectors.permute(1, 0, 2)  # (batch_size, num_chars, dim_context)
                .unsqueeze(0)  # (1, batch_size, num_chars, dim_context)
                .expand(seq_len, -1, -1, -1)  # (seq_len, batch_size, num_chars, dim_context)
                .reshape(seq_len * batch_size, num_char, dim_context),  # (seq_len*batch_size, num_chars, dim_context)
        )  # (seq_len*batch_size, 1, dim_context)
        # reshape attn_contexts (seq_len*batch_size,1,dim_context) -> (seq_len, batch_size, dim_context)
        attn_contexts = attn_contexts.view(seq_len, batch_size, dim_context)
        return attn_contexts

    def compute_attn_weights(
            self,
            inputs: torch.Tensor,
            total_char: int,
            init_means: torch.Tensor = None,
    ) -> T.Dict[str, torch.Tensor]:
        """
        Args:
            inputs: (seq_len, batch_size, dim_input)
                input used to compute attention
            total_char:
                max total number of characters
            init_means: (batch_size, num_mixtures) or (1, batch_size, num_mixtures)
                current means of the gaussians (None: all zeros)

        Returns:
            attn_weights (seq_len, batch_size, total_char)
                one for each time step, total_char = num_char + extra_chars
            means (seq_len, batch_size, num_mixtures)
                mean of each gaussian window
            vars (seq_len, batch_size, num_mixtures)
                variance of each gaussian window
            weights (seq_len, batch_size, num_mixtures)
                weights of each gaussian window
        """
        seq_len = inputs.size(0)
        batch_size = inputs.size(1)
        if init_means is None:
            init_means = self.get_init_means(
                batch_size=batch_size,
                device=inputs.device,
            )  # (batch_size,num_mixtures)

        # get the weight, mean, and std for each time step
        outs = self.main(inputs).unsqueeze(3)  # (seq_len, batch_size, num_mixtures*3,1)
        # make positive
        outs = self.make_pos_fun(outs)
        delta_means, vars, weights = torch.chunk(
            outs,
            chunks=3,
            dim=2,
        )  # each is (seq_len, batch_size, num_mixtures,1)

        # small tweaks
        delta_means = delta_means / 25.0
        vars = torch.clamp(vars, min=0.01, max=100.0)
        # end small tweaks

        means = delta_means.cumsum(dim=0) + init_means.view(
            1, batch_size, self.num_mixtures, 1
        )  # (seq_len, batch_size, num_mixtures, 1)

        # compute attn_weights
        cidxs = torch.arange(total_char, dtype=torch.float, device=inputs.device).view(
            1, 1, 1, total_char
        )  # (1,1,1,total_char)
        dists = -1.0 * (means - cidxs).pow(2) * vars  # (seq_len, batch_size, num_mixtures, total_char)

        if self.normalized_weights:
            log_weights = self.log_softmax(weights)
            attn_weights = torch.logsumexp(log_weights + dists, dim=2).exp()  # (seq_len, batch_size, total_char)
        else:
            log_weights = None
            attn_weights = torch.logsumexp(weights.log() + dists, dim=2).exp()  # (seq_len, batch_size, total_char)

        # # DEBUG
        if torch.isnan(attn_weights).any():
            print(f"inputs: nan: {torch.isnan(inputs).any()}, inf: {torch.isinf(inputs).any()}")
            print(f"outs: nan: {torch.isnan(outs).any()}, inf: {torch.isinf(outs).any()}")
            print(f"cidxs: nan: {torch.isnan(cidxs).any()}, inf: {torch.isinf(cidxs).any()}")
            print(f"dists: nan: {torch.isnan(dists).any()}, inf: {torch.isinf(dists).any()}")
            if log_weights is not None:
                print(f"log_weights: nan: {torch.isnan(log_weights).any()}, inf: {torch.isinf(log_weights).any()}")
            raise RuntimeError("attn_weights become nan")
        if torch.isinf(attn_weights).any():
            print(f"inputs: nan: {torch.isnan(inputs).any()}, inf: {torch.isinf(inputs).any()}")
            print(f"outs: nan: {torch.isnan(outs).any()}, inf: {torch.isinf(outs).any()}")
            print(f"cidxs: nan: {torch.isnan(cidxs).any()}, inf: {torch.isinf(cidxs).any()}")
            print(f"dists: nan: {torch.isnan(dists).any()}, inf: {torch.isinf(dists).any()}")
            if log_weights is not None:
                print(f"log_weights: nan: {torch.isnan(log_weights).any()}, inf: {torch.isinf(log_weights).any()}")
            raise RuntimeError("attn_weights become inf")

        return dict(
            attn_weights=attn_weights,  # (seq_len, batch_size, total_char)
            means=means.squeeze(3),  # (seq_len, batch_size, num_mixtures)
            vars=vars.squeeze(3),  # (seq_len, batch_size, num_mixtures)
            weights=weights.squeeze(3),  # (seq_len, batch_size, num_mixtures)
        )