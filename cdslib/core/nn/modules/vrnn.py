#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
# Author: Rick Chang
# This file implements Variational Recurrent Neural Network (VRNN) for maximum-likelihood sequence generation.


import inspect
import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn_utils import get_valid_mask
from ..nn_utils import HyperParams
from .graves import NetworkGraves
from .graves import ParamGraves
from .linear import StackedLinearLayers


class ParamVRNN(HyperParams):
    """A dictionary that contains the hyper-parameters of NetworkGraves."""

    def __init__(
        self,
        param_graves: ParamGraves = ParamGraves(),
        dim_latent: int = HyperParams.TBD.INT,  # actual dimension of the latent variable
        dim_f: int = HyperParams.TBD.INT,  # feature dimension
        # side information
        dim_g: int = 0,  # side information
        # query net
        dim_query: int = 256,  # dimension of posterior query
        num_query_layers: int = 1,
        dim_query_layers: T.Union[int, T.Sequence[int]] = 256,  # feature dimension of the query net (#layer -1)
        query_add_norm: bool = False,  # whether to add layer norm to the query net
        query_dropout_prob: float = 0.0,
        query_nonlinearity: str = "relu",
        # multi-head attention
        num_attn_heads: int = 4,
        attn_dropout_prob: float = 0.1,
        # posterior
        num_posterior_layers: int = 1,
        dim_posterior_layers: T.Union[int, T.Sequence[int]] = 256,
        min_posterior_std: float = 0.0,
        # feature dimension of the posterior net (#layer -1)
        posterior_add_norm: bool = False,  # whether to add layer norm
        posterior_dropout_prob: float = 0.0,
        posterior_type="indep",
        posterior_nonlinearity: str = "relu",
        # prior
        num_prior_layers: int = 1,
        dim_prior_layers: T.Union[int, T.Sequence[int]] = 256,  # feature dimension of the prior net (#layer -1)
        prior_add_norm: bool = False,  # whether to add layer norm
        prior_dropout_prob: float = 0.0,
        prior_nonlinearity: str = "relu",
        min_prior_std: float = 0.0,
        # transform
        num_transform_layers: int = 1,
        dim_transform_layers: T.Union[int, T.Sequence[int]] = 256,  # feature dimension (#layer -1)
        transform_add_norm: bool = False,
        transform_dropout_prob: float = 0.0,
        transform_nonlinearity: str = "relu",
        flag_bckwrd_compatible: bool = False,  # set to True to load older pre-trained model
    ):
        """
        Create Variational RNN.

        Args:
            param_graves:
                Please see the documentation of :py:class:`ParamGraves`.
            dim_latent: int
                Actual dimension of the latent variable.
                Note that it can be different from dim_z, which is the actual input dimension to net_graves.
            dim_f: int
                Feature dimension used to extract latent variables from with attention.

            .. side information
            dim_g: int
                Dimension of side information to be appended to the sampled latent variable

            .. query net

            dim_query: int
                Dimension of the query to the attention module computing the posterior approximation.
            num_query_layers: int
                Number of linear layers to mix attn_hs and attn_cs from net_graves to query.
            dim_query_layers: T.Union[int, T.Sequence[int]]
                Feature dimension of the linear layers in the net_query.
                Can be an integer for all layers or a list of (#layer -1), one for each layer.
            query_add_norm: bool
                Whether to add layer norm to the query layers
            query_dropout_prob: float
                Dropout probability of the query layers
            query_nonlinearity:
                Nonlinearity to use. See :py:class:`StackedLinearLayers` for supported functions.

            .. multi-head attention

            num_attn_heads: int
                Number of heads in the multi-head attention layer
            attn_dropout_prob: float
                Dropout probability used in the multi-head attention layer

            .. posterior

            num_posterior_layers: int
                Number of linear layers used in the posterior net that outputs posterior distribution's parameters.
            dim_posterior_layers: T.Union[int, T.Sequence[int]]
                Feature dimension of the posterior net.
                Can be an integer for all layers or a list of (#layer -1), one for each layer.
            posterior_add_norm: bool
                Whether to add layer norm in the posterior net.
            posterior_dropout_prob: float
                Dropout probability in the posterior net.
            posterior_type='diff',
                Determines how to model the posterior distribution.
                'diff': posterior distribution built upon prior distribution.
                'indep': posterior distribution is modeled separately from the prior distribution.
            posterior_nonlinearity:
                Nonlinearity to use. See :py:class:`StackedLinearLayers` for supported functions.
            min_posterior_std:
                Minimum std (it will be clamped if smaller)

            .. prior

            num_prior_layers: int
                Number of linear layers in the prior net that outputs parameters of prior distribution
            dim_prior_layers: T.Union[int, T.Sequence[int]]
                Feature dimension of the prior net.
                Can be an integer for all layers or a list of (#layer -1), one for each layer.
            prior_add_norm: bool
                Whether to add layer norm in prior net.
            prior_dropout_prob: float
                Dropout probability in the posterior net.
            prior_nonlinearity:
                Nonlinearity to use. See :py:class:`StackedLinearLayers` for supported functions.
            min_prior_std:
                Minimum std (it will be clamped if smaller)

            .. transform

            num_transform_layers: int
                Number of linear layers to transform sampled latent variables to zt.
                Can be 0 if latent variable should be directly used as zt.
            dim_transform_layers: T.Union[int, T.Sequence[int]]
                Feature dimension of the transform_layers.
                Can be an integer for all layers or a list of (#layer -1), one for each layer.
            transform_add_norm: bool
                Whether to add layer norm in transform layers
            transform_dropout_prob: float
                Dropout probability in transform layers.
            transform_nonlinearity:
                Nonlinearity to use. See :py:class:`StackedLinearLayers` for supported functions.
        """

        arg_names = inspect.getfullargspec(self.__init__).args[1:]
        sig = locals()
        arg_dict = {name: sig[name] for name in arg_names}
        super().__init__(**arg_dict)


class NetworkVRNN(nn.Module):
    """
    Variational RNN

    The inputs to the network are:

    - past outputs xs = [x0, x1, ..., x_{T-1}]
    - content sequence cs = [c1, ..., cN]
    - feature sequence fs = [f1, ..., fM]

    The output of the network:

    - ps = [p1, p2, ..., pT]
    - zs = [z1, z2, ..., zT]  sequence of sampled latent variable

    The network is composed of four parts:

    - A :py:class:`NetworkGraves` that handles all the rendering, or P(pt | zt, x0, ..., x_{t-1}, cs)
    - A multi-head dot-product attention that uses current hidden states to focus and extract info from fs
    - A fully connected net to model the posterior q(zt | fs, cs,  x0, ..., x_{t-1})
    - A fully connected net to model the prior p(zt | cs,  x0, ..., x_{t-1})

    It does not include (since they are application-dependent):

    - loss
    - sampling method

    Methods:
        1. forward

            - inputs:

                - [x0, x1, ..., x_{T-1}]
                - [c1, c2, ..., cN]
                - [f1, f2, ..., fM]
                - initial hidden states

            - outputs:

                - [p1, p2, ..., p_T]
                - [\hat{c1}, ... \hat{cT}]
                - final hidden states

        2. get_all_zero_hidden_states
        3. compute_zs
    """

    def __init__(self, param_dict: ParamVRNN = None, **kwargs):
        """
        Create a Variational Recurrent Neural Network (VRNN) model.

        Args:
            param_dict:
                A :py:class:`ParamVRNN` object to define the hyper-parameters of the network.
            kwargs:
                If param_dict is None, you can directly provide keyword arguments of :py:class:`ParamVRNN` here.
        """
        super().__init__()

        # read and set configs
        if param_dict is not None:
            self.config_dict = ParamVRNN(**param_dict)
        else:
            self.config_dict = ParamVRNN(**kwargs)

        for key in self.config_dict:
            setattr(self, key, self.config_dict[key])

        # construct sub-networks
        self._construct_networks()

    def _construct_networks(self):
        """Construct the sub-networks.

        Overview of the model:
            (1) A :py:class:`NetworkGraves` that handles all the rendering, or P(pt | zt, x0, ..., x_{t-1}, cs)
            (2) A multi-head dot-product attention that focuses and extracts info from fs for posterior computation.
            (3) A fully connected net to model the posterior q(zt | fs, cs,  x0, ..., x_{t-1})
            (4) A fully connected net to model the prior p(zt | cs,  x0, ..., x_{t-1})
        """

        # alex graves model as the renderer
        self.net_graves = NetworkGraves(self.param_graves)

        # copy essential hyper-parameters from param_graves
        self.dim_x = self.net_graves.dim_x
        self.dim_c = self.net_graves.dim_c
        self.dim_z = self.net_graves.dim_z
        self.dim_attn_rnn_h = self.net_graves.dim_attn_rnn_h
        self.dim_decode_rnn_h = self.net_graves.dim_decode_rnn_h

        # query net (mix attn_hs and attn_cs to the query)
        # input:
        #   attn_hs
        #   attn_cs
        # output:
        #   query
        dim_mix_layer_input = self.dim_attn_rnn_h + self.dim_c
        self.net_query = StackedLinearLayers(
            num_layers=self.num_query_layers,
            dim_input=dim_mix_layer_input,
            dim_output=self.dim_query,
            dim_features=self.dim_query_layers,
            nonlinearity=self.query_nonlinearity,
            add_norm_layer=self.query_add_norm,
            norm_fun=nn.LayerNorm,
            dropout_prob=self.query_dropout_prob,
            output_add_nonlinearity=True,  # added because multi-head attention has a linear transform at input
        )
        if self.flag_bckwrd_compatible:
            self.net_query = nn.Sequential(self.net_query)

        # multi-head dot-product attention
        # key: fs
        # value: fs
        # query: net_query(attn_ht, attn_cs)
        self.net_attn = nn.MultiheadAttention(
            embed_dim=self.dim_query,
            num_heads=self.num_attn_heads,
            dropout=self.attn_dropout_prob,
            kdim=self.dim_f,
            vdim=self.dim_f,
        )

        # posterior
        # input: attended values (seq_len, batch, dim_query)
        # output: posterior distribution parameters (mean, std), assuming independent Gaussian
        if self.posterior_nonlinearity == "leaky_relu":
            nonlinearity_fun = lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False)
        elif self.posterior_nonlinearity == "relu":
            nonlinearity_fun = lambda: nn.ReLU(inplace=False)
        elif self.posterior_nonlinearity == "tanh":
            nonlinearity_fun = nn.Tanh
        elif self.posterior_nonlinearity == "sigmoid":
            nonlinearity_fun = nn.Sigmoid
        elif self.posterior_nonlinearity in {"silu", "swish"}:
            nonlinearity_fun = lambda: nn.SiLU(inplace=False)
        else:
            raise ValueError("unsupported posterior_nonlinearity")

        self.net_posterior = nn.Sequential(
            nonlinearity_fun(),  # add a relu because multihead attention's output has a linear transform
            StackedLinearLayers(
                num_layers=self.num_posterior_layers,
                dim_input=self.dim_query,  # attended value has the same dimension as the query
                dim_output=self.dim_latent * 2,  # mean, diag of cov
                dim_features=self.dim_posterior_layers,
                nonlinearity=self.posterior_nonlinearity,
                add_norm_layer=self.posterior_add_norm,
                norm_fun=nn.LayerNorm,
                dropout_prob=self.posterior_dropout_prob,
                output_add_nonlinearity=False,
            ),
        )

        # prior
        # input: attn_hs, attn_cs
        # output: prior distribution parameters (mean, std), assuming independent Gaussian
        self.net_prior = StackedLinearLayers(
            num_layers=self.num_prior_layers,
            dim_input=self.dim_query,
            dim_output=self.dim_latent * 2,  # mean, diag of cov
            dim_features=self.dim_prior_layers,
            nonlinearity=self.prior_nonlinearity,
            add_norm_layer=self.prior_add_norm,
            norm_fun=nn.LayerNorm,
            dropout_prob=self.prior_dropout_prob,
            output_add_nonlinearity=False,
        )

        # latent transform layers
        # input: sampled latent variable zt
        # output: some transformation of zt
        if self.num_transform_layers > 0:
            self.net_transform = StackedLinearLayers(
                num_layers=self.num_transform_layers,
                dim_input=self.dim_latent + self.dim_g,
                dim_output=self.dim_z,  # mean, diag of cov
                dim_features=self.dim_transform_layers,
                nonlinearity=self.transform_nonlinearity,
                add_norm_layer=self.transform_add_norm,
                norm_fun=nn.LayerNorm,
                dropout_prob=self.transform_dropout_prob,
                output_add_nonlinearity=True,  # add a relu because lstm has a linear transform at input
            )
            if self.flag_bckwrd_compatible:
                self.net_transform = nn.Sequential(self.net_transform)

        else:
            assert (self.dim_latent + self.dim_g) == self.dim_z
            self.net_transform = None

    def get_zero_hidden_states(self, batch_size=1, device=torch.device("cpu")) -> T.Dict[str, T.Any]:
        """
        Generate all-zero hidden states that can be used to initialize the model
        or as a template.

        Args:
            batch_size:
                batch size
            device:
                the device to put the hidden states

        Returns:
            zero_hidden_state used by network_graves.
        """
        return self.net_graves.get_zero_hidden_states(batch_size=batch_size, device=device)

    def compute_zs(
        self,
        fs: torch.Tensor,
        attn_hs: torch.Tensor,
        attn_cs: torch.Tensor,
        valid_lens_fs: T.Sequence[int] = None,
        pz_multiplier: float = 0.0,
        qz_multiplier: float = 1.0,
        pz_std_multiplier: float = 1.0,
        qz_std_multiplier: float = 1.0,
        gs: torch.Tensor = None,
        pz_min_std: float = 0.0,
        qz_min_std: float = 0.0,
    ):
        """
        Compute the prior and the posterior through the following operations:
            1. mix attn_hs and attn_cs to get query
            2. dot-product attention (key: fs, value: fs, query: query)
            3. convert the output values to posterior's parameters (mean, std of indep. Gaussian)
            4. sample latent_variable from posterior (weighted by pz_multifier and qz_multiplier)
            5. convert sampled latent_variable to zs

        Args:
            fs: (seq_len_fs, batch, dim_f)
            attn_hs: (seq_len, batch, dim_attn_h)
            attn_cs: (seq_len, batch, dim_attn_h)
            valid_lens_fs: (batch,) list of the valid length of fs
            pz_multiplier: float,
            qz_multiplier: float,
            pz_std_multiplier: float,
            qz_std_multiplier: float,
            gs: (seq_len, batch, dim_g)

        Returns:
            avail_output: T.Dict[str, T.Any]
                zs: (seq_len, batch, dim_z)
                queries: (seq_len, batch, dim_query)
                sampled_latents: (seq_len, batch, dim_latent)
                pz_means: (seq_len, batch, dim_latent)
                pz_stds: (seq_len, batch, dim_latent)
                qz_means: (seq_len, batch, dim_latent)
                qz_stds: (seq_len, batch, dim_latent)
                attn_fs_weights: (batch, seq_len, seq_len_fs), attention weight between queries and fs
        """

        # mix attn_hs and attn_cs to get queries
        queries = self.net_query(torch.cat((attn_hs, attn_cs), dim=2))

        # compute prior
        raw_pzs = self.net_prior(queries)  # (seq_len, batch, dim)
        pz_means, pz_stds_raw = raw_pzs.chunk(2, dim=2)  # (seq_len, batch, dim_latent)
        pz_stds = F.softplus(pz_stds_raw).clamp(min=pz_min_std)

        # compute posterior
        if valid_lens_fs is not None:
            key_padding_mask = torch.logical_not(get_valid_mask(valid_lens_fs, max_len=fs.size(0), device=fs.device))
            # BoolTensor, (batch, num_cells), trues will be ignored
        else:
            key_padding_mask = None

        # compute the attention
        attn_output, attn_fs_weights = self.net_attn(
            query=queries,
            key=fs,
            value=fs,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        # attn_output: (seq_len, batch, dim_query)
        # attn_output_weights: (batch, seq_len, seq_len_fs)

        # use the attn_output to compute the distribution parameters
        raw_zs = self.net_posterior(attn_output)  # (seq_len, batch, dim)
        qz_means, qz_stds_raw = raw_zs.chunk(2, dim=2)  # (seq_len, batch, dim_latent)
        qz_stds = F.softplus(qz_stds_raw).clamp(min=qz_min_std)

        if self.posterior_type == "indep":
            pass
        elif self.posterior_type == "diff":
            qz_means = pz_means + qz_means
            qz_stds = pz_stds * qz_stds
        else:
            raise NotImplementedError

        # sample latent variable
        sampled_latents = pz_multiplier * (
            pz_means + pz_std_multiplier * pz_stds * torch.randn_like(pz_stds)
        ) + qz_multiplier * (qz_means + qz_std_multiplier * qz_stds * torch.randn_like(qz_stds))
        # (seq_len, batch, dim_latent)

        if gs is None:
            transform_input = sampled_latents  # (seq_len, batch, dim_latent)
        else:
            transform_input = torch.cat((sampled_latents, gs), dim=-1)  # (seq_len, batch, dim_latent+dim_g)

        # transform sampled latent variable to zs
        if self.net_transform is not None:
            zs = self.net_transform(transform_input)
        else:
            zs = transform_input

        return {
            "zs": zs,  # after another transformation
            "queries": queries,
            "sampled_latents": sampled_latents,  # before transformation
            "pz_means": pz_means,
            "pz_stds": pz_stds,
            "qz_means": qz_means,
            "qz_stds": qz_stds,
            "attn_fs_weights": attn_fs_weights,
        }

    def forward(
        self,
        xs: torch.Tensor,
        cs: torch.Tensor,
        fs: torch.Tensor,
        gs: torch.Tensor = None,
        init_hidden_states: T.Dict[str, torch.Tensor] = None,
        valid_lens_fs: T.Sequence[int] = None,
        pz_multiplier: float = 0.0,
        qz_multiplier: float = 1.0,
        pz_std_multiplier: float = 1.0,
        qz_std_multiplier: float = 1.0,
    ):
        """
        Run the network T (= seq_len) times and generate T outputs, where T is the sequence length of xs.

        Args:
            xs: (seq_len, batch, dim_x)
                Past input sequence (starting from x0, which is usually zero).
                Can be constructed by calling construct_teacher_vectors(xs_gt, past_to_use=1).
            cs: (num_chars, batch, dim_c)
                Content sequence (can be onehot encoded or some embedding)
            gs: (seq_len, batch, dim_g)
                Side information sequence (gs[t] is concatenated to
                sampled_latent before transformed into z and given to the denoder_rnn)
            fs: (seq_len_2, batch, dim_f)
                Feature sequence that the latent variables will be extracted from
            init_hidden_states:
                A dict containing the hidden states of the model
                'attn_rnn_h'
                'decode_rnn_h'
                'attn_c'
                'attn_mean_idxs'

        Returns:
            A dict containing:
                ps: (seq_len, batch, dim_p)
                attn_cs: (seq_len, batch, dim_c)
                attn_weights: (seq_len, batch, num_chars+1)
                attn_fs_weights: (seq_len, batch, seq_len_2)
                attn_hs: (seq_len, batch, dim_attn_h)
                decode_hs: (seq_len, batch, dim_decode_h)
                hidden_states: A dict containing the hidden states of the model

                    - 'attn_rnn_h'
                    - 'decode_rnn_h'
                    - 'attn_c'
                    - 'attn_mean_idxs'

                .. outputs from zt_fun

                zs: (seq_len, batch, dim_z)
                queries: (seq_len, batch, dim_query)
                sampled_latents: (seq_len, batch, dim_latent)
                pz_means: (seq_len, batch, dim_latent)
                pz_stds: (seq_len, batch, dim_latent)
                qz_means: (seq_len, batch, dim_latent)
                qz_stds: (seq_len, batch, dim_latent)
                attn_fs_weights: (batch, seq_len, seq_len_fs), attention weight between queries and fs
        """

        # create qt_fun as a lambda function
        def zt_fun(avail_inputs):
            return self.compute_zs(
                fs=fs,
                attn_hs=avail_inputs["attn_hs"],
                attn_cs=avail_inputs["attn_cs"],
                valid_lens_fs=valid_lens_fs,
                pz_multiplier=pz_multiplier,
                qz_multiplier=qz_multiplier,
                pz_std_multiplier=pz_std_multiplier,
                qz_std_multiplier=qz_std_multiplier,
                gs=gs,
                pz_min_std=self.min_prior_std,
                qz_min_std=self.min_posterior_std,
            )

        # execute net_graves
        return_dict = self.net_graves(
            xs=xs,
            cs=cs,
            init_hidden_states=init_hidden_states,
            zt_fun=zt_fun,
        )

        # expand zt_fun_outputs into return_dict
        zt_fun_outputs = return_dict["zt_fun_outputs"]
        assert zt_fun_outputs is not None
        return_dict.update(zt_fun_outputs)
        return_dict.pop("zt_fun_outputs")

        return return_dict
