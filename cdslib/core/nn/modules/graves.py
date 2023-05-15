#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
# Author: Rick Chang
# This file implements Alex Graves' LSTM model for maximum-likelihood sequence generation.


#
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
# The file implements Alex Graves' model in https://arxiv.org/abs/1308.0850

import inspect
import typing as T

import numpy as np
import torch
import torch.nn as nn

from ..nn_utils import get_constant_rnn_hidden_states
from ..nn_utils import HyperParams
from .attention import GaussianSlidingWindows
from .linear import StackedLinearLayers
from .lstm import LSTMCellLayers


class ParamGraves(HyperParams):
    """
    A dictionary that contains the hyper-parameters of Network_Graves.
    """

    def __init__(
        self,
        dim_x: int = HyperParams.TBD.INT,  # dimension of the input
        dim_c: int = HyperParams.TBD.INT,  # dimension of the content
        dim_p: int = HyperParams.TBD.INT,  # dimension of the output
        dim_z: int = 0,  # dimension of the optional z
        # attn_rnn
        num_attn_rnn_layers: int = 1,  # number of layers of the attn_rnn
        dim_attn_rnn: T.Union[int, T.Sequence[int]] = 512,  # dimension of the attn_rnn
        # attn_layer
        num_window_attn_mixtures: int = 10,  # number of mixtures in the gaussian attention layer
        num_window_attn_layers: int = 1,  # number of layers in the gaussian attention layer
        dim_window_attn_layers: T.Union[int, T.Sequence[int]] = 128,  # , len = num_window_attn_layers-1
        # decode_rnn
        num_decode_rnn_layers: int = 2,
        dim_decode_rnn: int = 512,
        # output_layer
        output_layer_type: str = "StackedLinearLayers",
        feed_z_to_decode_rnn: bool = True,
        num_output_layers: int = 1,  # int
        dim_output_layers: T.Union[int, T.Sequence[int]] = 128,
        # T.Union[int, T.Sequence[int]], len = num_output_layers-1
        output_layer_nonlinearity="relu",
        output_layer_add_norm_layer=False,
        output_layer_norm_fun: T.Callable = torch.nn.LayerNorm,
        output_layer_dropout_prob: float = 0,
        output_layer_normalize: bool = True,
        output_layer_normalize_first: bool = True,
        output_layer_init_freq_first_layer: float = 1.0,
        output_layer_init_freq_other_layer: float = 25.0,
        # style mapping layer
        num_mapping_layers: int = 4,
        dim_mapping_layers: int = 256,
        mapping_layer_nonlinearity: str = "silu",
        mapping_layer_dropout_prob: float = 0.0,
    ):
        """
        Args:
            dim_x: int
                Dimension of the past input sequence xs
            dim_c: int
                Dimension of the content cs
            dim_p: int
                Output feature dimension of the network
            dim_z: int = 0
                Dimension of an additional latent vector zt if zt_fun is provided in forward.  Default: 0

            .. attn_rnn

            num_attn_rnn_layers: int = 1
                Number of layers in the attention lstm
            dim_attn_rnn: T.Union[int, T.Sequence[int]] = 512
                Dimension of the attention lstm. Can be a single integer or a list, one for each layer.

            .. attn_layer

            num_window_attn_mixtures: int = 10
                Number of Gaussin mixtures used by the Gaussian window attention layer.
            num_window_attn_layers: int = 1
                Number of layers in the Gaussian window attention layer.
            dim_window_attn_layers: T.Union[int, T.Sequence[int]] = 128
                Dimension of the window attention layers.
                Can be a single integer (all layers share the same dimension,
                or a list of {num_window_attn_layers-1}, one for each layer, except the last layer, whose dimension
                is determined by num_window_attn_mixtures)

            .. decode_rnn

            num_decode_rnn_layers: int = 2
                Number of decode LSTM layers.
            dim_decode_rnn: int = 512
                Dimension of the decode LSTM (accept only integer -- all layers share the same dimension).

            .. output layers

            output_layer_type:
                "StackedLinearLayers": stacked of linear layers
                "StackedModulatedLinearLayers": stacked of modulated linear layers
                "StackedFiLMLayers": stacked of FiLM layers
            feed_z_to_decode_rnn:
                Whether to concatenate z to the input of decode_rnn
            num_output_layers: int = 1
                Number of output linear layers
            dim_output_layers: T.Union[int, T.Sequence[int]] = 128
                Dimension of the output layers.
                Can be a single integer (all layers share the same dimension,
                or a list of {num_output_layers-1}, one for each layer, except the last layer, whose dimension
                is determined by dim_p)
            output_layer_nonlinearity: str = 'relu'
                Nonlinearity used in the output layers (except the last layer, which does not has one).
                Choices: relu, leaky_relu, sigmoid, tanh
            output_layer_add_norm_layer: bool = False
                Whether to add a normalization layer before the nonlinearity in the output layers.
            output_layer_norm_fun: T.Callable = torch.nn.LayerNorm
                Normalization layer to use.
                Should be a function that takes only the dim_feature, e.g., torch.nn.LayerNorm.
                You can pass a lambda function if want to change the default parameters)

                    ex: lambda x: torch.nn.LayerNorm(x, eps=1e-5, elementwise_affine=False)

            output_layer_dropout_prob: float = 0
                Dropout probability used in the output layers (after the nonlinearity)

            output_layer_normalize:
                Whether to normalize in the StackedModulatedLinearLayers
            output_layer_normalize_first:
                Whether to normalize the input in the StackedModulatedLinearLayers
            output_layer_init_freq_first_layer:
                Initial scale of the linear layer in FiLM to use in the first output layer
            output_layer_init_freq_other_layer:
                Initial scale of the linear layer in FiLM to use in the second to the last output layers

            .. style mapping layers

            num_mapping_layers:
                number of layers to transform z to the moluation parameters used in `StackedModulatedLinearLayers` and
                `StackedFiLMLayers`.
            dim_mapping_layers:
                dimension of the style mapping layers. int or list of num_layers-1.
            mapping_layer_nonlinearity:
                nonlinearity to use in the style mapping layers.
            mapping_layer_dropout_prob:
                dropout probability to use in the style mapping layers.
        """
        arg_names = inspect.getfullargspec(self.__init__).args[1:]
        sig = locals()
        arg_dict = {name: sig[name] for name in arg_names}
        super().__init__(**arg_dict)


class NetworkGraves(nn.Module):
    """
    The Alex Graves' model in https://arxiv.org/abs/1308.0850

    The network includes (see Fig 12 in the paper):

    - attn_rnn
    - gaussian sliding window attention
    - decode_rnn
    - output layer

    It does not include (since they are application-independent):

    - loss
    - sampling method

    Methods:
        1. forward

            - inputs:

                - [x0, x1, ..., x_{T-1}]
                - [c1, c2, ..., cN]
                - initial hidden states
                - (optional) zt_fun that generates zt from current states

            - outputs:

                - [p1, p2, ..., p_T]
                - [\hat{c1}, ... \hat{cT}]
                - final hidden states

        2. get_all_zero_hidden_states

    """

    def __init__(self, param_dict: ParamGraves = None, **kwargs):
        """Create an Alex Graves' model.

        Args:
            param_dict:
                A :py:class:`ParamGraves` object to define the hyper-parameters of the network.
            kwargs:
                If param_dict is None, you can directly provide keyword arguments of :py:class:`ParamGraves` here.
        """
        super().__init__()

        # read and set configs
        if param_dict is not None:
            self.config_dict = ParamGraves(**param_dict)
        else:
            self.config_dict = ParamGraves(**kwargs)

        for key in self.config_dict:
            setattr(self, key, self.config_dict[key])

        # construct sub-networks
        self._construct_networks()

    def _construct_networks(self):
        """Construct the sub-networks.

        Overview of the model:
            attn_rnn (lstm_cell_layers):
              input: x_{t-1}, last_attn_context
              output: ht
            attn_layer (gaussian window attention):
              input: ht,
              output: attn_weights, attn_context, attn_delta_means
            decode_rnn (LSTM):
              input: attn_context, ht, x_{t-1}, (zt, if provided)
              output: raw_output
            output_layer:
              input: raw_output
              output: pt
        """

        ## attn_rnn
        # input:
        # (1) x_{t-1} (batch, dim_x)
        # (2) last_attn_context (batch, dim_c)
        dim_attn_rnn_input = self.dim_x + self.dim_c
        self.attn_rnn = LSTMCellLayers(
            num_layers=self.num_attn_rnn_layers,
            dim_input=dim_attn_rnn_input,
            dim_hidden=self.dim_attn_rnn,
        )
        self.dim_attn_rnn_h = self.attn_rnn.dim_hidden[-1]

        ## attn_layer
        # input: attn_rnn_h
        # output: attn_weights, attn_context, attn_delta_means
        dim_attn_layer_input = self.dim_attn_rnn_h
        self.attn_layer = GaussianSlidingWindows(
            num_mixtures=self.num_window_attn_mixtures,
            dim_input=dim_attn_layer_input,
            num_layers=self.num_window_attn_layers,
            dim_features=self.dim_window_attn_layers,
            pos_fun="softplus",
        )

        ## decode rnn
        # input:
        # (1) attn_context (batch, dim_c)
        # (2) h_attn_rnn
        # (3) x_{t-1}
        # (4) z (batch, dim_z), if provided
        dim_decode_rnn_input = self.dim_c + self.dim_attn_rnn_h + self.dim_x
        if self.feed_z_to_decode_rnn:
            dim_decode_rnn_input += self.dim_z

        self.decode_rnn = nn.LSTM(
            input_size=dim_decode_rnn_input,
            hidden_size=self.dim_decode_rnn,
            num_layers=self.num_decode_rnn_layers,
            batch_first=False,
            bidirectional=False,
        )
        self.dim_decode_rnn_h = self.dim_decode_rnn

        # output layers
        # input: decode_ht
        dim_output_layer_input = self.dim_decode_rnn_h
        if self.output_layer_type == "StackedLinearLayers":
            self.output_layer = StackedLinearLayers(
                num_layers=self.num_output_layers,
                dim_input=dim_output_layer_input,
                dim_output=self.dim_p,
                dim_features=self.dim_output_layers,
                nonlinearity=self.output_layer_nonlinearity,
                add_norm_layer=self.output_layer_add_norm_layer,
                norm_fun=self.output_layer_norm_fun,
                dropout_prob=self.output_layer_dropout_prob,
            )
            if self.dim_z > 0 and not self.feed_z_to_decode_rnn:
                raise ValueError(f"{self.dim_z}, {self.feed_z_to_decode_rnn}")
        elif self.output_layer_type == "StackedModulatedLinearLayers":
            from experiments.ricklib.core.nn import StackedModulatedLinearLayers

            self.output_layer = StackedModulatedLinearLayers(
                num_layers=self.num_output_layers,
                dim_input=dim_output_layer_input,
                dim_output=self.dim_p,
                dim_features=self.dim_output_layers,
                nonlinearity=self.output_layer_nonlinearity,
                linear_bias=True,
                dropout_prob=self.output_layer_dropout_prob,
                output_add_nonlinearity=False,
                normalize=self.output_layer_normalize,
                normalize_first=self.output_layer_normalize_first,
            )
            dim_total_modulation = np.sum(self.output_layer.dim_modulators)
            self.net_modulation_mapping = StackedLinearLayers(
                num_layers=self.num_mapping_layers,
                dim_input=self.dim_z,
                dim_output=dim_total_modulation * 2,
                dim_features=self.dim_mapping_layers,
                nonlinearity=self.mapping_layer_nonlinearity,
                add_norm_layer=False,
                dropout_prob=self.mapping_layer_dropout_prob,
                output_add_nonlinearity=False,
            )
        elif self.output_layer_type == "StackedFiLMLayers":
            from experiments.ricklib.core.nn import StackedFiLMLayers

            self.output_layer = StackedFiLMLayers(
                num_layers=self.num_output_layers,
                dim_input=dim_output_layer_input,
                dim_output=self.dim_p,
                dim_features=self.dim_output_layers,
                linear_bias=True,
                dropout_prob=self.output_layer_dropout_prob,
                output_add_nonlinearity=False,
                init_freq_first_layer=self.output_layer_init_freq_first_layer,
                init_freq_other_layer=self.output_layer_init_freq_other_layer,
            )
            dim_total_modulation = np.sum(self.output_layer.dim_modulators)
            self.net_modulation_mapping = StackedLinearLayers(
                num_layers=self.num_mapping_layers,
                dim_input=self.dim_z,
                dim_output=dim_total_modulation * 2,
                dim_features=self.dim_mapping_layers,
                nonlinearity=self.mapping_layer_nonlinearity,
                add_norm_layer=False,
                dropout_prob=self.mapping_layer_dropout_prob,
                output_add_nonlinearity=False,
            )
        else:
            raise NotImplementedError

    def forward(
        self,
        xs: torch.Tensor,
        cs: torch.Tensor,
        init_hidden_states: T.Dict[str, torch.Tensor] = None,
        zt_fun: T.Callable = None,
    ):
        r"""
        Run the network T (= seq_len) times and generate T outputs, where T is the sequence length of xs.

        Args:
            xs: (seq_len, batch, dim_x)
                Past input sequence (starting from x0, which is usually zero).
                Can be constructed by calling construct_teacher_vectors(xs_gt, past_to_use=1).
            cs: (num_chars, batch, dim_c)
                Content sequence (can be onehot encoded or some embedding)
            init_hidden_states:
                A dict containing the hidden states of the model

                    - `attn_rnn_h`
                    - `decode_rnn_h`
                    - `attn_c`
                    - `attn_mean_idxs`

                If None is provided, use all-zero hidden state.
            zt_fun:
                A function (or callable object like nn.module) that takes inputs and return zt.
                Inputs:

                    avail_inputs: T.Dict[str, T.Any], which contains the following keys:

                        - `attn_hs`: (seq_len, batch, dim_attn_h)
                        - `attn_cs`: \hat{c1, ... ct}:  (seq_len, batch, dim_c)

                Output:

                    avail_outputs: T.Dict[str, T.Any]

                        - `zs`: (seq_len, batch, dim_z)

                Note that if attn_hs has a length of n, the function should return n zt's.

        Returns:
            A dict containing:

                ps: (seq_len, batch, dim_p)
                attn_cs: (seq_len, batch, dim_c)
                attn_weights: (seq_len, batch, num_chars+1)
                attn_hs: (seq_len, batch, dim_attn_h)
                decode_hs: (seq_len, batch, dim_decode_h)
                hidden_states: A dict containing the hidden states of the model

                    - `attn_rnn_h`
                    - `decode_rnn_h`
                    - `attn_c`
                    - `attn_mean_idxs`

                zt_fun_outputs: T.Dict[str, T.Any] output dict of zt_fun is provided or None if not provided

        """

        seq_len = xs.size(0)
        batch_size = xs.size(1)
        device = xs.device

        if init_hidden_states is None:
            init_hidden_states = self.get_zero_hidden_states(batch_size=batch_size, device=device)

        # get hidden states
        last_attn_context = init_hidden_states["attn_c"]
        last_attn_means = init_hidden_states["attn_mean_idxs"]
        attn_rnn_h = init_hidden_states["attn_rnn_h"]
        decode_rnn_h = init_hidden_states["decode_rnn_h"]

        # create buffer
        attn_contexts = []  # list of (1, batch_size, self.dim_c)
        attn_weights = []  # list of (1, batch_size, num_chars + 1)
        attn_hs = []  # list of (batch_size, self.dim_attn_rnn_h)

        for t in range(seq_len):
            # attn_rnn
            attn_rnn_input = torch.cat((xs[t], last_attn_context), dim=1)  # (batch, dim)
            attn_h, attn_rnn_h = self.attn_rnn(attn_rnn_input, attn_rnn_h)  # (batch, dim_h)

            if not torch.isfinite(attn_h).all():
                print(f"xs: {torch.isnan(xs).any()}, inf: {torch.isinf(xs).any()}")
                print(f"xs[t ({t})]: nan: {torch.isnan(xs[t]).any()}, inf: {torch.isinf(xs[t]).any()}")
                print(f"last_attn_context: nan: {torch.isnan(last_attn_context).any()}, inf: {torch.isinf(last_attn_context).any()}")
                raise RuntimeError("attn_h become nan")

            # attn_layer
            attn_layer_input = attn_h.unsqueeze(0)  # (1, batch, dim)
            attn_context, attn_weight, attn_means, _, _ = self.attn_layer(
                attn_layer_input, cs, init_means=last_attn_means, extra_chars=1
            )  # add an extra char at the end in case needed

            # update
            last_attn_means = attn_means
            last_attn_context = attn_context.squeeze(0)
            attn_hs.append(attn_h)
            attn_contexts.append(attn_context)
            attn_weights.append(attn_weight)

        attn_hs = torch.stack(attn_hs, dim=0)
        attn_weights = torch.cat(attn_weights, dim=0)
        attn_contexts = torch.cat(attn_contexts, dim=0)

        # get zt if zt_fun is not None
        if zt_fun is not None:
            avail_inputs = {
                "attn_hs": attn_hs,
                "attn_cs": attn_contexts,
            }
            zt_fun_outputs = zt_fun(avail_inputs)
            zs = zt_fun_outputs["zs"]
        else:
            assert self.dim_z == 0
            zt_fun_outputs = None
            zs = torch.zeros(seq_len, batch_size, self.dim_z, device=device)

        if self.feed_z_to_decode_rnn:
            decode_rnn_input = [attn_hs, xs, attn_contexts, zs]
        else:
            decode_rnn_input = [attn_hs, xs, attn_contexts]
        decode_rnn_input = torch.cat(decode_rnn_input, dim=2)  # seq_len, batch, dim
        decoder_hs, decode_rnn_h = self.decode_rnn(decode_rnn_input, decode_rnn_h)

        # output layer
        if self.output_layer_type == "StackedLinearLayers":
            ps = self.output_layer(decoder_hs)  # seq_len, batch, (num_mixtures*6+1+1)
        elif self.output_layer_type == "StackedModulatedLinearLayers":
            modulation_weights, modulation_biases = self.net_modulation_mapping(zs).chunk(2, dim=-1)
            ps = self.output_layer(
                decoder_hs,
                weight=modulation_weights,
                bias=modulation_biases,
            )  # seq_len, batch, (num_mixtures*6+1+1)
        elif self.output_layer_type == "StackedFiLMLayers":
            modulation_weights, modulation_biases = self.net_modulation_mapping(zs).chunk(2, dim=-1)
            ps = self.output_layer(
                decoder_hs,
                freq=modulation_weights,
                phase_shift=modulation_biases,
            )  # seq_len, batch, (num_mixtures*6+1+1)
        else:
            raise NotImplementedError

        # compile final hidden states
        final_hidden_states = {
            "attn_rnn_h": attn_rnn_h,
            "decode_rnn_h": decode_rnn_h,
            "attn_c": last_attn_context,
            "attn_mean_idxs": last_attn_means,
        }

        return_vals = {
            "ps": ps,
            "attn_cs": attn_contexts,
            "attn_weights": attn_weights,
            "attn_hs": attn_hs,
            "decode_hs": decoder_hs,
            "hidden_states": final_hidden_states,
            "zt_fun_outputs": zt_fun_outputs,
        }
        return return_vals

    def get_zero_hidden_states(
        self, batch_size: int = 1, device: torch.device = torch.device("cpu")
    ) -> T.Dict[str, T.Any]:
        """
        Generate all-zero hidden states that can be used to initialize the model
        or as a template.

        Args:
            batch_size:
                batch size
            device:
                the device to put the hidden states

        Returns:
            A dict containing the hidden states.

                - `attn_rnn_h`: list of num_attn_rnn_layers, each a list of 2, each is (batch_size, dim_attn_rnn[layer_idx])
                - `decode_rnn_h`: list of 2, each is (num_decode_rnn_layers, batch_size, dim_decode_rnn)
                - `attn_c`: (batch_size, dim_c)
                - `attn_mean_idxs`: (batch_size, num_window_attn_mixtures)
        """

        # attn_h: list of num_attn_rnn_layers, each is a list of 2, each is (batch_size, dim_attn_rnn[layer_idx])
        attn_rnn_h = self.attn_rnn.get_zero_hidden_states(batch_size=batch_size, device=device)

        # decode_h: list of 2, each is (num_decode_rnn_layers, batch_size, dim_decode_rnn)
        decode_rnn_h = get_constant_rnn_hidden_states(
            rnn_type="lstm",
            num_layers=self.num_decode_rnn_layers,
            hidden_size=self.dim_decode_rnn_h,
            batch_size=batch_size,
            bidirectional=False,
            const=0.0,
            device=device,
        )

        # attn_c (batch_size, dim_c)
        attn_c = torch.zeros(batch_size, self.dim_c, device=device)

        # attn_mean_idxs (batch_size, num_window_attn_mixtures)
        attn_mean_idxs = self.attn_layer.get_init_means(batch_size=batch_size, device=device)

        return {
            "attn_rnn_h": attn_rnn_h,
            "decode_rnn_h": decode_rnn_h,
            "attn_c": attn_c,
            "attn_mean_idxs": attn_mean_idxs,
        }
