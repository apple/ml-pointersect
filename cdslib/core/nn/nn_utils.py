#
# Copyright (C) 2021 Apple Inc. All rights reserved.
# Author: Rick Chang
#
# This file implements the util function for layers.py.

from abc import ABC
from collections.abc import MutableMapping
import enum
from enum import Enum
import typing as T

import numpy as np
import torch


def init_weight(
    weight: torch.Tensor,
    w_init_gain: str = "linear",
    init_method: str = "xavier_normal",
    lrelu_nslope: float = 0.01,
    kaiming_fan_mode: str = "fan_in",
):
    """
    A helper function to initialize the weights of a linear/convolutional layer.

    Args:
        weight:
            (*), an n-dimensional torch.Tensor to be initialized
        w_init_gain:
            The nonlinearity after the linear layer. Can be chosen from the functions supported by
            torch.nn.init.calculate_gain.
            This includes:
            'linear', 'relu', 'silu', 'leaky_relu', 'relu', 'tanh', 'sigmoid'.
        init_method:
            The initialization method. Can be chosen from:
                'normal':
                    randomly sampeld from a Gaussian distribution
                'uniform':
                    randomly sampeld from a uniform distribution
                'xavier_uniform':
                    Check :py:func:`torch.nn.init.xavier_uniform_`.
                'xavier_normal'
                    Check :py:func:`torch.nn.init.xavier_normal_`.
                'xavier':
                    same as 'xavier_normal'
                'kaiming_uniform':
                    Check :py:func:`torch.nn.init.kaiming_uniform_`.
                'kaiming_normal'
                    Check :py:func:`torch.nn.init.kaiming_normal_`.
                'kaiming':
                    same as 'kaiming_normal'
                'orthogonal':
                    Check :py:func:`torch.nn.init.orthogonal_`.
        lrelu_nslope:
            Negative slope used in the leaky-relu.
        kaiming_fan_mode:
            Fan mode used by kaiming_* init_methods.
    Returns:
        Does not return. The function directly modifies the content of weight.

    Note that the function contains torch.no_grad, so there is no need to wrap it with one.
    """

    # handle silu/swish
    if w_init_gain in {"silu", "swish"}:
        # since silu and relu has similar shape, use the gain for relu
        w_init_gain = "relu"

    # calculate gain
    if w_init_gain == "leaky_relu":
        gain = torch.nn.init.calculate_gain(w_init_gain, lrelu_nslope)
        kaiming_a = lrelu_nslope
    else:
        gain = torch.nn.init.calculate_gain(w_init_gain)
        kaiming_a = 0

    if init_method in {
        "kaiming_uniform",
        "kaiming_normal",
        "kaiming",
    } and w_init_gain not in {"relu", "leaky_relu"}:
        print("using kaiming init method on %s, not recommended" % (w_init_gain))

    if init_method == "normal":
        torch.nn.init.normal_(weight, 0.0, gain)
    if init_method == "uniform":
        torch.nn.init.uniform_(weight, -gain, gain)
    elif init_method == "xavier_uniform":
        torch.nn.init.xavier_uniform_(weight, gain=gain)
    elif init_method == "xavier_normal" or init_method == "xavier":
        torch.nn.init.xavier_normal_(weight, gain=gain)
    elif init_method == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(weight, a=kaiming_a, mode=kaiming_fan_mode, nonlinearity=w_init_gain)
    elif init_method == "kaiming_normal" or init_method == "kaiming":
        torch.nn.init.kaiming_normal_(weight, a=kaiming_a, mode=kaiming_fan_mode, nonlinearity=w_init_gain)
    elif init_method == "orthogonal":
        torch.nn.init.orthogonal_(weight, gain=gain)
    else:
        raise NotImplementedError("initialization method [%s] is not implemented" % init_method)


def detach(x: T.Union[torch.Tensor, T.Dict[str, T.Any], T.Sequence[torch.Tensor]]):
    """
    Detach each element in x, regardless if it is a tensor or nested list of tensors.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, dict):
        for key, val in x.items():
            x[key] = detach(val)
        return x
    elif isinstance(x, T.Sequence):
        return [detach(xi) for xi in x]
    else:
        raise NotImplementedError


def randn_like(x: T.Union[torch.Tensor, T.Sequence[torch.Tensor]]):
    """
    Create a new tensor or nested list of tensors that has the same shape as x.
    Each of the tensor is filled with iid samples from a standard normal distribution.
    """
    if isinstance(x, torch.Tensor):
        return torch.randn_like(x)
    elif isinstance(x, T.Sequence):
        return [randn_like(xi) for xi in x]
    else:
        raise NotImplementedError


def get_constant_rnn_hidden_states(
    rnn_type: str,
    num_layers: int,
    hidden_size: int,
    batch_size: int = 1,
    bidirectional: bool = False,
    const: float = 0.0,
    device=torch.device("cpu"),
):
    r"""
    Construct a hidden state filled with the given constant number.
    It is a convenient function to get a initial hidden state for pytorch's RNN, GRU, and LSTM.

    Args:
        rnn_type:
            Type of the rnn module.
            'lstm': :py:class:`torch.nn.LSTM`
            'gru': :py:class:`torch.nn.GRU`
            'rnn': :py:class:`torch.nn.RNN`
        num_layers:
            Number of layers
        hidden_size:
            Dimension of the hidden state for all layers.
        batch_size:
            Batch size.
        bidirectional:
            Whether the rnn module is bidirectional.
        const:
            Constant to fill the output hidden state.
        device:
            torch.device to create the hidden state.
    Returns:
        Hidden state of the asked rnn module.

        LSTM:
            list of two tensors: [(num_layers * directions, batch_size, hidden_size),
            (num_layers * directions, batch_size, hidden_size)].
        RNN:
            tensor (num_layers * directions, batch_size, hidden_size).
        GRU:
            tensor (num_layers * directions, batch_size, hidden_size).
    """
    assert rnn_type in {"lstm", "gru", "rnn"}
    directions = 2 if bidirectional else 1
    if rnn_type == "lstm":
        return [
            torch.ones(num_layers * directions, batch_size, hidden_size, device=device) * const,
            torch.ones(num_layers * directions, batch_size, hidden_size, device=device) * const,
        ]
    elif rnn_type in {"gru", "rnn"}:
        return torch.ones(num_layers * directions, batch_size, hidden_size, device=device) * const
    else:
        raise RuntimeError(f"wrong rnn_type {rnn_type}")


def get_valid_mask(
    valid_lens: T.Union[T.Sequence[int], torch.LongTensor, np.ndarray],
    max_len: int = None,
    device=torch.device("cpu"),
    invalid=False,
) -> torch.BoolTensor:
    """
    Returns a BoolTensor B, where B[i,j] = True if j < valid_lens[i].

    Args:
        valid_lens: (batch,)
        max_len: int, max length of the mask. If None, use max(valid_len)
        device: the device of the output mask.
        invalid: invert the result

    Returns:
         boolTensor (batch, max_len)
    """
    if max_len is None:
        max_len = torch.max(valid_lens)

    if isinstance(valid_lens, torch.Tensor):
        pass
    elif isinstance(valid_lens, (list, tuple)):
        valid_lens = torch.tensor(valid_lens, dtype=torch.long, device=device)
    elif isinstance(valid_lens, np.ndarray):
        valid_lens = torch.from_numpy(valid_lens).to(dtype=torch.long, device=device)
    else:
        raise NotImplementedError
    idxs = torch.arange(0, max_len, device=valid_lens.device)  # (max_len,)
    if not invalid:
        mask = idxs < valid_lens.unsqueeze(1)
    else:
        mask = idxs >= valid_lens.unsqueeze(1)
    return mask.to(device=device)


def construct_teacher_vectors(gt_vectors, past_to_use=0, latest_first=False):
    """
    construct time-shifted version of the ground-truth to be used as teacher-forcing input
    :param gt_vectors: (seq_len, batch, dim_feature)
    :param latest_first: True: latest to oldest, False, oldest to latest
    :return: (seq_len, batch, past_stroke_to_use*dim_feature),

    for example (past_to_use=2, latest_first=True):
    gt_vec: 1 2 3 4 5 6 7
    out:    (0,0) (1,0) (2,1) (3,2) (4,3) (5,4) (6,5)

    """
    if past_to_use == 1:
        return construct_last_vectors(gt_vectors)

    assert len(gt_vectors.shape) == 2 or len(gt_vectors.shape) == 3
    if len(gt_vectors.shape) == 2:
        gt_vectors = gt_vectors.unsqueeze(2)

    batch_size = gt_vectors.size(1)
    seq_len = gt_vectors.size(0)
    dim_feature = gt_vectors.size(2)
    if past_to_use == 0:
        return torch.zeros(seq_len, batch_size, 0, device=gt_vectors.device, dtype=gt_vectors.dtype)

    teacher_vectors = [None for _ in range(past_to_use)]
    for i in range(past_to_use):
        past = i + 1
        init_vecs = torch.zeros(
            past,
            batch_size,
            dim_feature,
            device=gt_vectors.device,
            dtype=gt_vectors.dtype,
        )
        gt_vecs = gt_vectors[:-past]
        t_vecs = torch.cat([init_vecs, gt_vecs], dim=0)
        if latest_first:
            teacher_vectors[i] = t_vecs
        else:
            teacher_vectors[past_to_use - i - 1] = t_vecs

    teacher_vectors = torch.cat(teacher_vectors, dim=2)
    assert teacher_vectors.shape[0] == seq_len
    assert teacher_vectors.shape[1] == batch_size
    assert teacher_vectors.shape[2] == past_to_use * dim_feature

    return teacher_vectors


def construct_last_vectors(gt_vectors, delay=1):
    """
    construct time-shifted version of the ground-truth to be used as teacher-forcing input
    where out[t] = gt_vector[t-1]
    :param gt_vectors: (seq_len, batch, dim_feature)
    :return: (seq_len, batch, dim_feature)
    """

    if delay == 0:
        return gt_vectors  # not cloned

    assert len(gt_vectors.shape) == 2 or len(gt_vectors.shape) == 3
    if len(gt_vectors.shape) == 2:
        gt_vectors = gt_vectors.unsqueeze(2)

    out = torch.zeros_like(gt_vectors)
    out[delay:] = gt_vectors[0:-delay]  # cloned
    return out


def construct_onehot_vectors(labels: torch.Tensor, total_classes: int) -> torch.Tensor:
    """
    Given labels
    :param labels: (seq_len, batch_size,), (batch_size,), or (*) int, [0, total_classes-1]
    :param total_classes: total number of classes
    :return: onehot embedding in float (*, total_classes) and same device as labels
    """
    ori_label_shape = labels.shape
    onehots = torch.zeros(labels.numel(), total_classes, device=labels.device)
    onehots.scatter_(1, labels.view(-1, 1), 1)
    return onehots.view(*ori_label_shape, total_classes)


class HyperParams(ABC, MutableMapping):
    """A dictionary that contains the hyper-parameters of a Network."""

    class TBD(Enum):
        ANY = enum.auto()
        INT = enum.auto()
        FLOAT = enum.auto()
        STR = enum.auto()
        DICT = enum.auto()
        LIST = enum.auto()
        SET = enum.auto()
        TENSOR = enum.auto()
        NDARRAY = enum.auto()
        PARAM = enum.auto()
        BOOL = enum.auto()

    def __init__(self, *args, **kwargs):
        self.param_dict: T.Dict[str, T.Any] = dict()  # name -> value of all params
        self.param_dict.update(*args, **kwargs)

    def __str__(self):
        return str(self.param_dict)

    def __repr__(self):
        return str(self.param_dict)

    def __getitem__(self, key):
        return self.param_dict[key]

    def __setitem__(self, key, value):
        self.param_dict[key] = value

    def __delitem__(self, key):
        del self.param_dict[key]

    def __iter__(self):
        return iter(self.param_dict)

    def __len__(self):
        return len(self.param_dict)

    def check_valid(self):
        return self.check_dict_valid(d=self.param_dict)
        # valid = True
        # for key, val in self.param_dict.items():
        #     if isinstance(val, HyperParams):
        #         valid = valid and val.check_valid()
        #         if not valid:
        #             return False
        #     elif isinstance(val, dict):
        #         for k, v in val.items():
        #             if isinstance(v, HyperParams.TBD):
        #                 return False
        #     elif isinstance(val, (list, tuple, set)):
        #         for v in val:
        #             if isinstance(v, HyperParams.TBD):
        #                 return False
        #     elif isinstance(val, HyperParams.TBD):
        #         return False
        # return valid

    @staticmethod
    def check_dict_valid(d: T.MutableMapping[str, T.Any]) -> bool:
        valid = True
        for k, v in d.items():
            if isinstance(v, HyperParams.TBD):
                return False
            elif isinstance(v, HyperParams):
                valid = valid and v.check_valid()
                if not valid:
                    return False
            elif isinstance(v, T.MutableMapping):
                valid = valid and HyperParams.check_dict_valid(v)
                if not valid:
                    return False
            elif isinstance(v, (list, tuple, set)):
                for val in v:
                    if isinstance(val, HyperParams.TBD):
                        return False
        return valid


def pad_till_sequence_length(
        x: torch.Tensor,
        min_seq_len: int,
        pad_val: float = 0.,
        batch_first: bool = False,
):
    """
    Pad x in the sequence dimension so that x has
    sequence length at least min_seq_len

    Args:
        x:
            (seq_len, b, dim) if not batch_first
            (b, dim, seq_len) otherwise
        min_seq_len:
            min seq_len of the padded x
        pad_val:
            value to pad x with
        batch_first:
            whether x is (seq_len, b, dim) if not batch_first, or
            (b, dim, seq_len) otherwise

    Returns:
        (min_seq_len, b, dim) or (b, dim, min_seq_len) if `batch_first` is True

    """
    # pad x so that x is long enough to be downsampled
    if batch_first:
        b, c, seq_len = x.shape
        if seq_len < min_seq_len:
            tmp = x
            x = torch.ones(b, c, min_seq_len, dtype=x.dtype, device=x.device) * pad_val
            x[..., :seq_len] = tmp
    else:
        seq_len, b, c = x.shape
        if seq_len < min_seq_len:
            tmp = x
            x = torch.ones(min_seq_len, b, c, dtype=x.dtype, device=x.device) * pad_val
            x[:seq_len] = tmp
    return x