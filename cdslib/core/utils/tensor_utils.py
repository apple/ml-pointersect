#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# This file implements util functions to manipulate tensors in bulk.

import torch
import typing as T


def tensorfun(
    func: T.Callable[[torch.Tensor], torch.Tensor],
    arr: T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[T.Any, torch.Tensor]],
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[T.Any, torch.Tensor], None]:
    """
    Apply `func(tensor)` to any tensors in the list or dict.

    Args:
        func:
            function that takes a tensor and outputs a tensor
        arr:
            list of tensor, dict of tensor

    Returns:
        list or dict containing the output tensors
    """

    if arr is None:
        return None

    if isinstance(arr, torch.Tensor):
        return func(arr)

    if isinstance(arr, (list, tuple)):
        return [tensorfun(func, a) for a in arr]

    if isinstance(arr, dict):
        out_dict = dict()
        for key in arr:
            out_dict[key] = tensorfun(func, arr[key])
        return out_dict

    raise RuntimeError(f'Not supported type: {type(arr)}, {arr}')


def random_crop(
    xs: torch.Tensor,
    valid_lens: T.Union[torch.Tensor, T.List[int]],
    min_ratio: float = 0.1,
    max_ratio: float = 1.0,
    pad_val: float = 0.,
    min_seq_len: int = 1,
) -> T.Tuple[torch.Tensor, T.Union[torch.Tensor, T.List[int]]]:
    """
    Randomly crop a sequence from each sequence in xs.

    Args:
        xs:
            (seq_len, batch, *)  input sequences
        valid_lens:
            (batch,)  valid sequence length of xs
        min_ratio:
            min ratio in valid sequence length
        max_ratio:
            max ratio in valid sequence length
        pad_val:
            value used to pad when forming a tensor
        min_seq_len:
            minimum sequence length in the cropped sequence

    Returns:
        xs_cropped:
            (seq_len_2, batch, *).  cropped sequences
        new_valid_lens:
            (batch,)  valid sequence length of xs_cropped.
    """
    if valid_lens is None:
        valid_lens = [xs.size(0)] * xs.size(1)

    if isinstance(valid_lens, (list, tuple)):
        valid_lens = torch.tensor(valid_lens)
        is_list = True
        valid_lens_device = None
    else:
        valid_lens_device = valid_lens.device
        is_list = False

    # determine new len
    rs = torch.rand(xs.size(1), device=xs.device) * (max_ratio - min_ratio) + min_ratio  # (batch,)
    new_valid_lens = torch.maximum(
        torch.minimum((valid_lens * rs).long(), valid_lens),
        min_seq_len * torch.ones(1, dtype=torch.long, device=rs.device),
    )
    new_valid_lens = torch.minimum(new_valid_lens, valid_lens)

    # determine starting point
    last_idxs = (valid_lens - new_valid_lens)  # (included)
    s_idxs = (torch.rand(xs.size(1), device=xs.device) * (last_idxs + 1)).long()
    xs_cropped = []
    cropped_valid_lens = []
    for i in range(xs.size(1)):
        x_cropped = xs[s_idxs[i]:s_idxs[i] + new_valid_lens[i], i]
        assert x_cropped.size(0) > 0
        xs_cropped.append(x_cropped)
        cropped_valid_lens.append(x_cropped.size(0))

    xs_cropped = torch.nn.utils.rnn.pad_sequence(
        xs_cropped,
        batch_first=False,
        padding_value=pad_val,
    )  # (seq_len, batch, *)

    if is_list:
        return xs_cropped, cropped_valid_lens
    else:
        cropped_valid_lens = torch.tensor(cropped_valid_lens, device=valid_lens_device)
        return xs_cropped, cropped_valid_lens
