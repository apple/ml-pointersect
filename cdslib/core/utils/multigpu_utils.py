#
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
"""
initial code from: https://github.com/NVIDIA/tacotron2/blob/master/distributed.py
"""
import datetime
import os

import torch
from torch.autograd import Variable
import torch.distributed as dist


def init_distributed(n_gpus=-1, rank=-1, auto_detect: bool = True):
    print("Initializing Distributed", flush=True)

    if auto_detect:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(weeks=2),
        )
    else:
        # Set cuda device so everything is done on the right GPU.
        if os.environ.get("MASTER_ADDR", None) is None or os.environ.get("MASTER_PORT", None) is None:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"  # f'{port_number}'  # '12355'

        # Initialize distributed communication
        dist.init_process_group(
            backend="nccl",
            world_size=n_gpus,
            rank=rank,
            timeout=datetime.timedelta(weeks=2),
        )

    print(
        "Done initializing distributed. rank = {}, world size = {}".format(
            rank if rank != -1 else dist.get_rank(),
            n_gpus if n_gpus != -1 else dist.get_world_size(),
        ),
        flush=True,
    )


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def apply_gradient_allreduce(module):
    """
    This version of DistributedDataParallel is designed to be used in conjunction with the multiproc.py
    launcher included with this example. It assumes that your run is using multiprocess with 1
    GPU/process, that the model is on the correct device, and that torch.set_device has been
    used to set the device.
    Parameters are broadcasted to the other processes on initialization of DistributedDataParallel,
    and will be all-reduced at the finish of the backward pass.

    Note by Rick: Compared to the simple pytorch's DDP wrapper on modules, which synchronizes at both forward and backward,
    wrapping the method around a module only synchronize the gradient (not the output)! This is more efficient when
    multiple networks are concatenated together, and if we don't care about the intermediate inputs from all threads.
    Also, the method does NOT synchronize buffers (so batchnorm's statistics are on a smaller dateset).
    """
    if not hasattr(dist, "_backend"):
        module.warn_on_half = True
    else:
        module.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

    for p in module.state_dict().values():
        if not torch.is_tensor(p):
            continue
        dist.broadcast(p, 0)

    def allreduce_params():
        if module.needs_reduction:
            module.needs_reduction = False
            buckets = {}
            for param in module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.dtype
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
            if module.warn_on_half:
                if torch.cuda.HalfTensor in buckets:
                    print(
                        "WARNING: gloo dist backend for half parameters may be extremely slow."
                        + " It is recommended to use the NCCL backend in this case. This currently requires"
                        + "PyTorch built from top of tree master."
                    )
                    module.warn_on_half = False

            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                dist.all_reduce(coalesced)
                coalesced /= dist.get_world_size()
                for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    for param in list(module.parameters()):

        def allreduce_hook(*unused):
            Variable._execution_engine.queue_callback(allreduce_params)

        if param.requires_grad:
            param.register_hook(allreduce_hook)

    def set_needs_reduction(self, input, output):
        self.needs_reduction = True

    module.register_forward_hook(set_needs_reduction)
    return module


def reduce_tensor(tensor: torch.Tensor):
    """Gather and sum the tensor from all gpus."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt
