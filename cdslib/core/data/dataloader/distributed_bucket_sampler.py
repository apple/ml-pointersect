#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
# Author: Rick Chang

import copy
import logging
import math
import os
import typing as T

import numpy as np
import torch.distributed as dist

log = logging.getLogger(__name__)
if log.getEffectiveLevel() == logging.NOTSET:
    log.setLevel(os.environ.get("LOGLEVEL", "INFO"))


class DistributedBucketSampler:
    r"""
    The class assembles batches by grouping data of similar lengths.
    It returns indices of the data in the dataset within a batch --
    in other words, it is a BatchSampler of pytorch.

    Compared to the typical bucket sampler, this implementation supports
    distributed data parallel. Specifically, it makes sure all processes
    sample from the same bucket at the same time (so overall time spent
    on individual batches across GPUs are approximately the same).

    Here is an example to use the distributed bucket sampler:

    .. code-block:: python

        from distributed_bucket_sampler import DistributedBucketSampler
        import numpy as np

        train_sampler = DistributedBucketSampler(
            seq_lens=seq_lens,  # a list containing the sequence lengths of samples in dataset
            batch_size=options.batch_size,
            num_replicas=options.n_gpus if options.distributed_run else 1,
            rank=options.rank if options.distributed_run else 0,
            drop_last=False,  # False is recommended
            bucket_boundaries=10,  # can be the number of bins or the bin edges
            seed=options.random_seed,
            shuffle=True,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=options.num_threads,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            batch_sampler=train_sampler,
        )

        for epoch in range(max_epoch):
            train_sampler.set_epoch(epoch)
            for b, batch in enumerate(dataloader):
                loss = compute_loss(batch)

    """

    def __init__(
        self,
        seq_lens: T.List[int],
        batch_size: int,
        num_replicas: T.Optional[int] = None,
        rank: T.Optional[int] = None,
        drop_last: bool = False,
        shuffle: bool = True,
        bucket_boundaries: T.Union[int, T.List[int]] = 10,
        seed: int = 0,
    ):
        """
        Args:
            seq_lens:
                sequence length of individual samples.
            batch_size:
                batch size of individual process
            num_replicas (int, optional): Number of processes participating in
                distributed training. By default, :attr:`world_size` is retrieved from the
                current distributed group.
            rank (int, optional): Rank of the current process within :attr:`num_replicas`.
                By default, :attr:`rank` is retrieved from the current distributed
                group.
            drop_last:
                whether to drop the last few data that cannot form a batch. It fills in data
                to form a batch with batch_size.  Recommended: `False`.
            shuffle:
                whether to shuffle the data within buckets (according to their length of course).
                Note that the order of the batches is always random (shuffled).
            bucket_boundaries:
                int (number of bins) or a list (containing the edges in ascending order,
                excluding two outmost boundaries).

                edge       0   1   2
                range   ___|___|___|___
        """

        # determine rank and replica
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )

        self.seq_lens = seq_lens
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.bucket_boundaries = bucket_boundaries

        # determine buckets
        self._determine_buckets()

        # construct batches
        self._construct_batches()

    def _determine_buckets(self):
        """Divide data into buckets based on seq_lens."""

        if isinstance(self.bucket_boundaries, (int, str)):
            self.bucket_boundaries = np.histogram_bin_edges(self.seq_lens, bins=self.bucket_boundaries)

        self.bucket_boundaries = np.array(self.bucket_boundaries)
        # insert 0 and inf to bucket_boundaries
        self.bucket_boundaries = np.append(np.insert(self.bucket_boundaries, 0, -np.inf), np.inf)

        # assign data to bins
        self.bucket_idxs = np.digitize(self.seq_lens, self.bucket_boundaries)  # (batch,)

        self.bucket_idx_to_data_idxs = dict()  # bucket idx -> data idx
        for i in range(len(self.bucket_idxs)):
            if self.bucket_idx_to_data_idxs.get(self.bucket_idxs[i]) is None:
                self.bucket_idx_to_data_idxs[self.bucket_idxs[i]] = []
            self.bucket_idx_to_data_idxs[self.bucket_idxs[i]].append(i)  # sorted

    def set_epoch(self, epoch: int):
        """Set the epoch number.
        Should be called at the beginning of every epoch, before iterator is constructed.
        """
        self.epoch = epoch

    def _construct_batches(self):
        """Construct the batches."""

        local_bid2sid = dict()  # bucket id -> sample id
        self.batches = []
        for bid in self.bucket_idx_to_data_idxs:
            idxs = copy.deepcopy(self.bucket_idx_to_data_idxs[bid])
            num_samples = len(idxs)

            # # sort the data idx in each bucket (no need to sort, already sorted)
            # idxs = np.sort(idxs)

            # shuffle individual samples in each bucket
            if self.shuffle:
                # rng should be independent of rank
                rng = np.random.default_rng(self.seed + self.epoch)
                rng.shuffle(idxs)

            # padding data so that the number of samples in each
            # non-empty bucket is a multiple of batch_size * num_replica
            if not self.drop_last:
                num_batch_needed = (
                    math.ceil(math.ceil(num_samples / self.batch_size) / self.num_replicas) * self.num_replicas
                )
                num_samples_needed = num_batch_needed * self.batch_size
            else:
                # if drop_last = True, it can potentially drop a lot of samples
                num_samples_needed = (
                    ((num_samples // self.batch_size) // self.num_replicas) * self.batch_size * self.num_replicas
                )

            # add extra samples to make it evenly divisible
            padding_size = num_samples_needed - num_samples
            if padding_size > 0:
                if padding_size <= num_samples:
                    idxs += idxs[:padding_size]
                else:
                    idxs += (idxs * math.ceil(padding_size / num_samples))[:padding_size]
            else:
                idxs = idxs[:num_samples_needed]

            assert len(idxs) % (self.batch_size * self.num_replicas) == 0

            # subsample the index based on rank
            idxs = idxs[self.rank :: self.num_replicas]
            local_bid2sid[bid] = idxs

            # divide into batches
            for i in range(0, len(idxs), self.batch_size):
                b_idxs = idxs[i : (i + self.batch_size)]
                assert len(b_idxs) == self.batch_size
                self.batches.append(b_idxs)

        # shuffle all_batches
        # note that it is important that all ranks use the same random seed
        # Since every bucket has the same number of batches in every rank,
        # the bucket index will be the same across all ranks
        rng = np.random.default_rng(self.seed + self.epoch)
        ridxs = rng.permutation(len(self.batches))
        self.batches = [self.batches[r] for r in ridxs]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        self._construct_batches()
        return iter(self.batches)
