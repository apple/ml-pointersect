#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#

import unittest
from cdslib.core.data.dataloader.distributed_bucket_sampler import DistributedBucketSampler
import numpy as np


class TestDistributedBucketSampler(unittest.TestCase):
    def _test(
            self,
            batch_size: int,
            num_replicas: int,
            drop_last: bool,
            shuffle: bool,
            bucket_boundaries: int,
    ):
        seq_lens = list(range(100))
        all_samplers = []
        for rank in range(num_replicas):
            sampler = DistributedBucketSampler(
                seq_lens=seq_lens,
                batch_size=batch_size,
                num_replicas=num_replicas,
                rank=rank,
                drop_last=drop_last,
                shuffle=shuffle,
                bucket_boundaries=bucket_boundaries,
                seed=0,
            )
            all_samplers.append(sampler)

        # pprint(all_samplers[0].bucket_idx_to_data_idxs)

        # check all samplers have the same boundaries
        bucket_boundaries = all_samplers[0].bucket_boundaries
        for rank in range(num_replicas):
            assert np.allclose(all_samplers[rank].bucket_boundaries, bucket_boundaries)

        for sampler in all_samplers:
            sampler.set_epoch(10)

        # check all samplers have the same number of batches
        total_batches = [len(sampler) for sampler in all_samplers]
        assert len(np.unique(total_batches)) == 1
        total_batch = total_batches[0]

        # construct iterators
        all_iters = [iter(sampler) for sampler in all_samplers]

        # gather all batches from all ranks
        seq_len_set = set(seq_lens)
        for batch_idx in range(total_batch):
            batches = [next(it) for it in all_iters]

            # check all data in a batch from the same bucket
            bidxs = []
            for batch in batches:
                bucket_idxs = np.digitize(batch, bucket_boundaries)  # (batch,)
                assert len(np.unique(bucket_idxs)) == 1
                bidxs.append(bucket_idxs[0])

            # check all ranks has the same bucket
            assert len(np.unique(bidxs)) == 1

            # make sure all data go through at least once
            for rank in range(num_replicas):
                seq_len_set -= set(batches[rank])

        if not drop_last:
            assert len(seq_len_set) == 0, f'{len(seq_len_set)}'

    def test1(self):
        self._test(
            batch_size=2,
            num_replicas=5,
            drop_last=False,
            shuffle=False,
            bucket_boundaries=2,
        )

    def test2(self):
        self._test(
            batch_size=3,
            num_replicas=7,
            drop_last=False,
            shuffle=False,
            bucket_boundaries=6,
        )

    def test3(self):
        self._test(
            batch_size=5,
            num_replicas=3,
            drop_last=True,
            shuffle=True,
            bucket_boundaries=3,
        )

    def test4(self):
        self._test(
            batch_size=5,
            num_replicas=3,
            drop_last=True,
            shuffle=True,
            bucket_boundaries=[50., 100.],
        )


if __name__ == '__main__':
    unittest.main()
