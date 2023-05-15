#
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
# Author: Rick Chang

import logging
import os

import numpy as np

log = logging.getLogger(__name__)
if log.getEffectiveLevel() == logging.NOTSET:
    log.setLevel(os.environ.get("LOGLEVEL", "INFO"))


class BucketSampler:
    """
    The class assembles batches by grouping data of similar length.
    It returns indices of the data in the dataset within a batch.

    Here is an example to use the bucket sampler:

    .. code-block:: python

        from bucket_sampler import Bucket_Sampler
        import numpy as np

        dataset_size = 50
        dataset_lengths = np.random.randint(1, 100, dataset_size)
        print(dataset_lengths)

        batch_size = 4
        drop_last = True
        shuffle = True
        bucket_boundaries = [10,20,30,40,50,60,70,80,90]  # I hand-assigned here, but it can be created by np.histogram_bin_edges
        batch_sampler = Bucket_Sampler(dataset_lengths, bucket_boundaries, batch_size, drop_last, shuffle)

        print('num batches = %d' % (len(batch_sampler)))
        for batch_idx, data_idxs in enumerate(batch_sampler):
            print('%d: '% batch_idx, end='' )
            print(data_idxs)

        # no need to assign batch_size, shuffle, drop_last, since batch_sampler determines them
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=bucket_sampler,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_fn
        )

    """

    def __init__(
        self,
        batch_size,
        drop_last,
        shuffle,
        bucket_boundaries=None,
        max_batch_combined_size=-1,
    ):
        """
        Args:
            batch_size:
                batch size
            drop_last:
                whether to drop the last few data that cannot form a batch (recommanded True).
            shuffle:
                whether to shuffle the data slightly (according to their length of course).
            bucket_boundaries:
                int (number of bins) or a list (containing the edges in ascending order,
                excluding two outmost boundaries).

                edge     0   1   2
                len   ___|___|___|___
            max_batch_combined_size:
                limitation on batch_size * seq_len.  -1: ignored
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.bucket_boundaries = np.array(bucket_boundaries) if bucket_boundaries is not None else None
        self.max_batch_combined_size = max_batch_combined_size
        self.dataset_lengths = None
        self.total_batches = None

    def set_bucket_boundaries(self, bucket_boundaries):
        """
        Args:
            bucket_boundaries:
                int (number of bins) or a list (containing the edges in ascending order, excluding two outmost boundaries)
                edge     0   1   2
                len   ___|___|___|___
        """
        self.bucket_boundaries = np.array(bucket_boundaries)
        # insert 0 and inf to bucket_boundaries
        self.bucket_boundaries = np.append(np.insert(self.bucket_boundaries, 0, -np.inf), np.inf)

        # assign data to bins
        if self.dataset_lengths is not None:
            self.bucket_idxs = np.digitize(self.dataset_lengths, self.bucket_boundaries)

    def set_dataset_lengths(self, dataset_lengths, global_idxs=None, bucket_boundaries=None):
        """
        Args:
            dataset_lengths:
                a list containing the length of each data
            global_idxs:
                global index if the dataset is combined and dataset_length are sub-selected from the combined dataste. None: 0:N
            bucket_boundaries:
                int (number of bins) or a list (containing the edges in ascending order, excluding two outmost boundaries)
                edge     0   1   2
                len   ___|___|___|___
        """
        self.dataset_lengths = dataset_lengths
        if isinstance(self.dataset_lengths, (list, tuple)):
            self.dataset_lengths = np.array(self.dataset_lengths)

        self.global_idxs = global_idxs  # only used when returned in a batch
        if self.global_idxs is None:
            self.global_idxs = np.arange(len(self.dataset_lengths))
        if bucket_boundaries is not None:
            self.bucket_boundaries = np.array(bucket_boundaries)
            # insert 0 and inf to bucket_boundaries
            self.bucket_boundaries = np.append(np.insert(self.bucket_boundaries, 0, -np.inf), np.inf)

        # assign data to bins
        self.bucket_idxs = np.digitize(self.dataset_lengths, self.bucket_boundaries)

        self.bucket_idx_to_data_idxs = dict()  # bucket idx -> data idx
        for i in range(len(self.bucket_idxs)):
            if self.bucket_idx_to_data_idxs.get(self.bucket_idxs[i]) is None:
                self.bucket_idx_to_data_idxs[self.bucket_idxs[i]] = []
            self.bucket_idx_to_data_idxs[self.bucket_idxs[i]].append(i)

    def _batching(self):
        # working in local index (0:N)

        # shuffle the data within each bucket separately
        if self.shuffle:
            for bid in self.bucket_idx_to_data_idxs.keys():
                np.random.shuffle(self.bucket_idx_to_data_idxs[bid])

        # divide each bucket into batches
        all_batches = []
        for bid in self.bucket_idx_to_data_idxs.keys():
            bucket_size = len(self.bucket_idx_to_data_idxs[bid])
            # num_rest = bucket_size % self.batch_size
            n_batches = bucket_size // self.batch_size
            for i in range(n_batches):
                all_batches.append(
                    self.bucket_idx_to_data_idxs[bid][(i * self.batch_size) : ((i + 1) * self.batch_size)]
                )

            if self.drop_last is False:
                rest = self.bucket_idx_to_data_idxs[bid][n_batches * self.batch_size :]
                if len(rest) > 0:
                    all_batches.append(rest)

        # check if the batch contains too many elements
        if self.max_batch_combined_size < 0:
            self.all_batches = all_batches
        else:
            # reduce the batch_size if it is too large (batch * seq_len > self.max_batch_combined_size)
            self.all_batches = all_batches
            current_idx = 0
            total_removed = 0
            while current_idx < len(self.all_batches):
                batch = self.all_batches[current_idx]
                batch_size = len(batch)
                seq_len = (self.dataset_lengths[batch]).max()
                batch_combined_size = batch_size * seq_len

                if batch_combined_size <= self.max_batch_combined_size:
                    current_idx += 1
                else:
                    # if batch_size == 1, remove the sample
                    if batch_size == 1:
                        self.all_batches[current_idx] = None
                        total_removed += 1
                        current_idx += 1
                        continue

                    # divide the batch into two part, add the new part to the end of all_batches
                    batch_size1 = batch_size // 2
                    self.all_batches[current_idx] = batch[:batch_size1]
                    self.all_batches.append(batch[batch_size1:])

            # remove [] from all_batches
            self.all_batches = [batch for batch in self.all_batches if batch is not None]
            log.debug(f"total_removed = {total_removed}")

        # shuffle the batches
        np.random.shuffle(self.all_batches)
        self.total_batches = len(self.all_batches)

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        assert self.bucket_boundaries is not None
        assert self.dataset_lengths is not None
        assert self.global_idxs is not None
        # group batches
        self._batching()
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx < self.total_batches:
            local_idxs = self.all_batches[self.batch_idx]
            gidxs = [self.global_idxs[i] for i in local_idxs]
            self.batch_idx += 1
            return gidxs
        else:
            raise StopIteration
