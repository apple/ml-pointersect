#
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
# Author: Rick Chang

import numpy as np


class SortSampler:
    """
    The class assembles batches by sorting the sequence length of samples.
    It returns indices of the data in the dataset within a batch.
    """

    def __init__(self, batch_size, max_batch_combined_size=-1):
        """
        Args:
            batch_size:
                batch size
            max_batch_combined_size:
                limitation on batch_size * seq_len.  -1: ignored
        """
        self.batch_size = batch_size
        self.max_batch_combined_size = max_batch_combined_size
        self.dataset_lengths = None
        self.total_batches = None

    def set_dataset_lengths(self, dataset_lengths, global_idxs=None):
        """
        Args:
            dataset_lengths:
                a list containing the length of each data
            global_idxs:
                global index if the dataset is combined and dataset_length are sub-selected
                from the combined dataste. None: 0:N
        """
        self.dataset_lengths = dataset_lengths
        if isinstance(self.dataset_lengths, (list, tuple)):
            self.dataset_lengths = np.array(self.dataset_lengths)

        self.global_idxs = global_idxs  # only used when returned in a batch
        if self.global_idxs is None:
            self.global_idxs = np.arange(len(self.dataset_lengths))

    def _batching(self):
        # working in local index (0:N)
        total_samples = len(self.dataset_lengths)

        permute_idxs = np.random.permutation(total_samples)
        self.dataset_lengths = [self.dataset_lengths[i] for i in permute_idxs]
        # use the same permute_idxs to permute global idxs (important)
        self.global_idxs = [self.global_idxs[i] for i in permute_idxs]

        # sort dataset length from short to long
        sort_idxs = np.argsort(self.dataset_lengths, kind="stable")

        # divide each bucket into batches
        all_batches = []
        index = 0  # index of sort_idxs
        while index < total_samples:
            current_batch_size = min(total_samples - index, self.batch_size)

            # Get the biggest sequence length on the batch (last one in batch, since sorted ascendingly)
            max_seq_len = self.dataset_lengths[sort_idxs[index + current_batch_size - 1]]

            # Adaptively shrink the batch size if the sequence is too long to fit the memory
            if self.max_batch_combined_size > 0:
                batch_combined_size = current_batch_size * max_seq_len
                while batch_combined_size > self.max_batch_combined_size:
                    if current_batch_size == 1:
                        break
                    current_batch_size = current_batch_size // 2
                    max_seq_len = self.dataset_lengths[sort_idxs[index + current_batch_size - 1]]
                    batch_combined_size = current_batch_size * max_seq_len
                if batch_combined_size > self.max_batch_combined_size:
                    # Even single sample won't fit the memory. Skip the input.
                    continue

            batch = sort_idxs[index : index + current_batch_size]
            all_batches.append(batch)
            index += current_batch_size

        self.all_batches = all_batches
        self.total_batches = len(self.all_batches)

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        assert self.dataset_lengths is not None
        assert self.global_idxs is not None
        # since we randomly permute the dataset_length and the sort is stable,
        # the result may be different every epoch
        self._batching()
        # shuffle the batches
        np.random.shuffle(self.all_batches)
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
