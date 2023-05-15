#
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
# Author: Rick Chang
# The file implements BatchPreparer, which prepares batches for training/validation/testing.

import logging
import math
import os
import typing as T

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .bucket_sampler import BucketSampler
from .sort_sampler import SortSampler

log = logging.getLogger(__name__)
if log.getEffectiveLevel() == logging.NOTSET:
    log.setLevel(os.environ.get("LOGLEVEL", "INFO"))


class BatchPreparer:
    """
    The batch preparer works as follows:

    - First adds datasets before iterator starts.
    - When creating the iterator (beginning of a epoch):
        1. It determines the samples to used in the iteration based on dataset_ratios.
        2. It groups samples of similar length into batch (using bucket_sampler or sort_sampler).
    - Every iteration:
        The samples are loaded in parallel using pytorch's dataloader.

    Data flow:

    - Each dataset is stored in a pytorch's Dataset (with an additional method 'get_seq_lens' implemented).

        ex: seq_lens = dataset.get_seq_lens()

        - seq_lens is a list of all samples stored in the dataset.
        - seq_lens[i] = length of dataset[i]

    - The datasets are concatenated to a single dataset by torch.utils.data.ConcatDataset for pytorch's dataloader.
    """

    def __init__(
        self,
        batch_size,
        collate_fn,
        drop_last=False,
        num_workers=0,
        batch_wrt_length=True,
        shuffle=True,
        max_total_samples_per_epoch=-1,
        max_batch_combined_size=-1,
        batch_sampler_type="sort",
    ):
        """
        Args:
            batch_size:
                typical batch_size
            collate_fn:
                collate function used to combine samples from dataset
            drop_last:
                whether to drop the rest of the data that cannot form a batch
            num_workers:
                number of loading threads to use
            batch_wrt_length:
                whether to batch samples of similar sequence lengths
            shuffle:
                whether to shuffle data within each bucket
            max_total_samples_per_epoch:
                maximum total number of samples per epoch. -1: ignored
            max_batch_combined_size:
                limitation on batch_size * seq_len.  -1: ignored
            batch_sampler_type:
                type of the batch sampler [bucket | sort]
        """

        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.batch_wrt_length = batch_wrt_length
        self.shuffle = shuffle
        self.max_total_samples_per_epoch = max_total_samples_per_epoch
        self.max_batch_combined_size = max_batch_combined_size
        self.batch_sampler_type = batch_sampler_type
        assert self.batch_sampler_type in {"bucket", "sort"}
        self.ready = False  # record the state whether combined_datasets is ready to be used

        self.datasets: T.List[Dataset] = []  # contains each of the dataset
        self.dataset_ratios: T.List[
            float
        ] = []  # contains the ratio of the samples in each dataset to be used in every epoch
        self.combined_dataset: ConcatDataset = None
        self.combined_seq_lens: T.List[int] = None
        self.selected_global_idxs: T.List[int] = None  # index (of the combined dataset) that is chosen
        if self.batch_sampler_type == "bucket":
            self.batch_sampler = BucketSampler(
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                shuffle=self.shuffle,
                bucket_boundaries=None,
                max_batch_combined_size=max_batch_combined_size,
            )
        elif self.batch_sampler_type == "sort":
            self.batch_sampler = SortSampler(
                batch_size=self.batch_size,
                max_batch_combined_size=max_batch_combined_size,
            )
        else:
            raise NotImplementedError

        self.dataloader: DataLoader = None  # pytorch dataloader
        self.dataloader_iter = None

    def add_dataset(self, dataset: Dataset, ratio=1.0, max_samples_per_epoch=None):
        """Read the samples from the dataset.

        Args:
            dataset:
                dataset to be added
            ratio:
                ratio of the samples in the dataset to be used every epoch
            max_samples_per_epoch:
                max number of samples to be used per epoch. None: ignored
                It will be used to constrain ratio.
                The number of samples per epoch = min(ratio * len(dataset), max_samples_per_epoch)
        """

        # the dataset should have get_seq_lens defined
        assert hasattr(dataset, "get_seq_lens") and callable(getattr(dataset, "get_seq_lens"))

        # add to dataset list
        self.datasets.append(dataset)
        if max_samples_per_epoch is None or max_samples_per_epoch < 0:
            self.dataset_ratios.append(ratio)
        else:
            r = min(ratio, float(max_samples_per_epoch) / len(dataset))
            self.dataset_ratios.append(r)

        # set ready to be False
        self.ready = False

    def _combine_datasets(self):
        if self.ready:
            return

        # gather sequence length
        self.combined_seq_lens = []
        for dset in self.datasets:
            seq_lens = dset.get_seq_lens()  # (N,)
            self.combined_seq_lens.extend(seq_lens)
        self.combined_seq_lens = np.array(self.combined_seq_lens, dtype=np.float64)

        # add a small perturbation in case all samples have the same length
        self.combined_seq_lens += np.random.rand(len(self.combined_seq_lens)) - 0.5

        # combine datasets
        self.combined_dataset = ConcatDataset(self.datasets)

        if self.batch_sampler_type == "bucket":
            # determine the bucket boundaries
            log.info("computing histogram")
            if self.batch_wrt_length:
                bucket_edges = np.histogram_bin_edges(self.combined_seq_lens, bins=10)[2:-2]
                self.batch_sampler.set_bucket_boundaries(bucket_edges)
            else:
                self.batch_sampler.set_bucket_boundaries([])  # 1 bucket for all

        # create pytorch dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.combined_dataset,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        # check if 0 <= ratios <= 1
        for ratio in self.dataset_ratios:
            assert 0 <= ratio <= 1

        # test if number of samples >= 1
        count = 0.0
        for i in range(len(self.datasets)):
            count += len(self.datasets[i]) * self.dataset_ratios[i]
        log.info(f"Total samples per epoch (pre-adjustment) = {math.floor(count)}")
        assert count >= 1

        # adjust the dataset ratios in case selected samples > max_total_samples_per_epoch
        if self.max_total_samples_per_epoch >= 0 and count > self.max_total_samples_per_epoch:
            multiplier = self.max_total_samples_per_epoch / count
            self.dataset_ratios = [r * multiplier for r in self.dataset_ratios]
            log.info(
                f"Adjusted dataset ratios by {multiplier} so total samples per epoch = {self.max_total_samples_per_epoch}"
            )

        self.ready = True

    def get_total_samples(self, multiplied_with_ratio: bool) -> int:
        """Return the total number of samples (multiplied with their ratios)."""
        count = 0
        for i in range(len(self.datasets)):
            if multiplied_with_ratio:
                count += len(self.datasets[i]) * self.dataset_ratios[i]
            else:
                count += len(self.datasets[i])
        return math.ceil(count)

    def __len__(self):
        """Return an estimation of the number of batches per epoch."""
        if self.batch_sampler.total_batches is not None:
            return self.batch_sampler.total_batches  # this is exact
        elif self.selected_global_idxs is not None:
            return math.ceil(
                len(self.selected_global_idxs) / self.batch_size
            )  # this is an estimation (due to max_combined_batch_size)
        else:
            total_samples = self.get_total_samples(multiplied_with_ratio=True)
            return math.ceil(total_samples / self.batch_size)  # this is an estimation (due to max_combined_batch_size)

    def __str__(self):
        return (
            f"Num datasets: {len(self.datasets)}\n"
            f"Total samples (without ratio): {self.get_total_samples(multiplied_with_ratio=False)}\n"
            f"Total samples (with ratio): {self.get_total_samples(multiplied_with_ratio=True)}\n"
            f'Individual size: {"  ".join([str(len(dset)) for dset in self.datasets])}\n'
            f'Individual ratio: {"  ".join([str(r) for r in self.dataset_ratios])}\n'
            f"Max total samples per epoch: {self.max_total_samples_per_epoch}\n"
            f"Batch size: {self.batch_size}\n"
            f"Num batches: {len(self)}\n"
            f"Shuffle: {self.shuffle}\n"
            f"Drop last: {self.drop_last}\n"
            f"Batch wrt length: {self.batch_wrt_length}"
        )

    def __iter__(self):
        """
        - randomly select sample indexes to be used in the epoch
        - reset the batch_sampler
        :return:
        """
        if not self.ready:
            self._combine_datasets()

        # randomly select samples based on ratio
        self.selected_global_idxs = None
        selected_idxs = []
        current_base_idx = 0
        for didx in range(len(self.datasets)):
            dset = self.datasets[didx]
            ratio = self.dataset_ratios[didx]
            num_to_select = int(np.ceil(len(dset) * ratio))
            ridxs = (
                np.random.permutation(len(dset))[:num_to_select] + current_base_idx
            )  # add to translate to global index
            selected_idxs.append(ridxs)
            current_base_idx += len(dset)
        selected_idxs = np.concatenate(selected_idxs, axis=0)
        self.selected_global_idxs = selected_idxs

        # gather seq_lens of the selected samples
        seq_lens = self.combined_seq_lens[selected_idxs]

        # reset batch_sampler to reflect current selection (provide global idx so the sampler returns global indexes)
        self.batch_sampler.set_dataset_lengths(seq_lens, global_idxs=selected_idxs)

        # reset the iterator of the datalaoder
        self.dataloader_iter = iter(self.dataloader)

        # # debug:
        # total_samples = sum([len(dset) for dset in self.datasets])
        # result_idxs = np.concatenate([b for b in self.bucket_sampler.all_batches])
        # assert len(result_idxs) == total_samples
        # assert len(np.unique(result_idxs)) == total_samples

        return self

    def __next__(self):
        # logging.debug(f"before next(dataloader_iter): {psutil.virtual_memory().available / (1024 ** 2)} MB")
        batch_data = next(self.dataloader_iter)
        # logging.debug(f"after next(dataloader_iter): {psutil.virtual_memory().available / (1024 ** 2)} MB")
        return batch_data  # output defined by the collate_fn
