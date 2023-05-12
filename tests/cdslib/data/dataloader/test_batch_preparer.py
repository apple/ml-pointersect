#
# Copyright (C) 2020 Apple Inc. All rights reserved.
#

import unittest
from cdslib.core.data.dataloader.batch_preparer import BatchPreparer
import os
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import typing as T
import math


class TMPDataset(Dataset):
    def __init__(self, arr: np.ndarray):
        self.arr = arr


    def __len__(self):
        return len(self.arr)


    def __getitem__(self, i):
        return self.arr[i]


    def get_seq_lens(self):
        return [self.arr[i].shape[1] for i in range(len(self.arr))]


class TMPCollate:
    def __call__(self, input_list):
        batch_size = len(input_list)
        seq_lens = [s.shape[0] for s in input_list]
        dim = input_list[0].shape[1]
        out = np.zeros((batch_size, max(seq_lens), dim), dtype='int32')  # pad with 0, int32 to be free of precision
        for i in range(batch_size):
            out[i, :seq_lens[i]] = input_list[i]
        return np.concatenate(input_list, axis=0)


class MyTestCase(unittest.TestCase):

    @staticmethod
    def _create_dataset(total_samples, seq_len, dim, val=None):
        """
        Return a pytorch dataset instance, which contains (total_samples, dim).
        :param val: constant fill in value. None: random array
        """
        if val is None:
            arr = np.random.randn(total_samples, seq_len, dim)
        else:
            arr = np.ones((total_samples, seq_len, dim)) * val
        arr = arr.astype(dtype='int32')
        return TMPDataset(arr)


    def _test(self,
              datasets: T.List[TMPDataset],
              dataset_ratios: T.List[float],
              dataset_max_samples: T.List[int],
              batch_size=10,
              num_workers=0,
              batch_wrt_length=True,
              max_total_samples_per_epoch=-1,
              max_batch_combined_size=-1,
              batch_sampler_type='sort',
              ):
        """
        test plan:
        - given created random datasets of different sizes
        - create batch_preparer of different dataset ratios
        - check the sum of the batches returned by the batch_preparer and the sum of the datasets (possibly weighted)
        """

        # create batch_preparer
        batch_preparer = BatchPreparer(
            batch_size=batch_size,
            collate_fn=TMPCollate(),
            num_workers=num_workers,
            batch_wrt_length=batch_wrt_length,
            shuffle=True,
            max_total_samples_per_epoch=max_total_samples_per_epoch,
            max_batch_combined_size=max_batch_combined_size,
            batch_sampler_type=batch_sampler_type,
        )
        # add datasets into batch_preparer
        for i in range(len(datasets)):
            batch_preparer.add_dataset(datasets[i],
                                       ratio=dataset_ratios[i],
                                       max_samples_per_epoch=dataset_max_samples[i])
        # test
        # collect the batches
        batches = []
        for i, batch in enumerate(batch_preparer):
            batches.append(batch)

        # if all ratios are 1.0, we check if the sum is correct
        if np.allclose(dataset_ratios, 1.0):
            total_sum = np.sum([b.sum() for b in batches])
            total_sum_gt = 0
            for dset in datasets:
                for i in range(len(dset)):
                    total_sum_gt += np.sum(dset[i])
            assert total_sum == total_sum_gt
        else:
            # make sure the dataset contains constant values
            # check weight sum with ratio
            total_sum = np.sum([b.sum() for b in batches])
            total_sum_gt = 0
            for didx in range(len(datasets)):
                dset = datasets[didx]
                if dataset_max_samples[didx] == -1:
                    dlen = len(dset)
                else:
                    dlen = dataset_max_samples[didx]
                if dlen > len(dset) * dataset_ratios[didx]:
                    dlen = math.ceil(len(dset) * dataset_ratios[didx])
                for i in range(dlen):
                    total_sum_gt += np.sum(dset[i])
            assert total_sum == total_sum_gt


    def test_1(self):
        """
        single dataset
        """
        datasets = []
        dataset_ratios = []
        dataset_max_samples = []

        # 1
        dset = self._create_dataset(total_samples=100, seq_len=20, dim=3, val=None)
        ratio = 1.0
        max_sample = -1
        datasets.append(dset)
        dataset_ratios.append(ratio)
        dataset_max_samples.append(max_sample)

        self._test(datasets, dataset_ratios, dataset_max_samples)


    def test_2(self):
        """
        two datasets of different lengths, with ratio = 1
        """
        datasets = []
        dataset_ratios = []
        dataset_max_samples = []

        # 1
        dset = self._create_dataset(total_samples=100, seq_len=20, dim=3, val=None)
        ratio = 1.0
        max_sample = -1
        datasets.append(dset)
        dataset_ratios.append(ratio)
        dataset_max_samples.append(max_sample)

        # 2
        dset = self._create_dataset(total_samples=200, seq_len=30, dim=3, val=None)
        ratio = 1.0
        max_sample = -1
        datasets.append(dset)
        dataset_ratios.append(ratio)
        dataset_max_samples.append(max_sample)

        self._test(datasets, dataset_ratios, dataset_max_samples)


    def test_3(self):
        """
        two datasets of different lengths, with ratio = 0.3 and 0.5
        """
        datasets = []
        dataset_ratios = []
        dataset_max_samples = []

        # 1
        dset = self._create_dataset(total_samples=100, seq_len=20, dim=3, val=1)
        ratio = 0.3
        max_sample = -1
        datasets.append(dset)
        dataset_ratios.append(ratio)
        dataset_max_samples.append(max_sample)

        # 2
        dset = self._create_dataset(total_samples=200, seq_len=30, dim=3, val=2)
        ratio = 0.5
        max_sample = -1
        datasets.append(dset)
        dataset_ratios.append(ratio)
        dataset_max_samples.append(max_sample)

        self._test(datasets, dataset_ratios, dataset_max_samples)


    def test_4(self):
        """
        two datasets of different lengths, with ratio = 0.3 and 0.5, max_sample smaller than len(dataset) * ratio
        """
        datasets = []
        dataset_ratios = []
        dataset_max_samples = []

        # 1
        dset = self._create_dataset(total_samples=100, seq_len=20, dim=3, val=1)
        ratio = 0.3
        max_sample = int(0.2 * 100)
        datasets.append(dset)
        dataset_ratios.append(ratio)
        dataset_max_samples.append(max_sample)

        # 2
        dset = self._create_dataset(total_samples=200, seq_len=30, dim=3, val=2)
        ratio = 0.5
        max_sample = int(0.3 * 200)
        datasets.append(dset)
        dataset_ratios.append(ratio)
        dataset_max_samples.append(max_sample)

        self._test(datasets, dataset_ratios, dataset_max_samples)


if __name__ == '__main__':
    unittest.main()
