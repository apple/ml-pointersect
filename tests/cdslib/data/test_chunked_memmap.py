#
# Copyright (C) 2020 Apple Inc. All rights reserved.
#

import numpy as np
import unittest
from cdslib.core.data.chunked_memmap import ChunkedMemmap
from timeit import default_timer as timer
from torch.utils.data import Dataset, DataLoader
import os
import shutil
from tqdm import tqdm
import math
import logging

# define a tmp pytorch dataset
class TmpDataset(Dataset):
    def __init__(self, wdir):
        self.wdir = wdir
        self.mmap = ChunkedMemmap(self.wdir, remove_exist=False)

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, idx):
        return np.array(self.mmap[idx])


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.path.exists('tmp_test_chunked_memmap'):
            raise RuntimeError('make sure tmp_test_chunked_memmap is empty to run the test')

    def test_write(self):

        # create random array
        samples_shape = [20, 10, 4]
        working_dir = 'tmp_test_chunked_memmap'
        samples = np.random.randn(*samples_shape).astype('float32')

        # determine chunk_idxs
        total_chunks = 3
        chunk_idxs = np.random.randint(total_chunks, size=samples_shape[0])

        # create chunked_memmap
        chunked_memmap = ChunkedMemmap(working_dir, remove_exist=True)
        chunked_memmap.add_all_samples(samples, chunk_idxs, dtype='float32')

        # check if the same
        arr = np.zeros(samples_shape)
        for i in range(samples_shape[0]):
            arr[i] = chunked_memmap[i]

        assert np.allclose(samples, arr)


    def test_multithread_write(self):

        # create random array
        samples_shape = [2000, 10, 4]
        working_dir = 'tmp_test_chunked_memmap'
        samples = np.random.randn(*samples_shape).astype('float32')
        num_workers = 5

        # determine chunk_idxs
        total_chunks = 3
        chunk_idxs = np.random.randint(total_chunks, size=samples_shape[0])

        # create chunked_memmap
        chunked_memmap = ChunkedMemmap(working_dir, remove_exist=True)
        chunked_memmap.add_all_samples(samples, chunk_idxs, dtype='float32', num_workers=num_workers)

        # check if the same
        arr = np.zeros(samples_shape)
        for i in range(samples_shape[0]):
            arr[i] = chunked_memmap[i]

        assert np.allclose(samples, arr)


    def test_load_existing(self):

        # create random array
        samples_shape = [1000, 10, 4]
        working_dir = 'tmp_test_chunked_memmap'
        samples = np.random.randn(*samples_shape).astype('float32')

        # determine chunk_idxs
        total_chunks = 3
        chunk_idxs = np.random.randint(total_chunks, size=samples_shape[0])

        # create chunked_memmap
        chunked_memmap = ChunkedMemmap(working_dir, remove_exist=True)
        chunked_memmap.add_all_samples(samples, chunk_idxs, dtype='float32')

        # check if the same
        arr = np.zeros(samples_shape)
        for i in range(samples_shape[0]):
            arr[i] = chunked_memmap[i]
        assert np.allclose(samples, arr)

        # read the just-written chunked_memmap
        chunked_memmap2 = ChunkedMemmap(working_dir, remove_exist=False)
        arr2 = np.zeros(samples_shape)
        stime = timer()
        for i in range(samples_shape[0]):
            arr2[i] = chunked_memmap2[i]
        print(f'read entire arr takes {timer() - stime} secs')
        assert np.allclose(samples, arr2)


    def test_pytorch_dataloader(self):

        # create random array
        samples_shape = [10000, 1000, 4]
        working_dir = 'tmp_test_chunked_memmap'
        samples = np.random.randn(*samples_shape).astype('float32')

        # determine chunk_idxs
        total_chunks = math.ceil(np.prod(samples_shape) / 10000000)
        print(f'total_chunks = {total_chunks}')
        chunk_idxs = np.random.randint(total_chunks, size=samples_shape[0])

        # determine valid_len
        squeeze_dim = 1
        valid_lens = np.random.randint(samples_shape[squeeze_dim], size=samples_shape[0])

        # make sure not-valid region = 0 in sample
        tmp_samples = np.zeros_like(samples)
        for i in range(samples_shape[0]):
            slice_idxs = [slice(None)] * len(samples_shape)
            slice_idxs[0] = i
            slice_idxs[squeeze_dim] = slice(valid_lens[i])
            tmp_samples[tuple(slice_idxs)] = samples[tuple(slice_idxs)]
        samples = tmp_samples

        # create chunked_memmap
        chunked_memmap = ChunkedMemmap(working_dir, remove_exist=True)
        chunked_memmap.add_all_samples(samples, chunk_idxs, dtype='float32', num_workers=8)

        # check if the same
        arr = np.zeros(samples_shape)
        for i in range(samples_shape[0]):
            arr[i] = chunked_memmap[i]
        assert np.allclose(samples, arr)

        # create a tmp pytorch dataset
        dataset = TmpDataset(wdir=working_dir)
        batch_size = 128
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, )
        arr2 = np.zeros(samples_shape)
        stime = timer()
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            arr2[batch_idx * batch_size:(batch_idx + 1) * batch_size] = batch.numpy()
        etime = timer()
        print(f'read entire arr takes {etime - stime} secs, or {(etime - stime) / len(arr2)} per sample')
        assert np.allclose(samples, arr2)


    def test_write_squeeze_sample_reader(self):

        total_samples = 10
        max_seq_len = 20
        dim = 4
        squeeze_dim = 1
        working_dir = 'tmp_test_chunked_memmap'

        rng = np.random.RandomState(12345)

        # create random samples of random length
        samples_list = []
        valid_lens = []
        for i in range(total_samples):
            rlen = rng.randint(max_seq_len-1) + 1
            sample = rng.randn(rlen, dim)
            samples_list.append(sample)
            valid_lens.append(rlen)

        # define a SampleReader
        class TMPReader:
            def __init__(self, samples_list):
                self.samples_list = samples_list


            def __len__(self):
                return len(self.samples_list)


            def __getitem__(self, i):
                return self.samples_list[i]

        samples = TMPReader(samples_list)

        # determine chunk_idxs
        total_chunks = 3
        chunk_idxs = rng.randint(total_chunks, size=total_samples)
        # make sure each chunk at least get one
        for i in range(total_chunks):
            chunk_idxs[i] = i

        # create chunked_memmap
        chunked_memmap = ChunkedMemmap(working_dir, remove_exist=True)
        chunked_memmap.add_all_samples(samples, chunk_idxs, dtype='float32')

        # check if the same
        for i in range(total_samples):
            arr = chunked_memmap[i]
            assert np.allclose(samples[i], arr[:valid_lens[i]])

        # read the just-written chunked_memmap
        chunked_memmap2 = ChunkedMemmap(working_dir, remove_exist=False)
        assert len(chunked_memmap2) == total_samples
        for i in range(total_samples):
            arr = chunked_memmap[i]
            assert np.allclose(samples[i], arr[:valid_lens[i]])

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('tmp_test_chunked_memmap'):
            shutil.rmtree('tmp_test_chunked_memmap')

if __name__ == '__main__':
    unittest.main()

