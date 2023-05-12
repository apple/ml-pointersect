#
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
# Author: Rick Chang

import json
import logging
import os
import queue
import shutil
import time
from timeit import default_timer as timer
import typing as T

import numpy as np
import torch
import torch.multiprocessing as mp

from cdslib.core.utils.print_and_save import NumpyJsonEncoder

log = logging.getLogger(__name__)
if log.getEffectiveLevel() == logging.NOTSET:
    log.setLevel(os.environ.get("LOGLEVEL", "INFO"))


class ChunkedMemmap:
    """Implements chunked memmap.

    - Designed to use with a dataset loader (no slicing support).
    - It handles saving/loading the meta data automatically, so can be used like a normal ndarray.
    - It supports chunking of the array to improve read/write speed and disk space efficiency.

    Usage:

    - add_all_samples(samples, chunk_idxs):
      divide samples (one per row) into different numpy memmap according to chunk_idx.
    - __getitem__:
      only support reading ith sample (no slicing).  The indexing is the same as if the samples are not chunked.
    - __len__:
      return total number of samples.
    """

    def __init__(self, working_dir: str, remove_exist=False):
        """Create memmaps that store chunked of the samples.

        Args:
            working_dir:
                the directory where all the chunked memmaps are stored.
            remove_exist:
                whether or not to remove existing content in the working_dir.
                Set to False if want to read existed chunked_memmap.
        """
        self.working_dir = working_dir
        self.info_filename = os.path.join(self.working_dir, "info.json")

        if remove_exist and os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

        # read the memmaps if exist
        self._read_info()

    def _read_info(self):
        """Read the info file if existed, else set everything to None."""
        if not os.path.exists(self.info_filename):
            self.dtype = None
            self.itemsize = None  # bytes per number
            self.chunk_idxs: T.Sequence[int] = None
            self.chunk_local_idxs: T.Sequence[int] = None  # local index within each chunk memmap
            self.chunk_filename_dict: T.Dict[int, str] = None  # chunk_idx -> memmap filename
            self.chunk_offsets_dict: T.Dict[
                int, T.Sequence[int]
            ] = None  # chunk_idx -> list of sample offset in the chunk
            self.chunk_sample_shapes_dict: T.Dict[
                int, T.Sequence[T.Sequence[int]]
            ] = None  # chunk_idx -> list of sample_shapes in the chunk
            self.chunk_shape_dict: T.Dict[int, T.Tuple[int]] = None  # chunk_idx -> chunk's memmap shape
        else:
            with open(self.info_filename, "r") as f:
                info_dict = json.load(f)

                for key in [
                    "dtype",
                    "itemsize",
                    "chunk_idxs",
                    "chunk_local_idxs",
                    "chunk_offsets_dict",
                    "chunk_sample_shapes_dict",
                    "chunk_shape_dict",
                ]:
                    setattr(self, key, info_dict[key])

                # convert the key values to int
                for key in [
                    "chunk_offsets_dict",
                    "chunk_sample_shapes_dict",
                    "chunk_shape_dict",
                ]:
                    tmp_dict = dict()
                    for chunk_idx in info_dict[key]:
                        tmp_dict[int(chunk_idx)] = info_dict[key][chunk_idx]
                    setattr(self, key, tmp_dict)

                # determine memmap filenames for each chunk (chunk_idx -> filename)
                self.chunk_filename_dict = self._determine_memmap_filenames(self.working_dir, self.chunk_idxs)

    def _write_info(self):
        """Write the info to info_filename."""
        if isinstance(self.chunk_idxs, np.ndarray):
            self.chunk_idxs = self.chunk_idxs.tolist()
        if isinstance(self.chunk_local_idxs, np.ndarray):
            self.chunk_local_idxs = self.chunk_local_idxs.tolist()

        info_dict = dict()
        for key in [
            "dtype",
            "itemsize",
            "chunk_idxs",
            "chunk_local_idxs",
        ]:
            info_dict[key] = getattr(self, key)

        # make sure dict keys are str
        for key in [
            "chunk_offsets_dict",
            "chunk_sample_shapes_dict",
            "chunk_shape_dict",
        ]:
            tmp_dict = dict()
            for chunk_idx in getattr(self, key):
                tmp_dict[str(chunk_idx)] = getattr(self, key)[chunk_idx]
            info_dict[key] = tmp_dict

        folder = os.path.dirname(self.info_filename)
        os.makedirs(folder, exist_ok=True)
        with open(self.info_filename, "w") as outfile:
            json.dump(info_dict, outfile, indent=4, cls=NumpyJsonEncoder)

    def __len__(self):
        if self.chunk_idxs is None:
            return 0
        else:
            return len(self.chunk_idxs)

    def __getitem__(self, idx: int):
        """Return the ith's sample (in global index) as if calling samples[idx].

        Note:
            1. Each sample is returned in their native shape (no padding)
            2. The sample is returned as ndarray (so int's shape will be (1,))
        """
        if isinstance(idx, slice):
            # do not support slicing
            raise NotImplementedError

        if self.chunk_filename_dict is None or self.chunk_shape_dict is None or self.chunk_idxs is None:
            return None

        # get chunk idx of ith sample
        target_chunk_idx = self.chunk_idxs[idx]

        # open memmap
        memmap_filename = self.chunk_filename_dict[target_chunk_idx]
        memmap_shape = self.chunk_shape_dict[target_chunk_idx]
        sample_shape = self.chunk_sample_shapes_dict[target_chunk_idx][self.chunk_local_idxs[idx]]

        # check if memmap_shape or sample_shape contains any 0
        if np.prod(memmap_shape) == 0 or np.prod(sample_shape) == 0:
            out = np.zeros(sample_shape, dtype=self.dtype)
            return out

        offset = self.chunk_offsets_dict[target_chunk_idx][self.chunk_local_idxs[idx]] * self.itemsize  # offset in byte
        arr = np.memmap(
            memmap_filename,
            dtype=self.dtype,
            mode="r",
            offset=offset,
            shape=tuple(sample_shape),
        )
        return np.array(arr)

    def get_shape(self, idx: int) -> T.Sequence[int]:
        """Returns the shape of idx-th sample."""
        # get chunk idx of ith sample
        target_chunk_idx = self.chunk_idxs[idx]
        # get sample shape
        sample_shape = self.chunk_sample_shapes_dict[target_chunk_idx][self.chunk_local_idxs[idx]]
        return sample_shape

    def add_all_samples(
        self,
        samples: T.Union[T.Sequence[np.ndarray], T.Iterable[np.ndarray]],
        chunk_idxs: T.Sequence[int] = None,
        dtype="float32",
        num_workers=0,
        remove_exist=False,
    ):
        """Create memmaps that store chunked of the samples. It will overwrite the working_dir.

        Args:
            samples:
                a sequence of samples of different shapes. samples[i] returns the ith sample,
                which is a np.ndarray.  Note that samples can be an object with __len__ and __getitem__ implemented.
            chunk_idxs:
                chunk index for each sample.  None: store all samples into one chunk.
            dtype:
                The data-type used to interpret the file contents.
            num_workers:
                number of workers to save the memmaps. It will affect the memory usage.
            remove_exist:
                whether or not to remove existing content in the folder. Note that
                we need the working_dir folder to be empty, so if remove_exist is False and
                the working_dir folder is not empty, we will throw an Runtime exception.

        Note:
            The memory requirement of the function is O(num_workers * num_samples_in_a_chunk * byte_per_sample).
            So divide the samples into more chunks if the physical memory is not enough.
        """
        # check if the working_dir is empty or clear the working_dir
        if os.path.exists(self.working_dir):
            if remove_exist:
                shutil.rmtree(self.working_dir)
            elif len(os.listdir(self.working_dir)) > 0:
                raise RuntimeError(f"working_dir {self.working_dir} is not empty.")
        os.makedirs(self.working_dir, exist_ok=True)

        self.dtype = dtype
        self.itemsize = np.dtype(self.dtype).itemsize  # bytes per number

        if chunk_idxs is None:
            chunk_idxs = [0] * len(samples)
        self.chunk_idxs = chunk_idxs
        assert len(self.chunk_idxs) == len(samples)

        # determine memmap filenames for each chunk (chunk_idx -> filename)
        self.chunk_filename_dict = self._determine_memmap_filenames(self.working_dir, self.chunk_idxs)

        # divide and save samples to memmaps
        target_chunk_idxs = list(self.chunk_filename_dict.keys())
        num_workers = min(num_workers, len(target_chunk_idxs))
        return_queue = mp.Queue(maxsize=len(target_chunk_idxs) + 5)
        if num_workers == 0:
            log.debug("saving the chunk one by one..")
            self._save_to_memmap(
                target_chunk_idxs,
                samples,
                self.chunk_idxs,
                self.chunk_filename_dict,
                self.dtype,
                return_queue,
            )
        else:
            # use num_workers to save
            n = len(target_chunk_idxs) // num_workers
            processes = []
            for i in range(num_workers):
                if i < num_workers - 1:
                    sub_target_chunk_idxs = target_chunk_idxs[i * n : (i + 1) * n]
                else:
                    sub_target_chunk_idxs = target_chunk_idxs[i * n :]
                p = mp.Process(
                    target=self._save_to_memmap,
                    args=(
                        sub_target_chunk_idxs,
                        samples,
                        self.chunk_idxs,
                        self.chunk_filename_dict,
                        self.dtype,
                        return_queue,
                    ),
                )
                p.daemon = True
                p.start()
                processes.append(p)

        # gather information from the return_queue
        gather_dict = dict()
        stime = timer()
        while len(gather_dict) < len(target_chunk_idxs):
            try:
                rdict = return_queue.get(timeout=0.1)  # block for 0.1 secs
                gather_dict[rdict["target_chunk_idx"]] = rdict  # chunk_idx -> return vals
            except queue.Empty:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        f"({timer() - stime} secs) finished chunks: {len(gather_dict)} / {len(target_chunk_idxs)}"
                    )
                    if num_workers > 0:
                        debug_txt = "\t"
                        debug_txt += "not finished processes (if any): "
                        finished = 0
                        for p_idx in range(len(processes)):
                            p = processes[p_idx]
                            if p.is_alive():
                                debug_txt += f"{p_idx} "
                            else:
                                finished += 1
                        log.debug(debug_txt)
                        log.debug(f"\t finished processes: {finished} / {len(processes)}")
                    time.sleep(1)  # sleep for 1 sec so it does not continuously printing debug messages

        # make sure the processes ends
        if num_workers > 0:
            for p in processes:
                p.join()

        # build chunk_local_idxs
        self.chunk_local_idxs = [0] * len(
            samples
        )  # global index -> local index in chunk_offsets and chunk_sample_shapes
        for chunk_idx in gather_dict:
            sample_idxs = gather_dict[chunk_idx]["sample_idxs"]
            for i in range(len(sample_idxs)):
                self.chunk_local_idxs[sample_idxs[i]] = i

        # save chunk_offsets and chunk_sample_shapes
        self.chunk_offsets_dict: T.Dict[
            int, T.Sequence[int]
        ] = dict()  # chunk_idx -> list of sample offset in the chunk
        self.chunk_sample_shapes_dict: T.Dict[
            int, T.Sequence[T.Sequence[int]]
        ] = dict()  # chunk_idx -> list of sample_shapes in the chunk
        self.chunk_shape_dict: T.Dict[int, T.Tuple[int]] = dict()  # chunk_idx -> chunk's memmap shape
        for chunk_idx in gather_dict:
            self.chunk_offsets_dict[chunk_idx] = gather_dict[chunk_idx]["chunk_offsets"]
            self.chunk_sample_shapes_dict[chunk_idx] = gather_dict[chunk_idx]["chunk_sample_shapes"]
            self.chunk_shape_dict[chunk_idx] = gather_dict[chunk_idx]["memmap_shape"]

        # save info to info_filename
        self._write_info()

        log.debug("finished saving all samples")

    @staticmethod
    def _determine_memmap_filenames(working_dir: str, chunk_idxs: T.Sequence[int]):
        """Determine the filenames of the memmaps."""
        # determine how many chunks we need
        unique_chunk_idxs = set(chunk_idxs)

        # determine chunk filename
        chunk_filename_dict = dict()
        for chunk_idx in unique_chunk_idxs:
            chunk_filename_dict[chunk_idx] = os.path.join(working_dir, f"chunk_{chunk_idx}.memmap")

        return chunk_filename_dict

    @staticmethod
    def _save_to_memmap(
        target_chunk_idxs,
        samples,
        chunk_idxs,
        chunk_filename_dict,
        dtype,
        return_queue: mp.Queue,
    ):
        """Save all the samples of the target_chunk_idx to the memmap.

        Overview:

        1. Read all samples in the target chunk into a list and save the shape of each samples in chunk_sample_shapes.
        2. Vectorize each sample and determine the size of the memmap required. Also gather the offset of each sample.
        3. Concatenate the samples into the memmap

        Returns:
            chunk_sample_shapes:
            chunk_offsets:
            memmap_shape:
        """
        for tidx, target_chunk_idx in enumerate(target_chunk_idxs):
            log.debug(f"start saving memmap chunk: {target_chunk_idx}")
            stime = timer()

            # gather index of samples with chunk_idx == target_chunk_idx
            if isinstance(chunk_idxs, (list, tuple)):
                chunk_idxs = np.array(chunk_idxs)
            sample_idxs = np.argwhere(chunk_idxs == target_chunk_idx)[:, 0]  # (N,)

            # save the samples to the list and store their shapes
            chunk_sample_shapes = []
            chunk_samples = []
            for idx in sample_idxs:
                sample = samples[idx]
                if isinstance(sample, torch.Tensor):
                    sample = sample.detach().cpu().numpy()
                elif isinstance(sample, np.ndarray):
                    pass
                elif isinstance(sample, T.Sequence):
                    sample = np.array(sample)
                else:
                    sample = np.array([sample])  # make sure int/float sample has a size = (1,)
                assert np.prod(sample.shape) > 0
                chunk_sample_shapes.append(sample.shape)
                chunk_samples.append(sample)

            # build the chunk local offset using the chunk_sample_shapes
            chunk_offsets = []
            current_offset = 0
            for i in range(len(chunk_sample_shapes)):
                chunk_offsets.append(current_offset)
                current_offset += np.prod(chunk_sample_shapes[i])
            memmap_size = current_offset

            # create memmap
            if memmap_size == 0:
                # do nothing, memmap cannot have shape = 0
                pass
            else:
                arr = np.memmap(
                    chunk_filename_dict[target_chunk_idx],
                    dtype=dtype,
                    mode="w+",
                    shape=(memmap_size,),
                )
                for i in range(len(chunk_samples)):
                    sample = chunk_samples[i]  # it is important to not to index samples again (speed)
                    if sample[0] is not None:
                        vec = sample.flatten().astype(dtype=dtype)
                    else:
                        # None, set to nan
                        vec = np.ones(sample.size, dtype=dtype) * np.nan
                    arr[chunk_offsets[i] : (chunk_offsets[i] + len(vec))] = vec
                # make sure arr is written to disk
                arr.flush()
                del arr

            log.debug(f"finished chunk {target_chunk_idx}, took {timer() - stime} secs")

            # put the result info to return_queue
            return_dict = {
                "target_chunk_idx": target_chunk_idx,
                "chunk_sample_shapes": chunk_sample_shapes,
                "chunk_offsets": chunk_offsets,
                "memmap_shape": (memmap_size,),
                "sample_idxs": sample_idxs,
            }
            return_queue.put(return_dict)
