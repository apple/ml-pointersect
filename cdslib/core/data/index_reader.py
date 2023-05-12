#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
# This file implements a simple Scp file reader.

import json
import os
import typing as T

import numpy as np
import tgt
import torch


class IndexFileReader:
    """
    The class implements a simple reader for an index file like scp files.

    Given an index file (e.g., scp file, with or without the unique id),
    the class creates a indexed list of the file content.

    For example, if i-th line in the scp file contains "uid_abc abc.npy",

    - dset[i] returns the array contained in abc.npy.
    - dset.get_uid(i) returns uid_abc.

    Currently, we support only npy files.

    Note:
        The class is not a proper scp file reader.
        If a typical scp reader is what you are after,
        please use kaldiio.

    """

    def __init__(self, index_filename: str, with_uid: bool, root_dir: str = None):
        """
        Args:
            index_filename:
                the index file that lists all the data, one per line.
            with_uid:
                If with_uid is True, each line is composed of
                `a_unique_id_of_the_data  filename_of_the_data`.
                If with_uid is False, each line is composed of
                `filename_of_the_data`.
            root_dir:
                The root folder for the files listed in the index_filename.
                For example, if root_dir = 'a/b' and index_filename[0] = 'c/d.npy',
                the file is at 'a/b/c/d.npy'.
                If None is given, use the folder of the index_filename as root_dir.
                If '' is given, no modification will be made.
        """
        self.index_filename = index_filename
        self.with_uid = with_uid
        self.root_dir = root_dir
        if self.root_dir is None:
            self.root_dir = os.path.dirname(self.index_filename)

        # read the index file
        self.uids, self.filenames = self._read_index_file(
            index_filename=self.index_filename,
            with_uid=self.with_uid,
            root_dir=self.root_dir,
        )

    def __len__(self):
        """Returns the number of files in the index file."""
        return len(self.filenames)

    def __getitem__(self, i: int) -> T.Any:
        """Returns the content of the i-th line in the index file."""

        filename = self.filenames[i]
        if not os.path.exists(filename):
            return None

        ext = os.path.splitext(filename)[1]
        if ext.lower() == ".npy":
            out = np.load(filename)
        elif ext.lower() == ".npz":
            # tmp = np.load(filename, allow_pickle=True)  # returns a np file reader
            # out = dict(**tmp)
            # tmp.close()
            tmp = np.load(filename, allow_pickle=True)  # returns a np file reader
            out = dict()
            for key in tmp:
                out[key] = tmp[key]
            tmp.close()
        elif ext.lower() == ".textgrid":
            out = tgt.io.read_textgrid(filename)  # tgt.core.TextGrid
        elif ext.lower() in {".pt", ".pth"}:
            # pytorch dict
            out = torch.load(filename, map_location=torch.device("cpu"))
        elif ext.lower() in {".json"}:
            with open(filename, "r") as f:
                out = json.load(f)
        else:
            raise NotImplementedError

        return out

    def get_uid(self, i: int):
        """Return the uid of the i-th line in the index file.
        If with_uid was set to False, returns None."""

        return self.uids[i]

    def get_invalid_idxs(self):
        """Return indexes of which the files do not exist."""
        # check if file exists
        invalid_indexes = []
        for i in range(len(self.filenames)):
            if not os.path.exists(self.filenames[i]):
                invalid_indexes.append((i))
        return invalid_indexes

    def _read_index_file(self, index_filename: str, with_uid: bool, root_dir: str):
        """Read the index file and returns the uids and filenames."""

        assert os.path.exists(index_filename)

        uids = []
        filenames = []
        with open(index_filename, "r") as f:
            for line in f:
                line = line.strip()
                if with_uid:
                    uid, filename = line.split(sep=" ", maxsplit=1)
                else:
                    uid, filename = None, line

                # handle root_dir
                filename = os.path.join(root_dir, filename)

                uids.append(uid)
                filenames.append(filename)

        return uids, filenames
