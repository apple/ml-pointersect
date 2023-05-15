#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

import json
import logging
import os
import shlex
import shutil
import subprocess
import typing as T

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.tensorboard import SummaryWriter


class NumpyJsonEncoder(json.JSONEncoder):
    """Custom encoder for saving numpy data types into a json file.

    Json only supports saving native python types like int, list, dict, etc.
    Therefore it raises exceptions if passed a numpy-typed number or array.
    The encoder translates numpy objects to python native objects automatically.

    Usage:
        When dumping a dict to json, use:
        json.dump(some_dict, filename, cls=NumpyJsonEncoder)
    """

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()

        if isinstance(
                obj,
                (
                        np.int_,
                        np.intc,
                        np.intp,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                ),
        ):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def print1D(arr: T.Union[torch.Tensor, np.ndarray, T.Sequence], num_per_line=-1, format=">6.3f") -> bool:
    """Print a 1D array.

    The function automatically breaks lines when printing a long 1D array.

    Args:
        arr:
            The 1D array to print.
        num_per_line:
            Number of elements to print per line. -1: print entire array in one line.
        format:
            The format to print each element. '>{spacing}.{precision}f'
            For example, if want right-aligned, a spacing of 6, and a precision of 3,
            set format to be '>6.3f'.

    Returns:
        Whether the cursor is at the beginning of a line.
    """
    if isinstance(arr, (torch.Tensor, np.ndarray)):
        n = arr.shape[0]
    else:
        n = len(arr)

    new_line = True
    current = 0
    for j in range(n):
        print(f"{arr[j]:{format}} ", end="")
        new_line = False
        current += 1
        if num_per_line >= 0 and current % num_per_line == 0:
            print("")
            current = 0
            new_line = True
    if new_line is False:
        print("")
        new_line = True
    return new_line


def print2D(arr: T.Union[torch.Tensor, np.ndarray, T.Sequence], num_per_line=-1, format=">6.3f") -> bool:
    """Print a 2D matrix.

    The function automatically breaks lines when printing a large 2D array.

    Args:
        arr:
            The matrix to print.
        num_per_line:
            Number of elements to print per line. -1: print entire row in one line.
        format:
            The format to print each element. '>{spacing}.{precision}f'
            For example, if want right-aligned, a spacing of 6, and a precision of 3,
            set format to be '>6.3f'.

    Returns:
        Whether the cursor is at the beginning of a line.
    """
    if isinstance(arr, (torch.Tensor, np.ndarray)):
        m = arr.shape[0]
    else:
        m = len(arr)

    new_line = True
    for i in range(m):
        print(f"[row {i}]")
        new_line = print1D(arr[i], num_per_line, format=format)
        if new_line is False:
            print("")
            new_line = True

    return new_line


def save_code(src_dir: str, dest_dir: str, excluded_folders=("artifacts", "__MACOSX")):
    """Backup the code in a src directory to a dst directory.

    The function recursively copies the code (determined by their file extensions) in the folder base_src_dir
    to a target folder dest_dir while preserving the sub-folder structures.

    This is useful to take snapshot of the codebase.

    Args:
        src_dir:
            The root folder directory to copy the code.
        dest_dir:
            The destination folder.
        excluded_folders:
            A list of subfolder names to exclude from the copying.  None: no excluded sub-folders.

    """
    if not os.path.exists(src_dir):
        assert False, "Source directory : " + src_dir + " does not exist"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    path_queue = [""]

    while len(path_queue) > 0:
        current_path = path_queue.pop()  # last item in the queue
        if not os.path.isdir(os.path.join(src_dir, current_path)):
            src_file = os.path.join(src_dir, current_path)
            if src_file.endswith(
                    (
                            ".py",
                            ".sh",
                            ".txt",
                            ".cpp",
                            ".c",
                            ".h",
                            ".hpp",
                            ".ipynb",
                            ".yaml",
                            ".cu",
                    )
            ):
                shutil.copyfile(src_file, os.path.join(dest_dir, current_path))
        else:
            if excluded_folders is not None and current_path in excluded_folders:
                continue
            subdirs = os.listdir(os.path.join(src_dir, current_path))
            for subdir in subdirs:
                if subdir.startswith(".") or subdir.startswith("__"):  # ignore hidden directories or files
                    continue

                # create dst dir if src is a dir
                if os.path.isdir(os.path.join(src_dir, current_path, subdir)):
                    dst_subdir = os.path.join(dest_dir, current_path, subdir)
                    if not os.path.exists(dst_subdir):
                        os.makedirs(dst_subdir)

                path_queue.append(os.path.join(current_path, subdir))


class Logger:
    """Helper class to log information.

    The class implements a universal logger that can push information to
    - local shell
    - tensorboard

    To support a multi-thread environment, it includes the rank of the current thread
    in the logging.

    Usage:

        .. code-block:: python

            # create logger
            logger = Logger(
                log_filename=os.path.join('log', f'log_rank{rank}.txt'),
                tensorboard_dir=os.path.join('tensorboard', f'rank{rank}'),
                open_tensorboard=True,
                tensorboard_port=22222,
                rank=rank,
            )

            for iter in range(max_iter):
                # some random scalars
                loss_val1 = 123.
                loss_val2 = 456.

                # log loss
                logger.add_scalars(
                    main_tag='train',
                    tag_scalar_dict={'loss_name1': loss_val1, 'loss_name2': loss_val2},
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                    )

                # remember to flush the logger every iteration
                logger.flush()

            # remember to close the logger
            logger.close()
    """

    def __init__(
            self,
            log_filename=None,
            tensorboard_dir=None,
            tensorboard_num_history_figures=100,
            tensorboard_max_reload_threads=1,
            tensorboard_exe_path="/venv/bin/tensorboard",
            open_tensorboard=False,
            tensorboard_port=22222,
            rank=0,
            launch_tensorboard_at_parent_dir: bool = True,
    ):
        """Create the logger.

        Args:
            log_filename:
                Filename to save the log.  None: do not save to a file.
                Remember to account for the rank of the thread so they don't write to the same file.
            tensorboard_dir:
                The folder to save the tensorboard file. None: do not save to tensorboard.
                Remember to account for the rank of the thread so they don't write to the same folder.
            tensorboard_num_history_figures:
                Number of historic figures shown in tensorboard.
            tensorboard_max_reload_threads:
                Number of thread to load existing tensorboard files.
            tensorboard_exe_path:
                Path of the tensorboard exe file.
            open_tensorboard:
                Whether to open the tensorboard in a thread as a module.
            tensorboard_port:
                Port to open the tensorboard if open_tensorboard=True.
            rank:
                The rank of the current thread.  None: ignored.
            launch_tensorboard_at_parent_dir:
                whether to launch the tensorboard at the parent folder of tensorboard_dir
                (so that we can compare multiple experiments).
        """
        self.log_filename = log_filename
        self.tensorboard_dir = tensorboard_dir
        self.rank = rank

        # create logger
        logger_name = "global"
        if rank is not None:
            logger_name = f"{logger_name}.rank{rank:d}"
        self.logger = self._create_logger(log_file=self.log_filename, logger_name=logger_name)

        # create tensorboard
        if self.tensorboard_dir is not None and rank is not None and rank == 0:
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            self.tb_logger = SummaryWriter(log_dir=self.tensorboard_dir)
        else:
            self.tb_logger = None

        # start tensorboard if asked
        self.tensorboard_process = None
        if open_tensorboard and rank == 0:
            if launch_tensorboard_at_parent_dir:
                tensorboard_launch_dir = os.path.normpath(os.path.join(self.tensorboard_dir, "../../.."))
            else:
                tensorboard_launch_dir = self.tensorboard_dir
            self._open_tensorboard(
                tensorboard_dir=tensorboard_launch_dir,
                samples_per_plugin_image=tensorboard_num_history_figures,
                max_reload_threads=tensorboard_max_reload_threads,
                tensorboard_exe_path=tensorboard_exe_path,
                port=tensorboard_port,
            )

        # attributes to hold information
        self.buffer = None  # a dict to buffer the losses to output
        self.epoch = None
        self.batch_idx = None
        self.global_step = 0

    def add_scalar(
            self,
            tag: str,
            scalar_value: float,
            epoch: int = None,
            batch_idx: int = None,
            global_step: int = None,
    ):
        """Register a scalar.

        Args:
            tag:
                Name of the scalar.  Can be grouped by adding '/' in it.
            scalar_value:
                Value to log
            epoch:
                Current epoch. None: ignored.
            batch_idx:
                Current batch index. None: ignored.
            global_step:
                Current global step.  None: ignored.

        Note:
            See tensorboard's documentation https://pytorch.org/docs/stable/tensorboard.html for usage.
        """

        if epoch is not None:
            self.epoch = epoch
        if batch_idx is not None:
            self.batch_idx = batch_idx
        if global_step is not None:
            self.global_step = global_step
        if tag is not None and scalar_value is not None:
            if self.buffer is None:
                self.buffer = dict()
            self.buffer[tag] = scalar_value

    def add_scalars(
            self,
            tag_scalar_dict: T.Dict[str, float],
            main_tag: str = None,
            epoch: int = None,
            batch_idx: int = None,
            global_step: int = None,
    ):
        """Add multiple scalars.

        The tag of the scalars will be main_tag/tag.

        Note:
            The scalars are plotted in separated figures.

        Args:
            main_tag:
                Main tag of the scalar.
            tag_scalar_dict:
                A dictionary maps from tag to scalar value.
            epoch:
                Current epoch. None: ignored.
            batch_idx:
                Current batch index. None: ignored.
            global_step:
                Current global step.  None: ignored.
        """
        if main_tag is None:
            main_tag = ""

        if tag_scalar_dict is not None:
            for tag, scalar_value in tag_scalar_dict.items():
                tmp_tag = f"{main_tag}{tag}"
                self.add_scalar(
                    tag=tmp_tag,
                    scalar_value=scalar_value,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=global_step,
                )

    def add_figure(self, *args, **kwargs):
        """Register a figure to the tensorboard.

        See tensorboard's documentation https://pytorch.org/docs/stable/tensorboard.html for usage.
        """
        if self.tb_logger is not None:
            self.tb_logger.add_figure(*args, **kwargs)
        else:
            self.close_fig(*args, **kwargs)

    def close_fig(self, tag, figure, global_step=None, close=True, walltime=None):
        """close matplotlib figure in case tb_logger is None."""
        plt.close(figure)

    def add_audio(self, *args, **kwargs):
        """Register a figure to the tensorboard.

        See tensorboard's documentation https://pytorch.org/docs/stable/tensorboard.html for usage.
        """
        if self.tb_logger is not None:
            self.tb_logger.add_audio(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        """Register a histogram to the tensorboard.

        See tensorboard's documentation https://pytorch.org/docs/stable/tensorboard.html for usage.
        """
        if self.tb_logger is not None:
            self.tb_logger.add_histogram(*args, **kwargs)

    def info(self, text):
        """Print an info to the shell and the log file."""
        if self.rank is not None:
            text = f"({self.rank}){text}"
        self.logger.info(text)

    def flush(self):
        """Flush the buffer to console and tensorboard.

        Note:
            It uses the latest epoch, global_step, and batch_idx, so it should be called every iteration.
        """
        # print to console and log file
        if self.buffer is not None:
            text = f"[gstep {self.global_step}, epoch {self.epoch}, batch {self.batch_idx}] "
            for key in sorted(self.buffer.keys()):
                text += f"{key}={self.buffer[key]:.3e} "
            self.info(text)

        # send to tensorboard
        if self.buffer is not None and self.tb_logger is not None:
            for tag, scalar_value in self.buffer.items():
                self.tb_logger.add_scalar(tag=tag, scalar_value=scalar_value, global_step=self.global_step)

        if self.tb_logger is not None:
            self.tb_logger.flush()

        # clear the buffer
        self.buffer = None

    def close(self):
        """Close the logger and terminates the tensorboard."""
        if self.tb_logger is not None:
            self.tb_logger.close()
        if self.tensorboard_process is not None:
            self.tensorboard_process.terminate()
            # self.tensorboard_process.kill()

        # close the python logger
        if self.logger is not None:
            for handler in self.logger.handlers:
                handler.close()
                self.logger.removeHandler(handler)

    def _open_tensorboard(
            self,
            tensorboard_dir,
            samples_per_plugin_image=100,
            max_reload_threads=1,
            tensorboard_exe_path="/venv/bin/tensorboard",
            port=22222,
    ):
        """Create a thread to run tensorboard."""

        cmd = "%s --logdir %s --port %s --samples_per_plugin=images=%d " % (
            tensorboard_exe_path,
            tensorboard_dir,
            port,
            samples_per_plugin_image,
        )

        # import tensorflow's tensorboard to check tensorflow's version
        import tensorboard

        if int(tensorboard.__version__.split(".")[0]) >= 2:
            cmd = "%s --bind_all" % (cmd)  # version 2 requirement

        # create tensorboard thread
        print(cmd)
        self.tensorboard_process = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )

    @staticmethod
    def _create_logger(log_file=None, logger_name=None) -> logging.Logger:
        """Create a logger that prints to both console and a file.

        Args:
            log_file:
                File name to save the log. None: not save to a file.
            logger_name:
                Name of the logger.

        Returns:
            Logger that can be used to output into a file
        """

        logger = logging.getLogger(logger_name)
        if logger.getEffectiveLevel() == logging.NOTSET:
            logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

        if not logger.isEnabledFor(logging.INFO):
            logging.basicConfig(level=logging.INFO)

        if not logger.handlers:
            logger.propagate = 0  # avoids printing twice on console
            log_formatter = logging.Formatter("%(asctime)s  [%(levelname)-5.5s]  %(message)s")
            # attach handler to write to the log file
            if log_file is not None:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file, "w")
                file_handler.setFormatter(log_formatter)
                logger.addHandler(file_handler)
            # attach handler to print to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            logger.addHandler(console_handler)
        return logger


class StatisticsCollector:
    """
    Compute the average and standard deviation of a dictionary
    of values.
    """

    def __init__(self, convert_to_float: bool = True):
        """
        Args:
            convert_to_float:
                whether to convert input values (from Tensor, ndarray) to float
        """
        self.convert_to_float = convert_to_float
        self.x_sum_dict = dict()
        self.x2_sum_dict = dict()
        self.x_count_dict = dict()

    def record(
            self,
            val_dict: T.Dict[str, float],
    ):
        """
        Record the values in val_dict

        Args:
            val_dict:
                a dictionary containing the floats to compute statistics.
        """

        for key, val in val_dict.items():
            if isinstance(val, torch.Tensor) and val.numel() > 1:
                continue

            if self.convert_to_float:
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().item()
                elif isinstance(val, np.ndarray):
                    val = val.item()
                elif isinstance(val, int):
                    val = float(val)
                elif isinstance(val, float):
                    pass
                else:
                    raise NotImplementedError(f"{type(val)}")

            if key not in self.x_sum_dict:
                self.x_sum_dict[key] = val
                self.x2_sum_dict[key] = val ** 2
                self.x_count_dict[key] = 1
            else:
                self.x_sum_dict[key] = self.x_sum_dict[key] + val
                self.x2_sum_dict[key] = self.x2_sum_dict[key] + val ** 2
                self.x_count_dict[key] = self.x_count_dict[key] + 1

    def compute_statistics(self) -> T.Dict[str, T.Dict[str, float]]:
        """Compute the statistics.

        Returns:
            a dictionary containing the current statistics
            `"mean"`
            `"std"`
            `"variance"`
            `"second_moment"`
            `"count"`
        """

        mean_dict = dict()
        second_moment_dict = dict()
        variance_dict = dict()
        std_dict = dict()
        count_dict = dict()

        for key in self.x_sum_dict:
            mean = self.x_sum_dict[key] / self.x_count_dict[key]
            second_moment = self.x2_sum_dict[key] / self.x_count_dict[key]
            variance = second_moment - mean ** 2
            std = variance ** 0.5

            mean_dict[key] = mean
            second_moment_dict[key] = second_moment
            variance_dict[key] = variance
            std_dict[key] = std
            count_dict[key] = self.x_count_dict[key]

        return dict(
            mean=mean_dict,
            std=std_dict,
            variance=variance_dict,
            second_moment=second_moment_dict,
            count=count_dict,
        )


def imagesc(
        arr: np.ndarray,
        xs: T.Sequence[float] = None,
        ys: T.Sequence[float] = None,
        fig=None,
        axes=None,
        dpi=150,
        colorbar=True,
):
    """
    Mimic matlab's imagesc using matplotlib.

    Args:
        arr:
            2D matrix
        xs:
            coordinate of columns (None: use 0~N-1)
        ys:
            coordinate of rows (None: use 0~M-1)

    Returns:
        fig, axes
    """

    if fig is None:
        fig, axes = plt.subplots(dpi=dpi)
    elif axes is None:
        fig.add_axes([0, 0, 1, 1])

    if xs is None:
        xs = np.arange(arr.shape[1])
    if ys is None:
        ys = np.arange(arr.shape[0])

    def extents(ts):
        if len(ts) == 1:
            delta = 1
        else:
            delta = ts[1] - ts[0]
        return [ts[0] - delta / 2, ts[-1] + delta / 2]

    norm = matplotlib.colors.Normalize(vmin=arr.min(), vmax=arr.max())
    cmap = matplotlib.cm.get_cmap("viridis")

    axes.imshow(
        arr,
        aspect="auto",
        interpolation="none",
        extent=extents(xs) + extents(ys[::-1]),
        origin="upper",
        norm=norm,
        cmap=cmap,
    )
    if colorbar:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
        )

    return fig, axes
