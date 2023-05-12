#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
# Author: Rick Chang
# This file implements the base training procedure.

import inspect
import os
import random
import traceback
import typing as T
import warnings
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from cdslib.core.models.base_model import BaseModel
from cdslib.core.utils import multigpu_utils
from cdslib.core.utils import print_and_save
from cdslib.core.utils.argparse_utils import WarnDict as dict
from cdslib.core.utils.argparse_utils import print_options
from cdslib.core.utils.argparse_utils import read_config_file
from cdslib.core.utils.argparse_utils import recursive_dict_update
from cdslib.core.utils.print_and_save import Logger
from cdslib.core.utils.print_and_save import StatisticsCollector


def customer(funcobj):
    """A decorator indicating methods to be customized."""
    return funcobj


class BaseTrainProcess(ABC):
    r"""
    The base training procedure, including handling command line arguments
    through a yaml file, distributed data parallel, training loop, etc.

    It sets up a basic framework so that specific applications can easily extend.

    Example usage:
    .. code-block:: python
        if __name__ == '__main__':
            with TrainProcess() as trainer:
                trainer.run()


    Design logic:

    - To configure the settings, you can use the arguments of init or config file.
    But note that you should not use trainer file to configure the class.

    """

    def __init__(
            self,
            exp_tag: str = "exp",
            config_filename: str = None,
            trainer_filename: str = None,
            work_dir: str = ".",
            output_dir: str = None,
            rank: int = 0,
            n_gpus: int = 1,
            random_seed: int = 0,
            save_code: bool = True,
            exclude_dirs: T.List[str] = None,
            exp_tag_first: bool = False,
            open_tensorboard: bool = True,
            ddp_type: str = "ddp",
            start_epoch: int = 0,
            end_epoch: int = 1000000,
            max_train_epoch_batches: int = -1,
            max_valid_epoch_batches: int = -1,
            max_test_epoch_batches: int = -1,
            save_every_num_epoch: int = 1,
            validate_every_num_epoch: int = 1,
            test_every_num_epoch: int = 1,
            log_every_num_train_batch: int = 1,
            log_every_num_valid_batch: int = 1,
            log_every_num_test_batch: int = 1,
            visualize_every_num_train_batch: int = 1,
            visualize_every_num_valid_batch: int = 1,
            visualize_every_num_test_batch: int = 1,
            tensorboard_exe_path: str = "tensorboard",
            # overwrite_pretrained_names: T.List[str] = None,
            use_torchrun: bool = True,
            find_unused_parameters: bool = False,  # used for DDP, see _setup_for_distributed_learning
            *args,
            **kwargs,
    ):
        """
        Args:
            exp_tag (str):
                name of the experiment
            config_filename (str):
                the yaml or json filename of the model and dataloader, etc.
            trainer_filename (str):
                the previous trainer file to resume from. None: no resume.
            output_dir (str):
                the root dir of the outputs
            rank (int):
                the rank of the process. Optional if use torchrun to launch.
            n_gpus (int):
                number of gpus (world size). Optional if use torchrun to launch.
            random_seed (int):
                random seed
            save_code (bool):
                whether to save the code in output_dir for future reference.
            exclude_dirs (list of str):
                dirs to exclude from saving code
            exp_tag_first (bool):
                whether the output_dir struture is:

                - `True`: output_dir/exp_tag/checkpoint
                - `False`: output_dir/checkpoint/exp_tag
            open_tensorboard (bool):
                whether to open tensorboard
            ddp_type (str):
                method for Distributed Data Parallel (DDP).

                - `"ddp"`: use pytorch's DDP, which sync at construction, forward, and backward
                - `"gradient_only"`: use all_reduce to sync only at backward
            start_epoch (int):
                epoch to start
            end_epoch (int)
                epoch to end
            max_train_epoch_batches (int):
                max number of batches for training epoch. `-1`: unlimited (same as len(dataloader)).
            max_valid_epoch_batches (int):
                max number of batches for validation epoch. `-1`: unlimited (same as len(val_dataloader)).
            max_test_epoch_batches (int):
                max number of batches for test epoch. `-1`: unlimited (same as len(test_dataloader)).
            save_every_num_epoch (int):
                how often to save the model
            validate_every_num_epoch (int):
                how often to run validation
            test_every_num_epoch (int):
                how often to test the model
            log_every_num_train_batch (int):
                how often to log the training batch output
            log_every_num_valid_batch (int):
                how often to log the training batch output
            log_every_num_test_batch (int):
                how often to log the test batch output
            visualize_every_num_train_batch (int):
                how often to log the training batch output
            visualize_every_num_valid_batch (int):
                how often to log the training batch output
            visualize_every_num_test_batch (int):
                how often to log the test batch output
            tensorboard_exe_path (str):
                the path of tensorboard executable
            # overwrite_pretrained_names (list of str):
            #     list of names that will be omitted when loading
            #     an existing trainer file.
            #     Note that anything ends with _info (e.g., process_info)
            #     will always be overwritten (not loaded from pretrained file).
            use_torchrun:
                whether the distributed computing is launched by :py:`torchrun`.
        """

        # if overwrite_pretrained_names is None:
        #     overwrite_pretrained_names = []

        # if 'process_info' not in overwrite_pretrained_names:
        #     overwrite_pretrained_names.append('process_info')

        # if environment variables are set (e.g., by torchrun), use those.
        rank = int(os.environ.get("LOCAL_RANK", rank))
        global_rank = int(os.environ.get("RANK", rank))
        n_gpus = int(os.environ.get("LOCAL_WORLD_SIZE", n_gpus))
        global_world_size = int(os.environ.get("WORLD_SIZE", n_gpus))
        print(
            f"rank = {rank}, global_rank = {global_rank}, n_gpus = {n_gpus}, " f"world_size = {global_world_size}",
            flush=True,
        )

        if trainer_filename is not None and not os.path.exists(trainer_filename):
            warnings.warn(f"trainer_filename {trainer_filename} does not exist, start fresh")
            trainer_filename = None

        # make sure work_dir is in absolute path
        work_dir = os.path.abspath(work_dir)

        # store setup info about the trainer
        # (this also sets the default values)
        self.process_info = dict(
            exp_tag=exp_tag,
            config_filename=config_filename,
            trainer_filename=trainer_filename,
            work_dir=work_dir,
            output_dir=output_dir,
            rank=rank,  # local rank
            n_gpus=n_gpus,
            global_rank=global_rank,
            global_world_size=global_world_size,
            distributed_run=n_gpus > 1,
            random_seed=random_seed,
            save_code=save_code,
            exclude_dirs=exclude_dirs,
            exp_tag_first=exp_tag_first,
            open_tensorboard=open_tensorboard,
            ddp_type=ddp_type,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            max_train_epoch_batches=max_train_epoch_batches,
            max_valid_epoch_batches=max_valid_epoch_batches,
            max_test_epoch_batches=max_test_epoch_batches,
            save_every_num_epoch=save_every_num_epoch,
            validate_every_num_epoch=validate_every_num_epoch,
            test_every_num_epoch=test_every_num_epoch,
            log_every_num_train_batch=log_every_num_train_batch,
            log_every_num_valid_batch=log_every_num_valid_batch,
            log_every_num_test_batch=log_every_num_test_batch,
            visualize_every_num_train_batch=visualize_every_num_train_batch,
            visualize_every_num_valid_batch=visualize_every_num_valid_batch,
            visualize_every_num_test_batch=visualize_every_num_test_batch,
            tensorboard_exe_path=tensorboard_exe_path,
            # overwrite_pretrained_names=overwrite_pretrained_names,
            use_torchrun=use_torchrun,
            find_unused_parameters=find_unused_parameters,
        )

        self.logger: T.Union[Logger, None] = None
        self.device: T.Union[torch.device, None] = None
        self.epoch = start_epoch
        self.total_batch_count = 0

        # read config file
        self.load_options(filename=self.process_info["config_filename"])

        # a placeholder containing important information to be saved in checkpoint
        self._register_var_to_save(
            var_name=[
                "process_info",
                "options",
            ]
        )  # when resuming training, the process_info like rank might change.

        self._register_var_to_load(
            var_name=[
                "epoch",
                "total_batch_count",
            ]
        )  # these are the training state that should be loaded when resuming.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    def _register_var_to_save(
            self,
            var_name: T.Union[str, T.List[str]],
    ):
        """
        Make sure the var will be saved in the checkpoint.
        Note that it is just for recording purpose.
        They will NOT be loaded during retraining.
        """
        if isinstance(var_name, str):
            var_name = [var_name]

        if getattr(self, "var_names_to_save", None) is None:
            self.var_names_to_save = set()
        for name in var_name:
            self.var_names_to_save.add(name)

    def _register_var_to_load(
            self,
            var_name: T.Union[str, T.List[str]],
    ):
        """Register variable to be loaded during retraining."""
        if isinstance(var_name, str):
            var_name = [var_name]

        if getattr(self, "var_names_to_load", None) is None:
            self.var_names_to_load = set()
        for name in var_name:
            self.var_names_to_load.add(name)

    def run(self):
        """The main process."""

        # make sure to load config again
        self.load_options(filename=self.process_info["config_filename"])

        # initialize port for distribution run
        if self.process_info["distributed_run"]:
            multigpu_utils.init_distributed(
                self.process_info["n_gpus"],
                self.process_info["rank"],
                auto_detect=self.process_info["use_torchrun"],
            )
            print(
                "Distributed training is enabled. " "Make sure all the modules will be wrapped by all_reduced or DDP!"
            )

        # determine output dir
        if self.process_info["output_dir"] is None:
            self.process_info["output_dir"] = "artifacts"

        # determine exp_tag
        if self.process_info["rank"] == 0:
            ori_exp_tag = self.process_info["exp_tag"]
            self.process_info["exp_tag"] = self.determine_exp_tag(
                exp_tag=self.process_info["exp_tag"],
                artifact_root_dir=self.process_info["output_dir"],
                continuing=self.process_info["trainer_filename"] is not None,
            )
            if self.process_info["distributed_run"] and self.process_info["exp_tag"] != ori_exp_tag:
                raise RuntimeError(
                    f"distributed enable, "
                    f'but final exp_tag {self.process_info["exp_tag"]} != '
                    f"given exp_tag {ori_exp_tag}"
                )

        if self.process_info["distributed_run"]:
            torch.distributed.barrier()

        # create folder to save checkpoints, log, plots
        dir_dict = self._create_folders(
            root_dir=self.process_info["output_dir"],
            exp_tag=self.process_info["exp_tag"],
            exp_tag_first=self.process_info["exp_tag_first"],
        )
        self.process_info["dir_dict"] = dir_dict

        if self.process_info["exclude_dirs"] is None:
            self.process_info["exclude_dirs"] = []

        self.process_info["exclude_dirs"] += list(dir_dict.values())

        # save the code for future reference
        if self.process_info["rank"] == 0 and self.process_info["save_code"]:
            print_and_save.save_code(
                src_dir=os.getcwd(),
                dest_dir=dir_dict["code"],
                excluded_folders=self.process_info["exclude_dirs"],
            )

        # save options for future reference
        dir_dict["option_txt"] = os.path.join(dir_dict["option"], "options.txt")
        dir_dict["option_json"] = os.path.join(dir_dict["option"], "options.json")
        dir_dict["option_yaml"] = os.path.join(dir_dict["option"], "options.yaml")
        print_options(
            options=self.options,
            output_file=dir_dict["option_txt"],
            output_json_file=dir_dict["option_json"],
            output_yaml_file=dir_dict["option_yaml"],
        )

        # initialize logger
        dir_dict["log_txt"] = os.path.join(dir_dict["logs"], "log.txt")
        self.logger = Logger(
            rank=self.process_info["rank"],
            log_filename=dir_dict["log_txt"],
            tensorboard_dir=dir_dict["tensorboard"],
            open_tensorboard=self.process_info["open_tensorboard"],
            tensorboard_port=os.environ.get("TENSORBOARD_PORT", 22222),
            tensorboard_exe_path=self.process_info["tensorboard_exe_path"],
            launch_tensorboard_at_parent_dir=not self.process_info["exp_tag_first"],
        )

        self.logger.info(f'exp_tag = {self.process_info["exp_tag"]}')
        self.logger.info(f'work_dir = {self.process_info["work_dir"]}')
        self.logger.info(f'output_dir = {self.process_info["output_dir"]}')

        # send model to device
        self.device = self._determine_device(self.process_info)

        # download necessary data
        if self.process_info["rank"] == 0:
            self.download_assets()

        if self.process_info["distributed_run"]:
            torch.distributed.barrier()

        self.setup_assets()

        # additional steps
        self.additional_setup_before_dataloader()

        if self.process_info["distributed_run"]:
            torch.distributed.barrier()

        # get dataloader
        dataloader, val_dataloader, test_dataloader = self.get_dataloaders()

        # initialize random seed
        if self.process_info["random_seed"] >= 0 or self.process_info["distributed_run"]:
            assert self.process_info["random_seed"] >= 0, f"{self.process_info['random_seed']}"
            random.seed(self.process_info["random_seed"])
            np.random.seed(self.process_info["random_seed"])
            torch.manual_seed(self.process_info["random_seed"])
            torch.cuda.manual_seed(self.process_info["random_seed"])
            torch.cuda.manual_seed_all(self.process_info["random_seed"])

        # construct the model
        self.construct_models()

        # send models to device
        self._send_models_to_device(self.device)

        # construct optimizers
        self.construct_optimizers()

        # load pretrained trainer states
        if self.process_info["trainer_filename"] is not None:
            if os.path.exists(self.process_info["trainer_filename"]):
                self.load(self.process_info["trainer_filename"], device=self.device)

                # config the optimizer using the new states
                self.reconfigure_optimizer()
            else:
                warnings.warn(
                    f"trainer_filename = " f'{self.process_info["trainer_filename"]} ' f"is given but does not exist."
                )

        # setup distributed data parallel for model
        # sync initial parameters and setup all-reduce hook of gradient
        # no manually changing model parameter from now on
        if self.process_info["distributed_run"]:
            self._setup_for_distributed_learning()

        # additional steps
        self.additional_setups_before_train()

        # epoch loop
        while self.epoch <= self.process_info["end_epoch"]:
            self.logger.info("epoch %d" % (self.epoch))

            # setup for epoch (e.g., set_epoch for dataloader, etc, make sure models are in train mode)
            self.epoch_setup(
                epoch=self.epoch,
                dataloader=dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
            )
            if self.process_info["distributed_run"]:
                torch.distributed.barrier()

            # training
            self.logger.info("Training: -------------------------------------")
            self._set_train_mode()
            self.train_setup(epoch=self.epoch)
            if self.process_info["distributed_run"]:
                torch.distributed.barrier()

            out_dict = self._loop_dataloader(
                epoch=self.epoch,
                dataloader=dataloader,
                mode="train",
                step_func=self.train_step,
                visualize_func=self.visualize_train_step,
                total_batch_count=self.total_batch_count,
                max_epoch_batches=self.process_info["max_train_epoch_batches"],
                log_every_num_batch=self.process_info["log_every_num_train_batch"],
                visualize_every_num_batch=self.process_info["visualize_every_num_train_batch"],
            )
            self.total_batch_count = out_dict["total_batch_count"]

            if self.process_info["distributed_run"]:
                torch.distributed.barrier()

            # validation
            self.logger.info("Validating: -------------------------------------")
            self._set_eval_mode()
            self.validation_setup(epoch=self.epoch)
            if self.process_info["distributed_run"]:
                torch.distributed.barrier()

            out_dict = self._loop_dataloader(
                epoch=self.epoch,
                dataloader=val_dataloader,
                mode="valid",
                step_func=self.validation_step,
                visualize_func=self.visualize_validation_step,
                total_batch_count=self.total_batch_count,
                max_epoch_batches=self.process_info["max_valid_epoch_batches"],
                log_every_num_batch=self.process_info["log_every_num_valid_batch"],
                visualize_every_num_batch=self.process_info["visualize_every_num_valid_batch"],
            )

            if self.process_info["distributed_run"]:
                torch.distributed.barrier()

            # test
            self.logger.info("Testing: -------------------------------------")
            self._set_eval_mode()
            self.test_setup(epoch=self.epoch)
            if self.process_info["distributed_run"]:
                torch.distributed.barrier()

            out_dict = self._loop_dataloader(
                epoch=self.epoch,
                dataloader=test_dataloader,
                mode="test",
                step_func=self.test_step,
                visualize_func=self.visualize_test_step,
                total_batch_count=self.total_batch_count,
                max_epoch_batches=self.process_info["max_test_epoch_batches"],
                log_every_num_batch=self.process_info["log_every_num_test_batch"],
                visualize_every_num_batch=self.process_info["visualize_every_num_test_batch"],
            )

            if self.process_info["distributed_run"]:
                torch.distributed.barrier()

            # save model
            if self.epoch % self.process_info["save_every_num_epoch"] == 0 and self.process_info["rank"] == 0:
                # save model
                filename = os.path.join(dir_dict["checkpoint"], "epoch%d.pth" % (self.epoch))
                self.logger.info(f"saving model to {filename}")
                self.save(filename, epoch=self.epoch + 1)  # to continue on the next epoch
                self.logger.info("finished saving")

            if self.process_info["distributed_run"]:
                torch.distributed.barrier()

            # update epoch
            self.logger.add_scalar(tag="epoch", scalar_value=self.epoch, epoch=self.epoch)
            self.logger.info(f"end of epoch {self.epoch}")
            self.logger.flush()
            self.epoch += 1

        if self.process_info["distributed_run"]:
            torch.distributed.barrier()

        self.finish_procedure()

    def load_options(self, filename: str = None):
        """
        Load the config file, which is a dictionary of: var_name -> Dict[key, val].
        Each of the key in var_name will replace the current value of var_name.
        """

        if filename is None:
            self.options = dict()
            return self.options

        # parse the config file
        options = read_config_file(filename=filename)

        # replace
        for var_name in options:
            info = getattr(self, var_name, None)

            if info is None:
                setattr(self, var_name, options[var_name])
                # if isinstance(options[var_name], dict):
                #     info = dict()
                #     for key, val in options[var_name].items():
                #         info[key] = val
                #     setattr(self, var_name, info)
                # else:
                #     setattr(self, var_name, options[var_name])
            else:
                if isinstance(info, T.MutableMapping):
                    assert isinstance(options[var_name], T.MutableMapping), f"{var_name}, {type(options[var_name])}"
                    recursive_dict_update(tgt_dict=info, src_dict=options[var_name])
                    # for key, val in options[var_name].items():
                    #     info[key] = val
                else:
                    setattr(self, var_name, options[var_name])

        self.options = options
        return self.options

    def _create_folders(self, root_dir: str, exp_tag: str, exp_tag_first: bool):
        """
        Create the folders to store codes, checkpoints, etc.
        """

        dir_dict = dict()
        for dir_name in [
            "checkpoint",
            "tensorboard",
            "logs",
            "plots",
            "code",
            "option",
        ]:
            if exp_tag_first:
                dir_dict[dir_name] = os.path.join(root_dir, exp_tag, dir_name)
            else:
                dir_dict[dir_name] = os.path.join(root_dir, dir_name, exp_tag)
            os.makedirs(dir_dict[dir_name], exist_ok=True)

        return dir_dict

    @staticmethod
    def _determine_device(process_info: T.Dict[str, T.Any]):
        """Figure out which device (i.e., gpu or cpu) to use."""
        if torch.cuda.is_available() and process_info["n_gpus"] > 0:
            gpu_id = process_info["rank"] % process_info["n_gpus"]
            assert (
                    torch.cuda.device_count() >= process_info["n_gpus"]
            ), f'{torch.cuda.device_count()} {process_info["n_gpus"]}'
            # set the default cuda device
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
        return device

    def _send_models_to_device(self, device: torch.device):
        """Send all base model and nn.modules to device."""
        isModel = lambda net: isinstance(net, BaseModel)
        nets = inspect.getmembers(self, isModel)  # name, net
        for name, net in nets:
            getattr(self, name).to(device)

        isNNModule = lambda net: isinstance(net, torch.nn.Module)
        nets = inspect.getmembers(self, isNNModule)  # name, net
        for name, net in nets:
            getattr(self, name).to(device)

    def _set_train_mode(self):
        """Set all the nn.modules to train mode."""

        vs = inspect.getmembers(self, lambda v: isinstance(v, BaseModel))
        for v_name, v in vs:
            v.train()

        vs = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))
        for v_name, v in vs:
            v.train()

    def _set_eval_mode(self):
        """Set all the nn.modules to evaluation mode."""
        vs = inspect.getmembers(self, lambda v: isinstance(v, BaseModel))
        for v_name, v in vs:
            v.eval()

        vs = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))
        for v_name, v in vs:
            v.eval()

    def load(self, filename: str, device: torch.device = torch.device("cpu")):
        """
        Load the model saved at the filenames.
        It automatically gathers all the nn.modules and optimizers and load them.
        Everything will be loaded onto cpu.
        """
        checkpoint = torch.load(filename, map_location=device)  # dict of [name, data]
        loaded_names = set()

        # gather all base models
        nets = inspect.getmembers(self, lambda x: isinstance(x, BaseModel))  # name, net
        for name, net in nets:
            if name in checkpoint:
                getattr(self, name).load_state_dict(checkpoint[name])
            else:
                warnings.warn(f"{name} not exist in checkpoint")
            loaded_names.add(name)

        # gather all nn.modules
        nets = inspect.getmembers(self, lambda x: isinstance(x, torch.nn.Module))  # name, net
        for name, net in nets:
            if name in checkpoint:
                try:
                    getattr(self, name).load_state_dict(checkpoint[name])
                except:
                    new_state_dict = OrderedDict()
                    for key, val in checkpoint[name].items():
                        if key.startswith("module."):
                            new_key = key[7:]
                            new_state_dict[new_key] = val
                        else:
                            new_state_dict[key] = val
                    getattr(self, name).load_state_dict(new_state_dict)
            else:
                warnings.warn(f"{name} not exist in checkpoint")
            loaded_names.add(name)

        # gather all optimizers
        optimizers = inspect.getmembers(self, lambda x: isinstance(x, torch.optim.Optimizer))  # name, net
        for name, optimizer in optimizers:
            if name in checkpoint:
                getattr(self, name).load_state_dict(checkpoint[name])
            else:
                warnings.warn(f"{name} not exist in checkpoint")
            loaded_names.add(name)

        # gather all AMP scaler
        try:
            scalers = inspect.getmembers(self, lambda x: isinstance(x, torch.cuda.amp.GradScaler))  # name, net
            for name, scaler in scalers:
                if name in checkpoint:
                    getattr(self, name).load_state_dict(checkpoint[name])
                else:
                    warnings.warn(f"{name} not exist in checkpoint")
                loaded_names.add(name)
        except:
            pass

        # load those in var_names_to_load other things
        var_names_to_save = checkpoint.get("var_names_to_save", set())
        self._register_var_to_save(
            var_name=var_names_to_save,
        )

        var_names_to_load = checkpoint.get("var_names_to_load", set())
        self._register_var_to_load(
            var_name=var_names_to_load,
        )

        # all_names = set(checkpoint.keys())
        # load only those in var_names_to_load
        for name in var_names_to_load - loaded_names:
            # if name in self.process_info['overwrite_pretrained_names'] or \
            #         name not in checkpoint:
            #     continue
            if name not in checkpoint:
                continue
            setattr(self, name, checkpoint[name])

        # # make sure to start at a new epoch
        # if 'epoch' in checkpoint and \
        #         'epoch' not in self.process_info['overwrite_pretrained_names']:
        #     self.epoch = checkpoint['epoch'] + 1

        return checkpoint

    def save(self, filename: str, epoch: int = None):
        """Save the nn.module, optimizer, and customized save dict.

        Args:
            filename:
                the filename of the pth file to save
            epoch:
                the current epoch number
        """

        to_save = dict()

        # additional info to save
        to_save["var_names_to_save"] = getattr(self, "var_names_to_save", set())
        for var_name in to_save["var_names_to_save"]:
            to_save[var_name] = getattr(self, var_name, None)

        # additional info to load
        to_save["var_names_to_load"] = getattr(self, "var_names_to_load", set())
        for var_name in to_save["var_names_to_load"]:
            to_save[var_name] = getattr(self, var_name, None)

        # gather all model
        nets = inspect.getmembers(self, lambda v: isinstance(v, BaseModel))  # name, net
        for name, net in nets:
            to_save[name] = net.state_dict()

        # gather all nn.modules
        nets = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))  # name, net
        for name, net in nets:
            if isinstance(net, DDP):
                to_save[name] = net.module.state_dict()
            else:
                to_save[name] = net.state_dict()

        # gather all optimizers
        optimizers = inspect.getmembers(self, lambda v: isinstance(v, torch.optim.Optimizer))  # name, optimizer
        for name, optimizer in optimizers:
            to_save[name] = optimizer.state_dict()

        # gather all automatic mixed precision scaler
        try:
            scalers = inspect.getmembers(self, lambda v: isinstance(v, torch.cuda.amp.GradScaler))  # name, scaler
            for name, scaler in scalers:
                to_save[name] = scaler.state_dict()
        except:
            pass

        # save the classname
        to_save["classname"] = self.__class__.__name__
        to_save["epoch"] = epoch

        # save to the file
        torch.save(to_save, filename)

    def _setup_for_distributed_learning(self):
        """
        Setup the networks in the model for distributed computation.
        It automactically gather all modules.
        """

        # base models
        nets = inspect.getmembers(self, lambda v: isinstance(v, BaseModel))  # name, net
        for name, net in nets:
            self.logger.info(f"setting up model {name} for distributed learning")
            net.setup_for_distributed_learning(ddp_type=self.process_info["ddp_type"])

        # nn modules
        if self.process_info["ddp_type"] == "ddp":
            # using pytorch distributed data parallel
            # gather all nn.modules
            nets = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))  # name, net

            for name, net in nets:
                self.logger.info(f"setting {name} as DDP")
                first = None
                try:
                    first = next(net.parameters())
                except StopIteration:
                    pass
                if first is not None:  # DDP takes only modules that contain parameters
                    # see https://pytorch.org/docs/stable/notes/ddp.html,
                    # used when some parameters are not used in loss
                    # not optimal but a walk-around to enable multi-gpu
                    # net = DDP(
                    #     getattr(self, name),
                    #     device_ids=[self.device],
                    #     output_device=self.device,
                    # )
                    net = DDP(
                        getattr(self, name),
                        device_ids=[self.device],
                        output_device=self.device,
                        find_unused_parameters=self.process_info['find_unused_parameters'],
                    )
                    setattr(self, name, net)

        elif self.process_info["ddp_type"] == "apex_ddp":
            # using apex distributed data parallel
            # gather all nn.modules
            try:
                from apex.parallel import DistributedDataParallel as APEX_DDP
            except:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex")
            nets = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))  # name, net

            for name, net in nets:
                self.logger.info(f"setting {name} as DDP")
                # By default, apex.parallel.DistributedDataParallel overlaps communication with
                # computation in the backward pass.
                # model = DDP(model)
                # delay_allreduce delays all communication to the end of the backward pass.
                net = APEX_DDP(getattr(self, name))
                setattr(self, name, net)

        elif self.process_info["ddp_type"] == "gradient_only":
            self.logger.info("using apply_gradient_allreduce")
            from cdslib.core.utils.multigpu_utils import apply_gradient_allreduce

            # gather all nn.modules
            nets = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))  # name, net
            for name, net in nets:
                self.logger.info(f"setting {name} as gradient_only ddp")
                net = apply_gradient_allreduce(getattr(self, name))
                setattr(self, name, net)
        else:
            raise NotImplementedError(f"{self.process_info['ddp_type']}")

    def _loop_dataloader(
            self,
            epoch: int,
            dataloader: T.Sequence[T.Any],
            mode: str,  # train, valid, test
            step_func: T.Callable,
            visualize_func: T.Callable,
            total_batch_count: int,
            max_epoch_batches: int,
            log_every_num_batch: int,
            visualize_every_num_batch: int,
    ):

        if dataloader is None:
            return dict(
                total_batch_count=total_batch_count,
                total_epoch_batch=0,
                epoch_time=0,
                stats_dict=None,
            )

        # start running training loop
        epoch_stime = timer()
        batch_stime = timer()
        total_epoch_batch = 0

        # to compute the statistics
        statistics = StatisticsCollector(convert_to_float=True)

        for batch_idx, batch in enumerate(dataloader):
            if max_epoch_batches >= 0 and total_epoch_batch >= max_epoch_batches:
                break

            # train one step
            step_stime = timer()
            out_dict = step_func(
                epoch=epoch,
                bidx=batch_idx,
                batch=batch,
            )
            step_etime = timer()
            # record the outputs to compute statistics
            if out_dict is not None:
                statistics.record(out_dict)

            # print timing
            batch_time = step_etime - batch_stime
            step_time = step_etime - step_stime
            epoch_time = step_etime - epoch_stime

            # print loss
            if out_dict is not None and batch_idx % log_every_num_batch == 0:
                # print the output
                self.logger.add_scalars(
                    main_tag=f"{mode}_",
                    tag_scalar_dict=out_dict,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=total_batch_count,
                )

            # additional log function (e.g., visualization)
            if batch_idx % visualize_every_num_batch == 0:
                log_output = visualize_func(
                    epoch=epoch,
                    total_batch_count=total_batch_count,
                    bidx=batch_idx,
                    batch=batch,
                    out_dict=out_dict,
                    logger=self.logger,
                    batch_time=batch_time,
                    step_time=step_time,
                    epoch_time=epoch_time,
                )

            # flush logger every iteration
            self.logger.flush()

            total_batch_count += 1
            total_epoch_batch += 1
            batch_stime = timer()

        # log epoch timing
        epoch_time = timer() - epoch_stime
        self.logger.info(
            "[epoch %d (%s) finished]: loaded %d batches, takes %.2f mins (%.3f per batch)"
            % (
                epoch,
                mode,
                total_epoch_batch,
                epoch_time / 60.0,
                epoch_time / total_epoch_batch if total_epoch_batch > 0 else 0,
            )
        )

        # log output statistics
        stats_dict = statistics.compute_statistics()
        keys = list(stats_dict["mean"].keys())
        for key in keys:
            self.logger.info(f'{key}: mean / std = {stats_dict["mean"][key]:.4f} / {stats_dict["std"][key]:.4f}')
            self.logger.add_scalar(
                tag=f"{mode}_epoch_{key}",
                scalar_value=stats_dict["mean"][key],
                epoch=epoch,
            )
        self.logger.flush()

        return dict(
            total_batch_count=total_batch_count,
            total_epoch_batch=total_epoch_batch,
            epoch_time=epoch_time,
            stats_dict=stats_dict,
        )

    def _cleanup(self):
        if self.logger is not None:
            self.logger.close()

        if self.process_info["distributed_run"]:
            torch.distributed.destroy_process_group()

    @staticmethod
    def determine_exp_tag(
            exp_tag: str,
            artifact_root_dir: str,
            continuing: bool = False,
    ):
        """
        Make sure the exp_tag does not exist. If existed, modify it so that
        it is unique by appending the current date and time to it.

        Additionally, remove invalid characters from exp_tag.
        """

        # remove bad characters
        bad_chars = [";", ":", "!", "*", " "]
        for c in bad_chars:
            exp_tag = exp_tag.replace(c, "_")
        bad_chars = ["="]
        for c in bad_chars:
            exp_tag = exp_tag.replace(c, "-")

        # append a number after exp_tag to avoid overwrite existing exps
        if not continuing and (
                os.path.exists(os.path.join(artifact_root_dir, "checkpoint", exp_tag))
                or os.path.exists(os.path.join(artifact_root_dir, exp_tag, "checkpoint"))
        ):
            now_str = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
            exp_tag = "%s-%s" % (exp_tag, now_str)

        return exp_tag

    # ------------------------------------------------------------------------------------
    # customer methods
    # ------------------------------------------------------------------------------------

    # @customer
    # @staticmethod
    # def add_commandline_options(
    #         parser: argparse.ArgumentParser,
    # ) -> argparse.ArgumentParser:
    #     """Add commandline options related to model, dataloader, etc."""
    #     return parser

    @customer
    def download_assets(self):
        """
        Download the assets like dataset, pretrained models, etc.
        """

    @customer
    def setup_assets(self):
        """
        Setup the assets like dataset, pretrained models, etc.
        """

    @customer
    def additional_setup_before_dataloader(self):
        """Additional things to do before getting dataloader."""

    @customer
    @abstractmethod
    def get_dataloaders(self):
        """Returns the training dataloader and validation dataloader.
        The dataloaders iterate through batches.
        Returns:
            dataloader:
                dataloader for training
            val_dataloader:
                dataloader for validation
            test_dataloader:
                dataloader for testing
        """
        raise NotImplementedError

    @customer
    @abstractmethod
    def construct_models(self):
        """
        Construct the nn.Modules.
        No need to return the models --- they will be used through `self.xxxx`.
        """
        raise NotImplementedError

    @customer
    @abstractmethod
    def construct_optimizers(self):
        raise NotImplementedError

    @customer
    def reconfigure_optimizer(self):
        """Set the optimizer's state (e.g., init_step) after pre-trained models has been loaded."""

    @customer
    def additional_setups_before_train(self):
        """Additional things to do before training starts."""

    @customer
    def epoch_setup(
            self,
            epoch: int,
            dataloader: T.Sequence[T.Any],
            val_dataloader: T.Sequence[T.Any],
            test_dataloader: T.Sequence[T.Any],
    ):
        """Set up at the beginning of an epoch, before dataloder iterator
        is constructed.  It can be used to setup the batch sampler, etc."""

    @customer
    def train_setup(self, epoch: int):
        """Setup before training loop."""

    @customer
    @abstractmethod
    def train_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ) -> T.Dict[str, T.Any]:
        """One training step.

        Return a dictionary that will be passed to logging.
        """
        raise NotImplementedError

    @customer
    def visualize_train_step(
            self,
            epoch: int,
            total_batch_count: int,
            bidx: int,
            batch: T.Any,
            out_dict: T.Dict[str, T.Any],
            logger: Logger,
            batch_time: float,
            step_time: float,
            epoch_time: float,
    ):
        self.logger.info(
            f"batch uses: {batch_time:.4f} secs, "
            f"(step uses: {step_time:.4f} secs, "
            f"{step_time / batch_time * 100.:.2f}%)"
        )

    @customer
    def validation_setup(self, epoch: int):
        """Setup before validation loop."""

    @customer
    @abstractmethod
    def validation_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ) -> T.Dict[str, T.Any]:
        """One training step.

        Return a dictionary that will be passed to logging.
        """
        raise NotImplementedError

    @customer
    def visualize_validation_step(
            self,
            epoch: int,
            total_batch_count: int,
            bidx: int,
            batch: T.Any,
            out_dict: T.Dict[str, T.Any],
            logger: Logger,
            batch_time: float,
            step_time: float,
            epoch_time: float,
    ):
        self.logger.info(
            f"batch uses: {batch_time:.4f} secs, "
            f"(step uses: {step_time:.4f} secs, "
            f"{step_time / batch_time * 100.:.2f}%)"
        )

    @customer
    def test_setup(self, epoch: int):
        """Setup before test loop."""

    @customer
    @abstractmethod
    def test_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ) -> T.Dict[str, T.Any]:
        """One training step.

        Return a dictionary that will be passed to logging.
        """
        raise NotImplementedError

    @customer
    def visualize_test_step(
            self,
            epoch: int,
            total_batch_count: int,
            bidx: int,
            batch: T.Any,
            out_dict: T.Dict[str, T.Any],
            logger: Logger,
            batch_time: float,
            step_time: float,
            epoch_time: float,
    ):
        self.logger.info(
            f"batch uses: {batch_time:.4f} secs, "
            f"(step uses: {step_time:.4f} secs, "
            f"{step_time / batch_time * 100.:.2f}%)"
        )

    @customer
    def finish_procedure(self):
        pass
