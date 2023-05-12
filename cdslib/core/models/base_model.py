#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
# The file defines the base model
from abc import abstractmethod
from collections import OrderedDict
import inspect
import itertools
import typing as T
import warnings

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from apex.parallel import DistributedDataParallel as APEX_DDP
except:
    from torch.nn.parallel import DistributedDataParallel as APEX_DDP


class BaseModel:
    """
    The base model.  It is a convenient class the collects
    necessary information (e.g., normalization statistics,
    codemap, etc) with the model so they can be tracked in
    the future.

    You should define two functions:

    - forward:
        this is called during `training` (via Model(...))
    - infer:
        this is called during `inference` (via Model.infer(...))

    Note that it is not a `nn.Module`. It defines a collection
    of `nn.Modules` and their interaction during training and inference.

    Instead of defining as a single nn.Module, this way we allows more easily
    using different optimizer for different sub-nn.Module.
    """

    def __init__(self):
        super().__init__()
        self.var_names_to_save: T.Set[str] = set()
        self.var_names_to_load: T.Set[str] = set()
        self.buffer_names: T.Set[str] = set()
        self.parameter_names: T.Set[str] = set()
        self.device = torch.device("cpu")

    def register_buffer(self, name: str, tensor: torch.Tensor, persistent=True):
        self.buffer_names.add(name)
        setattr(self, name, tensor)

        if persistent:
            self.register_var_to_load(var_name=name)
        else:
            self.register_var_to_save(var_name=name)

    def register_parameter(self, name: str, param: torch.Tensor):
        self.parameter_names.add(name)
        setattr(self, name, param)
        self.register_var_to_load(var_name=name)

    def register_var_to_save(
        self,
        var_name: T.Union[str, T.List[str]],
    ):
        """
        Make sure the var will be saved in the state_dict.
        Note that it is just for recording purpose.
        They will NOT be loaded during retraining.
        """
        # assert torch.__version__ >= "1.10.0"
        if isinstance(var_name, str):
            var_name = [var_name]

        if getattr(self, "var_names_to_save", None) is None:
            self.var_names_to_save = set()
        for name in var_name:
            self.var_names_to_save.add(name)

    def register_var_to_load(
        self,
        var_name: T.Union[str, T.List[str]],
    ):
        """Register variable to be loaded during retraining."""
        # assert torch.__version__ >= "1.10.0"
        if isinstance(var_name, str):
            var_name = [var_name]

        if getattr(self, "var_names_to_load", None) is None:
            self.var_names_to_load = set()
        for name in var_name:
            self.var_names_to_load.add(name)

    def parameters(self):
        """Get all parameters from individual torch.nn.Module"""
        parameters = []

        # base model
        isModel = lambda net: isinstance(net, BaseModel)
        nets = inspect.getmembers(self, isModel)  # name, net
        for name, net in nets:
            parameters.append(net.parameters())

        #  nn.module
        isNNModule = lambda net: isinstance(net, torch.nn.Module)
        nets = inspect.getmembers(self, isNNModule)  # name, net
        for name, net in nets:
            parameters.append(net.parameters())

        parameters = itertools.chain(*parameters)
        return parameters

    def to(self, device: torch.device):
        """
        Send the model to device.
        """
        self.device = device

        # base model
        isModel = lambda net: isinstance(net, BaseModel)
        nets = inspect.getmembers(self, isModel)  # name, net
        for name, net in nets:
            getattr(self, name).to(device)

        # nn module
        isNNModule = lambda net: isinstance(net, torch.nn.Module)
        nets = inspect.getmembers(self, isNNModule)  # name, net
        for name, net in nets:
            getattr(self, name).to(device)

        # buffer
        for name in self.buffer_names:
            b = getattr(self, name, None)
            if b is None:
                continue
            else:
                setattr(self, name, b.to(device=device))

        # parameter
        for name in self.parameter_names:
            b = getattr(self, name, None)
            if b is None:
                continue
            else:
                setattr(self, name, b.to(device=device))

    def train(self):
        """Set all the nn.modules to train mode."""
        vs = inspect.getmembers(self, lambda v: isinstance(v, BaseModel))
        for v_name, v in vs:
            v.train()

        vs = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))
        for v_name, v in vs:
            v.train()

    def eval(self):
        """Set all the nn.modules to evaluation mode."""
        vs = inspect.getmembers(self, lambda v: isinstance(v, BaseModel))
        for v_name, v in vs:
            v.eval()

        vs = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))
        for v_name, v in vs:
            v.eval()

    def load_state_dict(
        self,
        state_dict: T.Dict[str, T.Any],
    ):
        """Load a pretrained model given a state dictionary."""

        loaded_names = set()

        # gather all base models
        nets = inspect.getmembers(self, lambda x: isinstance(x, BaseModel))  # name, net
        for name, net in nets:
            if name in state_dict:
                getattr(self, name).load_state_dict(state_dict[name])
            else:
                warnings.warn(f"{name} not exist in checkpoint")
            loaded_names.add(name)

        # gather all nn.modules
        nets = inspect.getmembers(self, lambda x: isinstance(x, torch.nn.Module))  # name, net
        for name, net in nets:
            if name in state_dict:
                try:
                    getattr(self, name).load_state_dict(state_dict[name])
                except:
                    new_sub_state_dict = OrderedDict()
                    for key, val in state_dict[name].items():
                        if key.startswith("module."):
                            new_key = key[7:]
                            new_sub_state_dict[new_key] = val
                        else:
                            new_sub_state_dict[key] = val
                    getattr(self, name).load_state_dict(new_sub_state_dict)
            else:
                warnings.warn(f"{name} not exist in checkpoint")
            loaded_names.add(name)

        # load those in var_names_to_load other things
        buffer_names = state_dict.get("buffer_names", set())
        parameter_names = state_dict.get("parameter_names", set())

        var_names_to_save = state_dict.get("var_names_to_save", set())
        self.register_var_to_save(
            var_name=var_names_to_save,
        )

        var_names_to_load = state_dict.get("var_names_to_load", set())
        self.register_var_to_save(
            var_name=var_names_to_load,
        )

        # load only those in var_names_to_load
        for name in var_names_to_load - loaded_names:
            if name not in state_dict:
                continue
            setattr(self, name, state_dict[name])

            if name in buffer_names:
                self.register_buffer(name, state_dict[name], persistent=True)

            if name in parameter_names:
                self.register_parameter(name, state_dict[name])

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""

        to_save = dict()

        # additional info to save
        to_save["buffer_names"] = getattr(self, "buffer_names", set())
        to_save["parameter_names"] = getattr(self, "parameter_names", set())
        to_save["var_names_to_save"] = getattr(self, "var_names_to_save", set())
        for var_name in to_save["var_names_to_save"]:
            to_save[var_name] = getattr(self, var_name, None)

        # additional info to load
        to_save["var_names_to_load"] = getattr(self, "var_names_to_load", set())
        for var_name in to_save["var_names_to_load"]:
            to_save[var_name] = getattr(self, var_name, None)

        # gather all base models
        nets = inspect.getmembers(self, lambda v: isinstance(v, BaseModel))  # name, net
        for name, net in nets:
            to_save[name] = net.state_dict()

        # gather all nn.modules
        nets = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))  # name, net
        for name, net in nets:
            if isinstance(net, DDP):
                to_save[name] = net.module.state_dict()
            elif isinstance(net, APEX_DDP):
                to_save[name] = net.module.state_dict()
            else:
                to_save[name] = net.state_dict()

        # save the classname
        to_save["classname"] = self.__class__.__name__

        return to_save

    def setup_for_distributed_learning(self, ddp_type: str):
        """
        Setup the networks in the model for distributed computation.
        It automactically gather all modules.
        """

        # gather all base models
        nets = inspect.getmembers(self, lambda v: isinstance(v, BaseModel))  # name, net
        for name, net in nets:
            print(f"setting up model {self.__class__.__name__}.{name} for distributed learning")
            net.setup_for_distributed_learning(ddp_type=ddp_type)

        if ddp_type == "ddp":
            # using pytorch distributed data parallel
            # gather all nn.modules
            nets = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))  # name, net
            for name, net in nets:
                print(f"setting {self.__class__.__name__}.{name} as DDP")
                first = None
                try:
                    first = next(net.parameters())
                except StopIteration:
                    pass
                if first is not None:  # DDP takes only modules that contain parameters
                    net = DDP(
                        getattr(self, name),
                        device_ids=[self.device],
                        output_device=self.device,
                    )
                    setattr(self, name, net)

        elif ddp_type == "apex_ddp":
            # using apex distributed data parallel
            # gather all nn.modules
            try:
                from apex.parallel import DistributedDataParallel as APEX_DDP
            except:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex")
            nets = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))  # name, net

            for name, net in nets:
                print(f"setting {self.__class__.__name__}.{name} as DDP")
                # By default, apex.parallel.DistributedDataParallel overlaps communication with
                # computation in the backward pass.
                # model = DDP(model)
                # delay_allreduce delays all communication to the end of the backward pass.
                net = APEX_DDP(getattr(self, name))
                setattr(self, name, net)

        elif ddp_type == "gradient_only":
            print("using apply_gradient_allreduce")
            from cdslib.core.utils.multigpu_utils import apply_gradient_allreduce

            # gather all nn.modules
            nets = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))  # name, net
            for name, net in nets:
                print(f"setting {self.__class__.__name__}.{name} as gradient-only ddp")
                net = apply_gradient_allreduce(getattr(self, name))
                setattr(self, name, net)
        else:
            raise NotImplementedError(f"{ddp_type}")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __str__(self):
        s = self.__class__.__name__

        # gather all base models
        nets = inspect.getmembers(self, lambda v: isinstance(v, BaseModel))  # name, net
        for name, net in nets:
            s += "\n"
            s += f"{name}: "
            s += str(net)

        # gather all nn.modules
        nets = inspect.getmembers(self, lambda v: isinstance(v, torch.nn.Module))  # name, net
        for name, net in nets:
            s += "\n"
            s += f"{name}: "
            s += str(net)

        return s

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Called during training.
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, *args, **kwargs):
        """
        Called during inference.
        """
        raise NotImplementedError
