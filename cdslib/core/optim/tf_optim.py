#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
# Author: Rick Chang
# This file implements a optimizer that has a warm-up period.

import torch.optim


class TFOptimizer:
    """
    A wrapper around torch.optim.Optimizer to adjust learning rate with a warm-up period.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int = 512,
        factor: float = 1.0,
        warmup: int = 2000,
        init_step: int = 0,
        use_dynamic_rate: bool = True,
    ):
        r"""
        Vary the learning rate every training step according to:

            .. math::
                lr = \text{factor} * d_{model}^{-0.5} * \min(\text{iter}^{-0.5}, \text{iter} * \text{warmup}^{-1.5})

        This corresponds to linearly increasing the learning rate for the first `warmup` training steps,
        and after that decreasing it proportionally to the inverse square root of the iteration (step).
        This optimizer was used by Vaswani, Ashish, et al. ("Attention is all you need." 2017).


        Args:
            optimizer (torch.optim.Optimizer):
                pytorch optimizer to warp around
            model_size (int):
                primary dimension of the model
            factor (float):
                scalar to multiply. It is to scale the learning rate.
            warmup (int):
                number of warm up steps to linearly increase the learning rate with
            init_step (int):
                initial step (for training recovery)
            use_dynamic_rate (bool):
                whether to (True) use the increasing-decreasing schedule
                or (False) the constant learning rate

        Examples:
            .. code-block:: python

                use_amp = True  # whether to use Automatic Mixed Precision
                net = SomeModel(...)
                parameters = net.parameters()

                # Create a pytorch optimizer
                # (it will handle everything other than learning rate,
                # e.g., normalization, momentum, etc)
                _optimizer = torch.optim.Adam(
                        parameters,  # parameters to optimize
                        lr=1e-3, # this will be overwritten by our optimzier
                        betas=(0.9, 0.98),
                        eps=1e-9)

                # wrap our learning rate scheduler
                optimizer = TFOptimizer(
                        optimizer=_optimizer,
                        model_size=512,
                        factor=1.0,
                        warmup=4000,
                        init_step=0)

                if use_amp:
                    scaler = torch.cuda.amp.GradScaler()  # only if using amp
                else:
                    scaler = None

                with torch.cuda.amp.autocast(enabled=use_amp):  # only if using amp
                    loss = net(x).sum()

                # zero grad
                optimizer.zero_grad()

                # compute gradient
                if use_amp:
                    scaler.scale(loss).backward()
                    optimizer.unscale(scaler=scaler)  #
                else:
                    loss.backward()

                # if gradient clipping
                nn.utils.clip_grad_norm_(parameters, max_grad_val)

                # update parameters
                optimizer.step(scaler=scaler)

                # update scaler
                if use_amp:
                    scaler.update()

        """
        self.optimizer = optimizer
        self.set_init_step(init_step=init_step)
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.use_dynamic_rate = use_dynamic_rate

    def set_init_step(self, init_step: int):
        """Set the initial step."""
        self._step = init_step

    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradient buffer."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, scaler=None):
        """Update parameters and learning rate."""
        self._step += 1
        if self.use_dynamic_rate:
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p["lr"] = rate
            self._rate = rate
            if scaler is None:
                self.optimizer.step()
            else:
                scaler.step(self.optimizer)
        else:
            if scaler is None:
                self.optimizer.step()
            else:
                scaler.step(self.optimizer)
            rates = []
            for p in self.optimizer.param_groups:
                rates.append(p["lr"])
            self._rate = sum(rates) / len(rates)

    def rate(self, step: int = None) -> float:
        """
        Compute the learning rate at a given step.

        Args:
            step (int or None):
                the step to compute the learning rate.
                If None, use the current step of the optimizer.

        Returns:
            learning rate (float)
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def get_current_lr(self):
        """Get the current learning rate."""
        return self._rate

    def unscale(self, scaler=None):
        """
        Unscale the gradient of the parameter controlled by the optimizer.
        Should only be called once per iteration.
        """
        if scaler is None:
            return
        else:
            scaler.unscale_(self.optimizer)
