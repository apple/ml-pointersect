#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#

import unittest
from cdslib.core.script.base_train import BaseTrainProcess, customer
import torch
import torch.utils.data
import cdslib.core.nn as cdsnn
import typing as T
import tempfile
import os
import torch.multiprocessing as mp
from cdslib.core.utils.print_and_save import print1D


class QuickDataset(torch.utils.data.Dataset):
    def __init__(self, xs: torch.Tensor, ys: torch.Tensor):
        self.xs = xs
        self.ys = ys
        assert self.xs.size(0) == self.ys.size(0)

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, i):
        return (
            self.xs[i],  # (dim_x,)
            self.ys[i],  # float
        )


class LinearRegressionTrain(BaseTrainProcess):
    r"""
    .. math::
        \min_x || y - X * x ||^2
    """

    def __init__(
            self,
            dim_w: int,
            dim_y: int,  # dataset size
            batch_size: int,
            lr: float = 1e-3,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dim_w = dim_w
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.lr = lr
        self._register_var_to_save(["dim_w", "dim_y"])

        self._construct_data()

    def _construct_data(self):
        g = torch.Generator()
        g.manual_seed(0)
        self.gt_w = torch.randn(self.dim_w, generator=g)  # (dim_w,)

        self.xs = torch.randn(self.dim_y, self.dim_w, generator=g)  # (n, dim_w)
        self.ys = self.xs @ self.gt_w  # (n,)

        self.xs_valid = torch.randn(self.dim_y, self.dim_w, generator=g)  # (n, dim_w)
        self.ys_valid = self.xs_valid @ self.gt_w  # (n,)

        self.xs_test = torch.randn(self.dim_y, self.dim_w, generator=g)  # (n, dim_w)
        self.ys_test = self.xs_test @ self.gt_w  # (n,)

    @customer
    def get_dataloaders(self):
        dset = QuickDataset(xs=self.xs, ys=self.ys)
        dset_valid = QuickDataset(xs=self.xs_valid, ys=self.ys_valid)
        dset_test = QuickDataset(xs=self.xs_test, ys=self.ys_test)

        if self.process_info['distributed_run']:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=dset,
                num_replicas=self.process_info['n_gpus'],
                rank=self.process_info['rank'],
                seed=0,
                shuffle=True,
                drop_last=False,
            )
            self.valid_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=dset_valid,
                num_replicas=self.process_info['n_gpus'],
                rank=self.process_info['rank'],
                seed=0,
                shuffle=True,
                drop_last=False,
            )
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=dset_test,
                num_replicas=self.process_info['n_gpus'],
                rank=self.process_info['rank'],
                seed=0,
                shuffle=True,
                drop_last=False,
            )
        else:
            self.train_sampler = None
            self.valid_sampler = None
            self.test_sampler = None

        dataloader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset=dset_valid,
            batch_size=self.batch_size,
            shuffle=(self.valid_sampler is None),
            sampler=self.valid_sampler,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=dset_test,
            batch_size=self.batch_size,
            shuffle=(self.test_sampler is None),
            sampler=self.test_sampler,
        )

        return dataloader, val_dataloader, test_dataloader

    @customer
    def construct_models(self):
        self.net = cdsnn.LinearLayer(
            in_features=self.dim_w,
            out_features=1,
            bias=False,
        )

        self.loss_values = dict()
        self.step_values = dict()
        self.step_count = 0

    @customer
    def construct_optimizers(self):
        self.optimizer = torch.optim.SGD(
            params=self.net.parameters(),
            lr=self.lr,
            # eps=1e-5,
        )

    @customer
    def epoch_setup(
            self,
            epoch: int,
            dataloader: T.Sequence[T.Any],
            val_dataloader: T.Sequence[T.Any],
            test_dataloader: T.Sequence[T.Any],
    ):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        if self.valid_sampler is not None:
            self.valid_sampler.set_epoch(epoch)
        if self.test_sampler is not None:
            self.test_sampler.set_epoch(epoch)

    def _step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,  # xs, ys:  (n, dim_x), (n,)
            mode: str,
    ) -> T.Dict[str, T.Any]:
        """One training step.
        Return a dictionary that will be passed to logging.
        """
        xs, ys_gt = batch  # (n, dim_x), (n,)
        xs = xs.to(self.device)
        ys_gt = ys_gt.to(self.device)
        ys = self.net(xs)
        loss = (ys_gt - ys.squeeze(-1)).pow(2).mean()

        if mode == 'train':
            # step optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            pass

        if mode not in self.loss_values:
            self.loss_values[mode] = []
            self.step_values[mode] = []

        if mode == 'train':
            self.step_count += 1

        self.loss_values[mode].append(loss.detach().cpu().item())
        self.step_values[mode].append(self.step_count)

        return dict(
            loss=loss.detach().cpu().item()
        )

    @customer
    def train_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ) -> T.Dict[str, T.Any]:
        return self._step(
            epoch=epoch,
            bidx=bidx,
            batch=batch,
            mode='train',
        )

    @customer
    def validation_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ) -> T.Dict[str, T.Any]:
        return self._step(
            epoch=epoch,
            bidx=bidx,
            batch=batch,
            mode='valid',
        )

    @customer
    def test_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ) -> T.Dict[str, T.Any]:
        return self._step(
            epoch=epoch,
            bidx=bidx,
            batch=batch,
            mode='test',
        )

    @customer
    def finish_procedure(self):
        for name in ['train']:
            print(f'{name}: {len(self.step_values[name])}')
            print1D(self.step_values[name], format='>6d')
            print1D(self.loss_values[name], format='>6.3f')
            print('')


class TestBaseTrainProcess(unittest.TestCase):
    """
    In the test, we will use linear regression to test BaseTrainProcess.
    """

    @staticmethod
    def _test(
            rank=0,
            dim_w=30,
            num_samples=1000,
            batch_size=32,
            lr=1e-1,
            num_epoch=100,
            exp_tag='test',
            n_gpus=0,
    ):

        with tempfile.TemporaryDirectory() as output_dir:
            with LinearRegressionTrain(
                dim_w=dim_w,
                dim_y=num_samples,
                rank=rank,
                n_gpus=n_gpus,
                batch_size=batch_size,
                lr=lr,
                exp_tag=exp_tag,
                config_filename=None,
                output_dir=output_dir,
                end_epoch=num_epoch,
                exp_tag_first=True,
                log_every_num_train_batch=100,
                log_every_num_valid_batch=100,
                log_every_num_test_batch=100,
                visualize_every_num_train_batch=1000,
                visualize_every_num_valid_batch=1000,
                visualize_every_num_test_batch=1000,
                validate_every_num_epoch=10,
                test_every_num_epoch=num_epoch,
                use_bolt=False,
                use_torchrun=False,
            ) as trainer:
                trainer.run()
                gt_w = trainer.gt_w

            # load the trained model
            if rank == 0:
                filename = os.path.join(
                    output_dir, exp_tag, 'checkpoint', f'epoch{num_epoch - 1}.pth')
                with LinearRegressionTrain(
                        dim_w=dim_w,
                        dim_y=num_samples,
                        batch_size=batch_size,
                        lr=lr,
                        exp_tag='test',
                        config_filename=None,
                        output_dir=output_dir,
                ) as trainer:
                    trainer.construct_models()
                    trainer.load(filename=filename)
                    est_w = trainer.net.weight.clone().detach()
                    assert torch.allclose(est_w, gt_w, rtol=0.1, atol=0.1)

            # continue training
            if rank == 0:
                filename = os.path.join(
                    output_dir, exp_tag, 'checkpoint', f'epoch{num_epoch}.pth')
                with tempfile.TemporaryDirectory() as output_dir:
                    with LinearRegressionTrain(
                        dim_w=dim_w,
                        dim_y=num_samples,
                        rank=rank,
                        n_gpus=1 if n_gpus >= 1 else 0,
                        batch_size=batch_size,
                        lr=lr,
                        exp_tag=exp_tag,
                        config_filename=None,
                        output_dir=output_dir,
                        end_epoch=num_epoch + 1,
                        exp_tag_first=True,
                        log_every_num_train_batch=100,
                        log_every_num_valid_batch=100,
                        log_every_num_test_batch=100,
                        visualize_every_num_train_batch=1000,
                        visualize_every_num_valid_batch=1000,
                        visualize_every_num_test_batch=1000,
                        validate_every_num_epoch=10,
                        test_every_num_epoch=num_epoch,
                        use_bolt=False,
                        trainer_filename=filename,
                    ) as trainer:
                        trainer.run()
                        gt_w = trainer.gt_w.cpu()
                        est_w = trainer.net.weight.clone().detach().cpu()
                        assert torch.allclose(est_w, gt_w, rtol=0.1, atol=0.1)
                        filename = os.path.join(
                            output_dir, exp_tag, 'checkpoint', f'epoch{num_epoch + 1}.pth')
                        assert os.path.exists(filename)

    def start_process(
            self,
            dim_w=30,
            num_samples=1000,
            batch_size=32,
            lr=1e-1,
            num_epoch=100,
            exp_tag='test',
            n_gpus=0,
    ):
        mp.spawn(
            TestBaseTrainProcess._test,
            args=(
                dim_w,
                num_samples,
                batch_size,
                lr,
                num_epoch,
                exp_tag,
                n_gpus,
            ),
            nprocs=n_gpus,
            join=True,
        )

    def test1(self):
        """Test cpu-only version."""
        self._test(n_gpus=0)

    def test2(self):
        """Test cuda version."""
        if torch.cuda.is_available():
            self._test(n_gpus=1)

    def test3(self):
        """Test cuda distributed version."""
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            self.start_process(n_gpus=min(2, torch.cuda.device_count()))


if __name__ == '__main__':
    unittest.main()
