# -*- coding: utf-8 -*-
# ---------------------

from time import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from conf import Conf
from dataset.ts_dataset import TSDataset
from models.temporal_fusion_t import tft_model
from progress_bar import ProgressBar
from utils import QuantileLoss, symmetric_mean_absolute_percentage_error, unnormalize_tensor, plot_temporal_serie
import data_formatters.utils as utils
from models.transformer import Transformer
from models.transformer_grn.transformer import Transformer as GRNTransformer


class Trainer(object):
    """
    Class for training and test the model
    """

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        torch.set_num_threads(3)

        self.cnf = cnf
        self.data_formatter = utils.make_data_formatter(cnf.ds_name)

        loader = TSDataset

        # init dataset
        dataset_train = loader(self.cnf, self.data_formatter)
        dataset_train.train()
        dataset_test = loader(self.cnf, self.data_formatter)
        dataset_test.test()

        # init model
        model_choice = self.cnf.all_params["model"]
        if model_choice == "transformer":
            # Baseline transformer
            self.model = Transformer(self.cnf.all_params)
        elif model_choice == "tf_transformer":
            # Temporal fusion transformer
            self.model = tft_model.TFT(self.cnf.all_params)
        elif model_choice == "grn_transformer":
            # Transformer + GRN to encode static vars
            self.model = GRNTransformer(self.cnf.all_params)
        else:
            raise NameError

        self.model = self.model.to(cnf.device)

        # init optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr)
        self.loss = QuantileLoss(cnf.quantiles)

        # init train loader
        self.train_loader = DataLoader(
            dataset=dataset_train, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=True, pin_memory=True,
        )

        # init test loader
        self.test_loader = DataLoader(
            dataset=dataset_test, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False, pin_memory=True,
        )

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)
        self.train_losses = []
        self.test_loss = []
        self.test_losses = {'p10': [], 'p50': [], 'p90': []}
        self.test_smape = []

        # starting values
        self.epoch = 0
        self.best_test_loss = None

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        # possibly load checkpoint
        self.load_ck()

        print("Finished preparing datasets.")

    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.progress_bar.current_epoch = self.epoch
            self.model.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            self.best_test_loss = self.best_test_loss

    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_test_loss': self.best_test_loss
        }
        torch.save(ck, self.log_path / 'training.ck')

    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        start_time = time()
        self.model.train()

        times = []
        for step, sample in enumerate(self.train_loader):
            t = time()
            self.optimizer.zero_grad()
            # Feed input to the model
            x = sample['inputs'].float().to(self.cnf.device)
            if self.cnf.all_params["model"] == "tf_transformer":
                output, _, _ = self.model.forward(x)
            else:
                output = self.model.forward(x)

            # Compute Loss
            loss, _ = self.loss(output.squeeze(), sample['outputs'].squeeze().float().to(self.cnf.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cnf.all_params['max_gradient_norm'])
            self.train_losses.append(loss.item())
            self.optimizer.step(None)

            # print an incredible progress bar
            times.append(time() - t)
            if self.cnf.log_each_step or (not self.cnf.log_each_step and self.progress_bar.progress == 1):
                print(f'\r{self.progress_bar} '
                      f'│ Loss: {np.mean(self.train_losses):.6f} '
                      f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')
            self.progress_bar.inc()

        # log average loss of this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
        self.train_losses = []

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')

    def test(self):
        """
        test model on the Test-Set
        """
        self.model.eval()
        output, sample = None, None

        t = time()
        for step, sample in enumerate(self.test_loader):

            # Hide future predictions from input vector, set to 0 (or 1) values where timestep > encoder_steps
            steps = self.cnf.all_params['num_encoder_steps']
            pred_len = sample['outputs'].shape[1]
            x = sample['inputs'].float().to(self.cnf.device)
            x[:, steps:, 0] = 1

            # Feed input to the model
            if self.cnf.all_params["model"] == "transformer" or self.cnf.all_params["model"] == "grn_transformer":

                # Auto-regressive prediction
                for i in range(pred_len):
                    output = self.model.forward(x)
                    x[:, steps + i, 0] = output[:, i, 1]
                output = self.model.forward(x)

            elif self.cnf.all_params["model"] == "tf_transformer":
                output, _, _ = self.model.forward(x)
            else:
                raise NameError

            output = output.squeeze()
            y, y_pred = sample['outputs'].squeeze().float().to(self.cnf.device), output

            # Compute loss
            loss, _ = self.loss(y_pred, y)
            smape = symmetric_mean_absolute_percentage_error(output[:, :, 1].detach().cpu().numpy(),
                                                             sample['outputs'][:, :, 0].detach().cpu().numpy())

            # De-Normalize to compute metrics
            target = unnormalize_tensor(self.data_formatter, y, sample['identifier'][0][0])
            p10_forecast = unnormalize_tensor(self.data_formatter, y_pred[..., 0], sample['identifier'][0][0])
            p50_forecast = unnormalize_tensor(self.data_formatter, y_pred[..., 1], sample['identifier'][0][0])
            p90_forecast = unnormalize_tensor(self.data_formatter, y_pred[..., 2], sample['identifier'][0][0])

            # Compute metrics
            self.test_losses['p10'].append(self.loss.numpy_normalised_quantile_loss(p10_forecast, target, 0.1))
            self.test_losses['p50'].append(self.loss.numpy_normalised_quantile_loss(p50_forecast, target, 0.5))
            self.test_losses['p90'].append(self.loss.numpy_normalised_quantile_loss(p90_forecast, target, 0.9))

            self.test_loss.append(loss.item())
            self.test_smape.append(smape)

        # Log stuff
        for k in self.test_losses.keys():
            mean_test_loss = np.mean(self.test_losses[k])
            print(f'\t● AVG {k} Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')
            self.sw.add_scalar(tag=k + '_test_loss', scalar_value=mean_test_loss, global_step=self.epoch)

        # log log log
        mean_test_loss = np.mean(self.test_loss)
        mean_smape = np.mean(self.test_smape)
        print(f'\t● AVG Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')
        print(f'\t● AVG SMAPE on TEST-set: {mean_smape:.6f} │ T: {time() - t:.2f} s')
        self.sw.add_scalar(tag='test_smape', scalar_value=mean_smape, global_step=self.epoch)
        self.sw.add_scalar(tag='test_loss', scalar_value=mean_test_loss, global_step=self.epoch)

        # save best model
        if self.best_test_loss is None or mean_test_loss < self.best_test_loss:
            self.best_test_loss = mean_test_loss
            torch.save(self.model.state_dict(), self.log_path / self.cnf.exp_name + '_best.pth')

    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()

            with torch.no_grad():
                self.test()

            self.epoch += 1
            self.save_ck()
