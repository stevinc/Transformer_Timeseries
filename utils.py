# -*- coding: utf-8 -*-
# ---------------------

import json
import os
from datetime import datetime
from enum import Enum
from typing import *
from typing import Callable, List, TypeVar

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL.Image import Image
from matplotlib import cm
from matplotlib import figure
from pathlib import Path
from torch import Tensor
from torch import nn
from torchvision.transforms import ToTensor


class QuantileLoss(nn.Module):
    ## From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles

    def numpy_normalised_quantile_loss(self, y_pred, y, quantile):
        """Computes normalised quantile loss for numpy arrays.
        Uses the q-Risk metric as defined in the "Training Procedure" section of the
        main TFT paper.
        Args:
          y: Targets
          y_pred: Predictions
          quantile: Quantile to use for loss calculations (between 0 & 1)
        Returns:
          Float for normalised quantile loss.
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        if len(y_pred.shape) == 3:
            ix = self.quantiles.index(quantile)
            y_pred = y_pred[..., ix]

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        prediction_underflow = y - y_pred
        weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
                          + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

        quantile_loss = weighted_errors.mean()
        normaliser = np.abs(y).mean()

        return 2 * quantile_loss / normaliser

    def forward(self, preds, target, ret_losses=True):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        if ret_losses:
            return loss, losses
        return loss


def unnormalize_tensor(data_formatter, data, identifier):
    data = pd.DataFrame(
        data.detach().cpu().numpy(),
        columns=[
            't+{}'.format(i)
            for i in range(data.shape[1])
        ])

    data['identifier'] = np.array(identifier)
    data = data_formatter.format_predictions(data)

    return data.drop(columns=['identifier']).values


def symmetric_mean_absolute_percentage_error(forecast, actual):
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    sequence_length = forecast.shape[1]
    sumf = np.sum(np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)), axis=1)
    return np.mean((2 * sumf) / sequence_length)


def plot_temporal_serie(y_pred, y_true):
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if isinstance(y_true, Tensor):
        y_true = y_true.detach().cpu().numpy()

    ind = np.random.choice(y_pred.shape[0])
    plt.plot(y_pred[ind, :, 0], label='pred_1')
    plt.plot(y_pred[ind, :, 1], label='pred_5')
    plt.plot(y_pred[ind, :, 2], label='pred_9')

    plt.plot(y_true[ind, :, 0], label='true')
    plt.legend()
    plt.show()


def imread(path):
    # type: (Union[Path, str]) -> Image
    """
    Reads the image located in `path`
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def pyplot_to_numpy(pyplot_figure):
    # type: (figure.Figure) -> np.ndarray
    """
    Converts a PyPlot figure into a NumPy array
    :param pyplot_figure: figure you want to convert
    :return: converted NumPy array
    """
    pyplot_figure.canvas.draw()
    x = np.fromstring(pyplot_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    x = x.reshape(pyplot_figure.canvas.get_width_height()[::-1] + (3,))
    return x


def pyplot_to_tensor(pyplot_figure):
    # type: (figure.Figure) -> Tensor
    """
    Converts a PyPlot figure into a PyTorch tensor
    :param pyplot_figure: figure you want to convert
    :return: converted PyTorch tensor
    """
    x = pyplot_to_numpy(pyplot_figure=pyplot_figure)
    x = ToTensor()(x)
    return x


def apply_colormap_to_tensor(x, cmap='jet', range=(None, None)):
    # type: (Tensor, str, Optional[Tuple[float, float]]) -> Tensor
    """
    :param x: Tensor with shape (1, H, W)
    :param cmap: name of the color map you want to apply
    :param range: tuple of (minimum possible value in x, maximum possible value in x)
    :return: Tensor with shape (3, H, W)
    """
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=range[0], vmax=range[1])
    x = x.detatch().cpu().numpy()
    x = x.squeeze()
    x = cmap.to_rgba(x)[:, :, :-1]
    return ToTensor()(x)

