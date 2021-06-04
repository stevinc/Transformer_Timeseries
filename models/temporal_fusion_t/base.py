# -*- coding: utf-8 -*-
# ---------------------

from abc import ABCMeta
from abc import abstractmethod
from typing import Union

import torch
from path import Path
from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()


    def kaiming_init(self, activation):
        # type: (str) -> ()
        """
        Apply "Kaiming-Normal" initialization to all Conv2D(s) of the model.
        :param activation: activation function after conv; values in {'relu', 'leaky_relu'}
        :return:
        """
        assert activation in ['ReLU', 'LeakyReLU', 'leaky_relu'], \
            '`activation` must be \'ReLU\' or \'LeakyReLU\''

        if activation == 'LeakyReLU':
            activation = 'leaky_relu'
        activation = activation.lower()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    @abstractmethod
    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        ...


    @property
    def n_param(self):
        # type: (BaseModel) -> int
        """
        :return: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    @property
    def current_device(self):
        # type: () -> str
        """
        :return: string that represents the device on which the model is currently located
            >> e.g.: 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...
        """
        return str(next(self.parameters()).device)


    @property
    def is_cuda(self):
        # type: () -> bool
        """
        :return: `True` if the model is on Cuda; `False` otherwise
        """
        return 'cuda' in self.current_device


    def save_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        save model weights in the specified path
        """
        torch.save(self.state_dict(), path)


    def load_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        load model weights from the specified path
        """
        self.load_state_dict(torch.load(path))


    def requires_grad(self, flag):
        # type: (bool) -> None
        """
        :param flag: True if the model requires gradient, False otherwise
        """
        for p in self.parameters():
            p.requires_grad = flag
