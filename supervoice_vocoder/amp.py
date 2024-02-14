import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import spectral_norm

from .activations import SnakeBeta, Snake
from .san_modules import SANConv2d
from .alias_free import *

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
        self.h = h

        self.convs1 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))
        ])

        self.convs2 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)),
            Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)),
            Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))
        ])

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x


class AMPBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        self.h = h

        self.convs = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))
        ])

        self.num_layers = len(self.convs) # total number of conv layers

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        for c, a in zip (self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

