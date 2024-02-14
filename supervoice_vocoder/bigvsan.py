import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import spectral_norm

from .activations import SnakeBeta, Snake
from .san_modules import SANConv2d
from .amp import AMPBlock1, AMPBlock2
from .alias_free import *
from .utils import dict_to_object

class BigVSAN(torch.nn.Module):
    def __init__(self):
        super(BigVSAN, self).__init__()

        # Config of the network
        h = dict_to_object({
            "resblock": "1",
            "upsample_rates": [4,4,2,2,2,2],
            "upsample_kernel_sizes": [8,8,4,4,4,4],
            "upsample_initial_channel": 1536,
            "resblock_kernel_sizes": [3,7,11],
            "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
            "activation": "snake",
            "snake_logscale": False,
            "num_mels": 100,
        })

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # pre conv
        self.conv_pre = Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)

        # define which AMPBlock to use. BigVSAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                            h.upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2)
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))

        # post conv
        if h.activation == "snake": # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif h.activation == "snakebeta": # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3)


    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x