# losely based on Temporal Convolutional Networks (TCN)

import torch

from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from torch import autograd

from torch import nn

from PIL import Image

class CausalConv1d(nn.Conv1d):
    """
    Simple 1d causal convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.slice_padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=self.slice_padding,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        return result[:, :, :-self.slice_padding]

class CausalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.1):
        super(CausalBlock, self).__init__()
        self.layer = nn.Sequential(CausalConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation),
                                   nn.ReLU(),
                                   nn.Dropout2d(dropout),
                                   CausalConv1d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation),
                                   nn.ReLU(),
                                   nn.Dropout2d(dropout),
                                   CausalConv1d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation),
                                   nn.ReLU(),
                                   nn.Dropout2d(dropout))
        # downsampling for residual
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        o = self.layer(x)
        o = torch.relu(o + self.residual(x) if self.residual else o + x)
        return o

class CausalNet(nn.Module):

    def __init__(self, input_size, output_size, out_channels, layers, kernel_size, dropout=0.1):
        super(CausalNet, self).__init__()
        l = []
        l.append(CausalBlock(input_size, out_channels, kernel_size, stride=1, dilation=1, dropout=dropout))
        for i in range(layers):
            dilation_size = 2 ** i
            l.append(CausalBlock(out_channels, out_channels, kernel_size,
                                 stride=1, dilation=dilation_size,
                                 dropout=dropout))
        self.lin = nn.Linear(out_channels, output_size)
        self.net = nn.Sequential(*l)

    def forward(self, x):
        o = self.net(x)
        o = self.lin(o.transpose(1, 2))
        return o.transpose(1, 2)
