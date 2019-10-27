'''Module to learn a steerable basis'''
import os
import sys
import time

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F

from scipy.special import binom


class Dconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, base, io_scales, 
                 stride=1, padding=1, bias=False, pad_mode='constant'):
        """Create Dconv2d object        

        Args:
            in_channels: ...
            out_channels: ...
            kernel_size: tuple (scales, height, width)
            base: float for downscaling factor
            io_scales: tuple (num_out_scales, num_in_scales)
            stride: ...
            padding: ...
            bias: bool
            pad_mode: ...
        """
        super(Dconv2d, self).__init__()
        # Channel info
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Kernel sizes
        self.kernel_scales = kernel_size[0]
        self.kernel_size = kernel_size[1:]
        # Needed to compute padding of dilated convs
        self.overlap = [self.kernel_size[0]//2, self.kernel_size[1]//2]
        self.io_scales = io_scales.copy()
        # Compute the dilations needed in the scale-conv
        dilations = np.power(base, np.arange(io_scales[1]))
        self.dilations = [int(d) for d in dilations]
        # Basic info
        self.stride = stride
        self.padding = [padding,padding]
        self.pad_mode = pad_mode
        # The weights
        weight_shape = (out_channels, in_channels, self.kernel_scales, self.kernel_size[0], self.kernel_size[1])
        self.weights = Parameter(torch.Tensor(*weight_shape))
        # Optional bias
        if bias == True:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()

    def __repr__(self):
        return ('{name}({in_channels}->{out_channels}, {kernel_scales}, {kernel_size}, ' 
                'dilations={dilations}, pad_mode={pad_mode})'
                .format(name=self.__class__.__name__, **self.__dict__))

    def reset_parameters(self):
        """
        # Custom Yu/Koltun-initialization
        stdv = 1e-2
        wsh = self.weights.size()
        self.weights.data.uniform_(-stdv, stdv)

        C = np.gcd(self.in_channels, self.out_channels)
        val = C / (self.out_channels)
        
        ci = self.kernel_size[0] // 2
        cj = self.kernel_size[1] // 2
        for b in range(self.out_channels):
            for a in range(self.in_channels):
                if np.floor(a*C/self.in_channels) == np.floor(b*C/self.out_channels):
                    self.weights.data[b,a,:,ci,cj] = val
                else:
                    pass
        """
        # Just your standard He initialization
        n = self.kernel_size[0] * self.kernel_size[1] * self.kernel_scales * self.in_channels
        self.weights.data.normal_(0, math.sqrt(2. / n))

        if self.bias is not None:
            self.bias.data.fill_(1)


    def forward(self, input):
        """Implement a scale conv the slow way

        Args:
            inputs: [batch, channels, scale, height, width]
        Returns:
            inputs: [batch, channels, scale, height, width]
        """
        # Dilations
        dilation = [(self.dilations[d], self.dilations[d]) for d in range(len(self.dilations))]
        # Number of scales in and out
        sin = self.io_scales[1]
        sout = self.io_scales[0]

        outputs = []
        # d is the index in the kernel, s is the index in the output
        for s in range(sout):
            # Cut out slices from the input
            t = np.minimum(s + self.kernel_scales, sout)
            x = input[:,:,s:t,:,:].reshape(input.size()[0],-1,input.size()[3],input.size()[4])
            # Cut out the weights
            weight_shape = (self.out_channels, self.in_channels*(t-s), self.kernel_size[0], self.kernel_size[1])
            w = self.weights[:,:,:t-s,:,:].reshape(weight_shape)
            # Convolve for one output scale, using appropriate padding
            padding = [int(dilation[s][0]*self.overlap[0]), int(dilation[s][1]*self.overlap[1])]
            outputs.append(F.conv2d(x, w, bias=self.bias, stride=self.stride, 
                                    padding=padding, dilation=dilation[s]))

        return torch.stack(outputs, 2)


class BesselConv2d(nn.Module):
    """Convolution with the discrete Gaussian of Lindeberg

    The discrete Gaussian is of the form: exp{-t} I_{x}(t), where t is the
    scale parameter (= sigma**2) and x is the integer position of the 
    filter taps. This filter allows us to have fine-grained control over the 
    scales at low blurs.

    The term I_{x}(t) is the modified Bessel function of first kind and 
    integer order. We can implement the entire function using 
    scipy.special.ive which is pretty handy.
    """
    def __init__(self, base=2., zero_scale=0.5, n_scales=8, scales=None):
        """Create a BesselConv2d object

        Args:
            base: float for factor to downscaling
            zero_scale: float for scale of input 
            n_scales: int for number of scales
            scales: optional pre-computed scales
        """
        super(BesselConv2d, self).__init__()

        if scales is not None:
            self.scales = scales
            self.base = None
            self.zero_scale = None
        else:
            self.base = base
            self.zero_scale = zero_scale
            self.n_scales = n_scales
            k = np.arange(1, n_scales)
            dilations = np.power(base, k)
            self.scales = (zero_scale**2)*(dilations**2 - 1.)

        print("Bessel scales: {}".format(self.scales))
        self.widths = np.asarray([4*int(np.ceil(np.sqrt(scale))) for scale in self.scales])
        self.weights = self._get_blur()

    def forward(self, input):
        """For now we do it the slow way

        Args:
            input: [batch, channels, height, width] tensor
        Returns:
            [batch, channels, scale, height, width] tensor
        """
        if self.scales != []:
            pad = self.widths
            output = [F.conv2d(input, self.weights[d][0].cuda(), bias=None, 
                   padding=(0,pad[d]), stride=1, dilation=1) for d in range(len(pad))]
            output = [F.conv2d(output[d], self.weights[d][1].cuda(), bias=None,
                   padding=(pad[d], 0), stride=1, dilation=1) for d in range(len(pad))]

            output = torch.stack(output, dim=2)
            input = torch.unsqueeze(input, 2)
            output = torch.cat([input, output], 2)
        else:
            output = torch.unsqueeze(input, 2)
        return output


    def _np2torch(self, x):
        return torch.from_numpy(x).type(torch.FloatTensor)


    def _get_blur(self):
        """Return a discrete gaussian blur conv with size number of pixels

        Returns:
            a list of kernels
        """
        from scipy.special import ive
        kernels = []
        for scale, width in zip(self.scales, self.widths):
            # Create 1D kernel first
            x = np.arange(-width, width+1)
            kernel = ive(np.abs(x), scale)
            kernel = kernel / np.sum(kernel)

            # Create x- and y-kernels
            kernelx = self._np2torch(kernel[np.newaxis,:])
            kernely = self._np2torch(kernel[:,np.newaxis])

            # This converts them to RGB-kernels...is this the best way?
            kernelx = kernelx.view(1,1,1,2*width+1)*torch.eye(3).view(3,3,1,1)
            kernely = kernely.view(1,1,2*width+1,1)*torch.eye(3).view(3,3,1,1)

            kernels.append((kernelx, kernely))

        return kernels
