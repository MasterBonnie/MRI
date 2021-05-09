import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from numba import njit
from skimage import data
from skimage.util import img_as_ubyte

import torch
from torch import nn


class MRINetwork(nn.Module):
    """ Neural network for MRI image processing """

    def __init__(self, input_dim, hdims, loss):

        super(MRINetwork, self).__init__()
        
        self.input_dim = input_dim

        # Variable defining the encoder / decoder network
        layers = []

        layers.append(nn.Linear(input_dim, hdims[0]))
        layers.append(nn.ReLU())

        for i in range(len(hdims) - 1):
            layers.append(nn.Linear(hdims[i], hdims[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hdims[-1], input_dim))
        self.feedforward_network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        self.loss = loss

    def __str__(self):
        return " ReLU denoising network "

    def forward(self, x):
        """
        Forward pass through the network
        """
        x = self.flatten(x)
        res = self.feedforward_network(x)
        return res

    def loss_function(self, x, y):
        y = self.flatten(y)
        return self.loss(x,y)



class MRIConvolutionalNetwork(nn.Module):
    """ Convolutional Neural Network """

    def __init__(self, loss):
        super(MRIConvolutionalNetwork, self).__init__()

        self.loss = loss
        # We use three convolutional layers,
        # each which does not increase the number of channels
        # After each of these conv. layers we apply a non-linear transform
        self.conv1 = nn.Conv2d(1, 3, 9,padding=4)
        self.non_linear_1 = nn.ReLU()

        self.conv2 = nn.Conv2d(3, 6, 5, padding=2)
        self.non_linear_2 = nn.ReLU()

        self.conv3 = nn.Conv2d(6, 1 , 5, padding=2)
        self.non_linear_3 = nn.ReLU()

    def __str__(self):
        return " Convolutional denoising network "

    def forward(self, x):
        """
        Forward pass through the network
        """
        x = self.conv1(x)
        x = self.non_linear_1(x)

        x = self.conv2(x)
        x = self.non_linear_2(x)

        x = self.conv3(x)
        x = self.non_linear_3(x)
        
        return x
    
    def loss_function(self, x, y):
        return self.loss(x,y)