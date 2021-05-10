import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from numba import njit
from skimage import data
from skimage.util import img_as_ubyte

import torch
from torch import nn
import torchvision

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

        self.conv3 = nn.Conv2d(6, 3, 5, padding=2)
        self.non_linear_3 = nn.ReLU()

        self.conv4 = nn.Conv2d(3, 1 , 5, padding=2)
        self.non_linear_4 = nn.ReLU()

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

        x = self.conv4(x)
        x = self.non_linear_4(x)

        return x
    
    def loss_function(self, x, y):
        return self.loss(x,y)

    def _calculate_mask(self, sub_image_size=40, stride=12):
        self.mask = torch.zeros(256,256)

        for i in range((256 - sub_image_size)//stride + 1):
            for j in range((256 - sub_image_size)//stride + 1):
                lower_index_x = i*stride
                upper_index_x = i*stride + sub_image_size

                lower_index_y = j*stride
                upper_index_y = j*stride + sub_image_size
                
                self.mask[lower_index_x:upper_index_x, lower_index_y:upper_index_y] += 1

    def reconstruct_full_image(self, full_img_batch, noisy_img_batch, writer, device, sub_image_size=40, stride=12):

        n = min(full_img_batch.size(0), 8)
        result = torch.zeros(8,1,256,256)

        self._calculate_mask(sub_image_size, stride)

        for index in range(n):
            img = noisy_img_batch[index, 0]

            for i in range((img.shape[0] - sub_image_size)//stride + 1):
                for j in range((img.shape[1] - sub_image_size)//stride + 1):
                    lower_index_x = i*stride
                    upper_index_x = i*stride + sub_image_size

                    lower_index_y = j*stride
                    upper_index_y = j*stride + sub_image_size
                    sub_image = torch.zeros(1,1,40,40, dtype=torch.double)
                    sub_image[0,0] = img[lower_index_x:upper_index_x, lower_index_y:upper_index_y]
                    sub_image = sub_image.to(device)

                    denoised_image = self.forward(sub_image)
                    denoised_image = denoised_image.cpu()

                    result[index, 0, lower_index_x:upper_index_x, lower_index_y:upper_index_y] += denoised_image[0,0]
            
            result[index, 0] /= self.mask

            # Normalize for Tensorboard
            result[index, 0] /= torch.max(result[index, 0])
            full_img_batch[index, 0] /= torch.max(full_img_batch[index, 0])

        comparison = torch.cat([full_img_batch[:n],
                                result])

        img_grid = torchvision.utils.make_grid(comparison.cpu(), nrows=2)
        writer.add_image(f"Original images and neural network denoising", img_grid) 
        writer.close()