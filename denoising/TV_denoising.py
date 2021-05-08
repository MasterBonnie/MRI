import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.util import random_noise
from skimage.transform import resize
from skimage.restoration import denoise_tv_chambolle

from numba import njit

import Perona_Malik_diffusion

"""
    Implements TV denoising of images using the technique described in the paper 
        http://www.ipol.im/pub/art/2012/g-tvd/?utm_source=doi chapter 2
"""

epsilon = 1e-9

@njit(cache=True)
def minmod(a,b):
    return ((np.sign(a) + np.sign(b))/2) * np.minimum(np.abs(a), np.abs(b))

@njit(cache=True)
def x_norm_grad(img, i, j):
    dx_f = img[i+1][j] - img[i][j]

    dy_f = img[i][j+1] - img[i][j]
    dy_b = img[i][j] - img[i][j-1]

    dy = minmod(dy_f, dy_b)

    return dx_f / np.sqrt(dx_f**2 + dy**2 + epsilon)

@njit(cache=True)
def y_norm_grad(img, i, j):
    dy_f = img[i][j+1] - img[i][j]

    dx_f = img[i+1][j] - img[i][j]
    dx_b = img[i][j] - img[i-1][j]

    dx = minmod(dx_f, dx_b)

    return dy_f / np.sqrt(dy_f**2 + dx**2 + epsilon)


@njit(cache=True)
def filter_3_loop(img, data, dt, lam):
    """
        Takes one time step in the gradient descent PDE
    """

    res = np.zeros(img.shape)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            x_term = x_norm_grad(img, i, j) - x_norm_grad(img, i-1, j)
            y_term = y_norm_grad(img, i, j) - y_norm_grad(img, i, j-1)
            

            res[i][j] = img[i][j] + dt*(x_term + y_term) - dt*lam*(img[i][j] - data[i][j])

    # Boundary conditions
    for i in range(img.shape[0]):
        res[i][0] = res[i][img.shape[1]-2]
        res[i][img.shape[1]-1] = res[i][img.shape[1]-2]

    for j in range(img.shape[1]):
        res[0][j] = res[1][j]
        res[img.shape[0]-1][j] = res[img.shape[0]-2][j]

    return res

def filter(data, dt, lam):

    mean_data = np.mean(data)
    u = np.zeros(data.shape) + mean_data

    for k in range(niter+1):
        if k % 100 == 0:
            print(f"Iteration {k}    ", end = "\r")
        u = filter_3_loop(u, data, dt, lam)

    return u

if __name__ == "__main__":
    
    sigma = 0.1 
    n = 256
    
    # noisy image
    f = resize(data.camera(),(n,n))
    f_delta = random_noise(f,var=sigma**2)

    # Parameters
    dt = 1e-6
    niter = 1000
    lam = 10

    # solve evolution equation
    f_f = filter(f_delta, dt, lam)

    # Reference denoising
    f_test = Perona_Malik_diffusion.filter(f_delta)

    # plot
    fig,ax = plt.subplots(1,3)

    ax[0].imshow(f_delta)
    ax[0].set_title('Noisy image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(f_f)
    ax[1].set_title('Result')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(f_test)
    ax[2].set_title('Perona-Malik')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    plt.show()