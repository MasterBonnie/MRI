import matplotlib.pyplot as plt
import numpy as np

from numba import njit

"""
    File containing different cost functions and their gradients
"""

# Regularizing parameter for the gradient of the TV norm
epsilon = 1e-8

# Create the mask for undersampling the data 
mask_indices = np.loadtxt("..\mask1.txt", dtype=np.int)

mask = np.zeros((256, 256))
mask[mask_indices[:,0], mask_indices[:,1]] = 1

def undersample_fourier(img):
    """
        Implementation of the "Sampling fourier transform"

        First applies the fourier transform to the image, then selects only those
        indices that are non-zero according to the specified mask
    """
    fourier_img = np.fft.fftshift(np.fft.fft2(img, norm="ortho"))

    return fourier_img[mask_indices[:,0], mask_indices[:,1]]


def undersample_fourier_adjoint(x):
    """
        Implementation of the "Adjoint Sampling fourier transform"

        This is the adjoint operator of the above function.
    """

    k_space = np.zeros((256,256), dtype=np.cdouble)
    k_space[mask_indices[:,0], mask_indices[:,1]] = x

    return np.abs(np.fft.ifft2(k_space, norm="ortho"))

@njit
def com_dot(a,b):
    return np.conjugate(a)*b

@njit
def gradient_l1(m):
    res = np.zeros(m.shape)

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            res[i][j] = m[i][j]/(np.sqrt(np.real(com_dot(m[i][j],m[i][j])) + epsilon))

    return res

@njit
def l1_DGT(img):

    res = 0

    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            res += np.sqrt( (img[i][j] - img[i-1][j])**2 + (img[i][j] - img[i][j-1])**2)

    return res

@njit
def grad_l1_DGT(img):

    res = np.zeros(img.shape)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            res[i][j] = (2*img[i][j] - img[i-1][j] - img[i][j-1])/np.sqrt( (img[i][j] - img[i-1][j])**2 + (img[i][j] - img[i][j-1])**2 + epsilon) \
                + (img[i][j] - img[i+1][j])/np.sqrt( (img[i+1][j] - img[i][j])**2 + (img[i+1][j] - img[i+1][j-1])**2 + epsilon) \
                + (img[i][j] - img[i][j+1])/np.sqrt( (img[i][j+1] - img[i-1][j+1])**2 + (img[i][j+1] - img[i][j])**2 + epsilon)
            
    return res

def gradient_cost_l1(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + lam * gradient_l1(m)

def cost_l1(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam * np.sum(np.abs(m))


def gradient_cost_tik(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + 2*lam*m

def cost_tik(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam * np.linalg.norm(m)**2

def cost_TV(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam * l1_DGT(m)

def gradient_cost_TV(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + lam*grad_l1_DGT(m)

def cost_combination(m, y, lam, alpha=0.04):
    """
        Adds both the TV regularization and the L1 regularization, through the extra coefficient alpha
    """
    return 0.5*np.linalg.norm(undersample_fourier_adjoint(undersample_fourier(m) - y)) + lam * l1_DGT(m) + alpha * np.sum(np.abs(m))

def gradient_cost_combination(m, y, lam, alpha=0.04):
    """
        See cost_combination
    """
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + lam*grad_l1_DGT(m) + alpha * gradient_l1(m)

if __name__ == "__main__":
    # Testing of functions

    pass