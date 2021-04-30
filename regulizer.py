import matplotlib.pyplot as plt
import numpy as np

from numba import njit
from skimage import data
from skimage.util import img_as_ubyte

# Regularizing parameter for the gradient of the TV norm
epsilon = 1e-7

# Create the mask for undersampling the data 
mask_indices = np.loadtxt("mask1.txt", dtype=np.int)

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
            res[i][j] = img[i][j]/(np.sqrt(np.real(com_dot(m[i][j],m[i][j])) + epsilon))

    return res

@njit
def tv_norm(img):
    """
        Implementation of the Total Variation (TV) norm of an image
    """
    result = 0

    # Here we assume that the discretized partial derivative is 0 at the boundary
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[0] - 1):
            result += np.sqrt(np.abs(img[i][j] - img[i-1][j])**2 + np.abs(img[i][j] - img[i][j-1])**2)

    return result

@njit
def gradient_tv_norm(img):
    """
        Implementation of the gradient of the TV norm
    """
    result = np.zeros(img.shape)

    # NOTE: we are not sure about what this is supposed to be...
    # Here we assume that the discretized partial derivatives are 0 at the boundary
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            result[i][j] = (img[i][j] - img[i][j-1])/(np.sqrt(np.abs(img[i][j] - img[i][j-1])**2 + np.abs(img[i+1][j-1] - img[i][j-1])**2 + epsilon)) + \
                      (img[i][j] - img[i-1][j])/(np.sqrt(np.abs(img[i-1][j+1] - img[i-1][j])**2 + np.abs(img[i][j] - img[i-1][j])**2 + epsilon)) - \
                      (img[i][j+1] - img[i+1][j] - 2*img[i][j])/(np.sqrt(np.abs(img[i][j+1] - img[i][j])**2 + np.abs(img[i+1][j] - img[i][j])**2 + epsilon))

    return result

def gradient_cost_l1(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + lam * gradient_l1(m)

def cost_l1(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam * np.sum(np.abs(m))

def gradient_cost_tik(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + 2*lam*m

def cost_tik(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam * np.linalg.norm(m)**2

def cost_TV(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam * tv_norm(m)

def gradient_cost_TV(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + lam*gradient_tv_norm(m)


def gradient_descent(y, lam, max_iter = 50, learning_rate = 1e-4):

    method = "TV"

    if method == "TV":
        gradient = gradient_cost_TV
        cost = cost_TV
    elif method == "tik":
        gradient = gradient_cost_tik
        cost = cost_tik
    elif method == "l1":
        gradient = gradient_cost_l1
        cost = cost_l1
    else:
        print("ERROR, no method chosen")
        return


    # Initial values 
    m = np.zeros((256,256))
    g = gradient(m, y, lam)
    lr = learning_rate

    for i in range(1, max_iter+1):
        m += lr*g
        g = gradient(m, y, lam)

        if i % 100 == 0:
            cost_val = cost(m, y, lam)
            print(f"Cost at iteration {i} is {cost_val}.")

        if i == 13000:
            lr = 1e-6

    return m


if __name__  == "__main__":
    #--------------------------------------------------
    # SETUP
    #--------------------------------------------------
    img = img_as_ubyte(data.brain())[0,:,:] 

    # Fourier transform
    fourier_img = np.fft.fftshift(np.fft.fft2(img))

    # Acquired underampled k-space data
    y = undersample_fourier(img)

    # Standard inverse image
    inverse_img = np.abs(undersample_fourier_adjoint(y))
    #--------------------------------------------------

    max_iter = 20000 
    learning_rate = 1e-4

    lam = 0.1
    
    inverse_img_grad = gradient_descent(y, lam, max_iter, learning_rate)

    fig,ax = plt.subplots(1,3)

    ax[0].imshow(img)
    ax[0].set_title('original image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(inverse_img)
    ax[1].set_title('Masked inverted image')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(inverse_img_grad)
    ax[2].set_title('Masked inverted image (grad desc)')
    ax[2].set_xticks([])
    ax[2].set_yticks([])


    plt.show()
