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
def l1_norm(img):
    """
        Implementation of the l1 norm of an image
    """
    result = 0

    # Here we assume that the discretized partial derivative is 0 at the boundary
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result += np.abs(img[i][j])

    return result

# @njit
def gradient_l1_norm(img):
    """
        Implementation of the gradient of the l1 norm
    """
    result = np.zeros((256,256), dtype=np.cdouble)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i][i] = img[i][j] / np.sqrt(np.real(np.vdot(img[i][j], img[i][j])) + epsilon)

    return result


@njit
def tv_norm(img):
    """
        Implementation of the Total Variation (TV) norm of an image
    """
    result = 0

    # Here we assume that the discretized partial derivative is 0 at the boundary
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[0]):
            result += np.sqrt((img[i][j] - img[i-1][j])**2 + (img[i][j] - img[i][j-1])**2)

    return result

@njit
def gradient_tv_norm(img):
    """
        Implementation of the gradient of the TV norm
    """
    result = np.zeros((256,256), dtype=np.cdouble)

    # NOTE: we are not sure about what this is supposed to be...
    # Here we assume that the discretized partial derivatives are 0 at the boundary
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            result[i][j] = (img[i][j] - img[i-1][j])/(np.sqrt((img[i][j] - img[i-1][j])**2 + (img[i-1][j+1] - img[i-1][j])**2 + epsilon)) + \
                      (img[i][j] - img[i][j-1])/(np.sqrt((img[i+1][j-1] - img[i][j-1])**2 + (img[i][j] - img[i][j-1])**2 + epsilon)) - \
                      (img[i+1][j] - img[i][j+1] - 2*img[i][j])/(np.sqrt((img[i+1][j] - img[i][j])**2 + (img[i][j+1] - img[i][j])**2 + epsilon))

    return result


def cost(x, y, lam):
    """
        Cost function for the regularized least squares problem

        PARAMS:
            x:  current guess for the minimum
            y:  Undersampled k-space data    
            lam: regularization weight
    """
    return 0.5*np.linalg.norm(undersample_fourier(x) - y)**2 + lam * tv_norm(x)

def gradient_cost(x, y, lam):
    """
        Gradient of the cost function above

        PARAMS:
            x:  current guess for the minimum
            y:  Undersampled k-space data    
            lam: regularization weight
    """
    return undersample_fourier_adjoint(undersample_fourier(x) - y) + lam*gradient_tv_norm(x)
 
def dot_matrix(a,b):
    """
        "Dot product" of two matrices

        Same as stacking, taking the conjugate of first argument, and then taking inner product
        Only returns the real part 
    """
    m = np.conjugate(a)*b
    return np.real(np.sum(m))

def con_grad_desc(y, lam, max_iter = 50, alpha = 0.05, beta = 0.6):
    """
        Performs nonlinear conjugate gradient descent on a TV regularised least squares problem, based on
        an inverse problem in MRI
    """

    # Initial values 
    m = np.zeros((256,256))
    g = gradient_cost(m, y, lam)
    d = -g

    for k in range(max_iter):
        print(f"Iteration {k}    ", end="\r")

        # line search
        t = 1
        while cost(m + t*d, y, lam) > cost(m, y, lam) + alpha * t * dot_matrix(g,d):
            t = beta*t

        # Update our guess
        m = m + t*d
        
        # Update the search parameters
        g_new = gradient_cost(m, y, lam)    

        search_dir = (np.linalg.norm(g_new)/np.linalg.norm(g))**2

        d = -g_new + search_dir*d
        g = g_new

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

    max_iter = 20
    lam = 0.01  

    alpha = 0.05
    beta = 0.6

    reg_inverse_img = np.abs(con_grad_desc(y, lam, max_iter, alpha, beta))


    fig,ax = plt.subplots(1,3)

    ax[0].imshow(img)
    ax[0].set_title('original image')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(inverse_img)
    ax[1].set_title('Masked inverted image')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(reg_inverse_img)
    ax[2].set_title('TV regularized image')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    plt.show()


