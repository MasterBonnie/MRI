import matplotlib.pyplot as plt
import numpy as np

from numba import njit

"""
    File containing different cost functions and their gradients
"""

# Regularizing parameter for the gradient of the TV norm
epsilon = 1e-12

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

def l2_grad_norm(m):
    pass

def gradient_l2_grad_norm(m):
    pass

def gradient_cost_l1(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + lam * gradient_l1(m)

def cost_l1(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam * np.sum(np.abs(m))


def gradient_cost_tik(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + 2*lam*m

def cost_tik(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam * np.linalg.norm(m)**2


def cost_l2_grad(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam*l2_grad_norm(m)

def gradient_cost_l2_grad(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + lam*gradient_l2_grad_norm(m)


def cost_TV(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam * TV_norm(m)

def gradient_cost_TV(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + lam*gradient_TV_norm(m)


def TV_norm(m):
    temp = TV_(m)
    return np.sum(np.power(np.real(temp*np.conjugate(temp)), 0.5))

def TV_(m):
    
    res = np.zeros((m.shape[0], m.shape[1], 2))

    index_x = np.array([i for i in range(1, m.shape[0])] + [m.shape[0] - 1], dtype=np.int)
    index_y = np.array([i for i in range(1, m.shape[1])] + [m.shape[1] - 1], dtype=np.int)

    Dx = m[index_x, :] - m
    Dy = m[:, index_y] - m

    res[:,:,0] = Dx
    res[:,:,1] = Dy

    return res

def TV_adj(m):

    return adjDx(m[:,:,0]) + adjDy(m[:,:,1])

def adjDy(x):
    end = x.shape[1] - 1
    index = np.array([0] + [i for i in range(end)], dtype=np.int)

    res = x[:, index] - x
    res[:, 0] = -1*x[:,0]
    res[:, end] = x[:, end - 1]

    return res

def adjDx(x):
    end = x.shape[0] - 1
    index = np.array([0] + [i for i in range(end)], dtype=np.int)

    res = x[index, :] - x
    res[0, :] = -1*x[0, :]
    res[end, :] = x[end-1, :]

    return res

def gradient_TV_norm(m):

    Dx = TV_(m)
    g = Dx*np.power(np.real(Dx*np.conjugate(Dx)) + epsilon, -0.5)
    grad = TV_adj(g)

    return grad 



# Attempt 50 for TV
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

def TV_denoise_norm(m, y, lam):
    return 0.5*np.linalg.norm(undersample_fourier(m) - y)**2 + lam * l1_DGT(m)

def grad_TV_denoise_norm(m, y, lam):
    return undersample_fourier_adjoint(undersample_fourier(m) - y) + lam*grad_l1_DGT(m)


if __name__ == "__main__":
    # Testing of functions

    test = np.array([[1,2,3], [2,3,4], [3,4,5]])

    temp = gradient_TV_norm(test)

    print(temp)
