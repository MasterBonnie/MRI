import matplotlib.pyplot as plt
import numpy as np

from skimage import io, data
from skimage.util import img_as_ubyte
from skimage.transform import resize

from cost import *

"""
    Update methods for different objectives defined in the cost.py file
"""

epsilon = 1e-10

def gradient_descent(y, lam, max_iter = 50, learning_rate = 1e-4, init=None):
    """
        Performs regular gradient descent on the given cost function
    """

    method = "l1"

    if method == "TV":
        gradient = grad_TV_denoise_norm
        cost = TV_denoise_norm
    elif method == "tik":
        gradient = gradient_cost_tik
        cost = cost_tik
    elif method == "l1":
        gradient = gradient_cost_l1
        cost = cost_l1
    else:
        print("ERROR, no method chosen")
        return

    optimizer = AdamOptimizer(eta = learning_rate)

    # Initial values 
    if init is not None:
        m = init
    else:
        m = np.zeros((256,256))
    
    g = gradient(m, y, lam)

    for i in range(1, max_iter+1):
        m = optimizer.update(i, m, g)
        g = gradient(m, y, lam)

        if i % 100 == 0:
            cost_val = cost(m, y, lam)
            print(f"Cost at iteration {i} is {cost_val}.")

    return m


class AdamOptimizer():
    """
        Class implementing the Adam gradient descent optimizer
        see https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
    """

    def __init__(self, eta = 0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m, self.v = np.zeros((256,256)), np.zeros((256,256))
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, input, g):

        self.m = self.beta1*self.m + (1-self.beta1)*g

        self.v = self.beta2*self.v + (1-self.beta2)*(g**2)

        ## bias correction
        m_corr = self.m/(1-self.beta1**t)
        v_corr = self.v/(1-self.beta2**t)

        ## update weights and biases
        input -= self.eta*(m_corr/(np.sqrt(v_corr)+self.epsilon))
        return input

def conjugate_gradient_descent(y, lam, max_iter=50, alpha=1e-4, beta=0.6, init=None):
    """
        Performs conjugate gradient descent on the given cost function
    """

    method = "TV"

    if method == "TV":
        gradient = grad_TV_denoise_norm
        cost = TV_denoise_norm
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
    if init is not None:
        m = init
    else:
        m = np.zeros((256,256))

    g = gradient(m, y, lam)
    d = -1*g

    for i in range(max_iter):

        t = 1
                
        while cost(m + t*d, y, lam) > cost(m, y, lam) + alpha * t * np.real(np.vdot(g,d)):
            # print(f"cost of search step {cost(m + t*d, y, lam)} and cost of rhs {cost(m, y, lam) + alpha * t * np.real(np.vdot(g,d))}")
            t = beta*t

        # print(t)
        m = m + t*d

        g_new = gradient(m, y, lam)
        gamma = (np.linalg.norm(g_new))**2/(np.sum(d*(g_new - g))+epsilon)
        d = -1*g_new + gamma*d
        g = g_new

        # TODO: Check gradient to stop earlier!
        print(f"Cost at iteration {i} is {cost(m, y, lam)}.")

    return m


if __name__  == "__main__":
    #--------------------------------------------------
    # SETUP
    #--------------------------------------------------
    
    # Difficult image
    # filename_training = "Y:\Datasets\OASIS I\disc1\OAS1_0001_MR1\PROCESSED\MPRAGE\SUBJ_111\OAS1_0001_MR1_mpr_n4_anon_sbj_111.img"    
    # img = io.imread(filename_training)
    # img = img[60]
    
    # Easy image
    img = img_as_ubyte(data.brain()[3])
    img = img/np.max(np.abs(img))

    # plt.imshow(img, cmap="gray")
    # plt.show()

    # Fourier transform
    fourier_img = np.fft.fftshift(np.fft.fft2(img))

    # Acquired underampled k-space data
    y = undersample_fourier(img)

    # Standard inverse image
    inverse_img = np.abs(undersample_fourier_adjoint(y))
    #--------------------------------------------------
    learning_rate = 1e-5

    max_iter = 40
    lam = 0.04
    
    inverse_img_grad = conjugate_gradient_descent(y, lam, max_iter, alpha=0.01, beta=0.6, init=inverse_img)
    # init = inverse_img.copy()
    # inverse_img_grad = gradient_descent(y, lam, max_iter, learning_rate, init)

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
