import matplotlib.pyplot as plt
import numpy as np

from skimage import io, data, metrics
from skimage.util import img_as_ubyte
from skimage.transform import resize

import os

from cost import *

"""
    Update methods for different objectives defined in the cost.py file
"""

epsilon = 1e-10

def gradient_descent_nn(y_1, y_2, lam, alpha, beta, max_iter = 50, learning_rate = 1e-4, init=None):
    """
        Performs regular gradient descent on the given cost function
    """

    gradient = gradient_cost_nn
    cost = cost_nn

    optimizer = AdamOptimizer(eta = learning_rate)

    # Initial values 
    if init is not None:
        m = init
    else:
        m = np.zeros((256,256))
    
    g = gradient(m, y_1, y_2, lam, alpha, beta)

    k = 1
    cost_val_1 = cost(m, y_1, y_2, lam, alpha, beta)
    cost_val_2 = cost_val_1 + 1

    while k < max_iter and cost_val_1 <= cost_val_2:
        m = optimizer.update(k, m, g)
        g = gradient(m, y_1, y_2, lam, alpha, beta)

        if k % 100 == 0:
            print(f"{k:05d} with cost {cost_val_1:4f}", end="\r")
            cost_val_2 = cost_val_1
            cost_val_1 = cost(m, y_1, y_2, lam, alpha, beta)

        k += 1

    print()
    return m

def gradient_descent(y, lam, alpha=0.02, max_iter = 50, learning_rate = 1e-4, init=None):
    """
        Performs regular gradient descent on the given cost function
    """

    gradient = gradient_cost_combination
    cost = cost_combination

    optimizer = AdamOptimizer(eta = learning_rate)

    # Initial values 
    if init is not None:
        m = init
    else:
        m = np.zeros((256,256))
    
    g = gradient(m, y, lam, alpha)

    k = 1
    cost_val_1 = cost(m, y, lam, alpha)
    cost_val_2 = cost_val_1 + 1

    while k < max_iter and cost_val_1 <= cost_val_2:
        m = optimizer.update(k, m, g)
        g = gradient(m, y, lam)

        if k % 100 == 0:
            print(f"{k:05d} with cost {cost_val_1:4f}", end="\r")
            cost_val_2 = cost_val_1
            cost_val_1 = cost(m, y, lam, alpha)

        k += 1
            
    print()
    
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

        ## update
        input -= self.eta*(m_corr/(np.sqrt(v_corr)+self.epsilon))
        return input

def conjugate_gradient_descent(y, lam, max_iter = 50, init=None):
    """
        Performs conjugate gradient descent on the given cost function
    """
    alpha = 0.02
    beta = 0.6

    gradient = gradient_cost_combination
    cost = cost_combination

    # Initial values 
    if init is not None:
        m = init
    else:
        m = np.zeros((256,256))

    g = gradient(m, y, lam)
    d = -1*g

    cost_val_1 = cost(m, y, lam, 0)
    cost_val_2 = cost_val_1 + 1

    k=1

    while k < max_iter and np.abs(cost_val_1 - cost_val_2) > 1e-3:

        t = 1
                
        while cost(m + t*d, y, lam, 0) > cost(m, y, lam, 0) + alpha * t * np.real(np.vdot(g,d)) and t > 1e-6:
            # print(f"cost of search step {cost(m + t*d, y, lam)} and cost of rhs {cost(m, y, lam) + alpha * t * np.real(np.vdot(g,d))}")
            t = beta*t

        # print(t)
        m = m + t*d

        g_new = gradient(m, y, lam)
        gamma = (np.linalg.norm(g_new))**2/(np.sum(d*(g_new - g))+epsilon)
        d = -1*g_new + gamma*d
        g = g_new

        print(f"{k:04d} with cost {cost_val_1:4f}", end="\r")
        cost_val_2 = cost_val_1
        cost_val_1 = cost(m, y, lam, 0)

        k = k+1
    
    print()

    return m

def conjugate_gradient_descent_nn(y_1, y_2, lam, lam_2, max_iter = 50, init=None):
    """
        Performs conjugate gradient descent on the given cost function
    """
    alpha = 0.02
    beta = 0.6

    gradient = gradient_cost_nn
    cost = cost_nn

    # Initial values 
    if init is not None:
        m = init
    else:
        m = np.zeros((256,256))

    g = gradient(m, y_1, y_2, lam, 0, lam_2)
    d = -1*g

    cost_val_1 = cost(m, y_1, y_2, lam, 0, lam_2)
    cost_val_2 = cost_val_1 + 1

    k=1

    while k < max_iter and np.abs(cost_val_1 - cost_val_2) > 1e-3:

        t = 1
                
        while cost(m + t*d, y_1, y_2, lam, 0, lam_2) > cost(m, y_1, y_2, lam, 0, lam_2) + alpha * t * np.real(np.vdot(g,d)) and t > 1e-6:
            # print(f"cost of search step {cost(m + t*d, y, lam)} and cost of rhs {cost(m, y, lam) + alpha * t * np.real(np.vdot(g,d))}")
            t = beta*t

        # print(t)
        m = m + t*d

        g_new = gradient(m, y_1, y_2, lam, 0, lam_2)
        gamma = (np.linalg.norm(g_new))**2/(np.sum(d*(g_new - g))+epsilon)
        d = -1*g_new + gamma*d
        g = g_new

        print(f"{k:04d} with cost {cost_val_1:4f}", end="\r")
        cost_val_2 = cost_val_1
        cost_val_1 = cost(m, y_1, y_2, lam, 0, lam_2)

        k = k+1
    
    print()

    return m


if __name__  == "__main__":
    #--------------------------------------------------
    # SETUP
    #--------------------------------------------------

    #indices = [40, 100, 65 + 160, 100 + 160, 80 + 320, 120 + 320, 60 + 480, 100 + 480]

    indices = [30, 40, 100, 110, 30+160, 65 + 160, 100 + 160, 115+160, 45 + 320, 80 + 320, 100 + 320, 120 + 320, 50 + 480, 60 + 480, 100 + 480, 120 + 480]

    full  =  lambda n : os.path.join(r"C:\Users\daan\Desktop\datasets\MRI_5\transformed\denoise_validation\full", f"mri{n}.npy")
    masked = lambda n : os.path.join(r"C:\Users\daan\Desktop\datasets\MRI_5\transformed\denoise_validation\masked", f"mri{n}.npy")

    save_path = r"C:\Users\daan\Desktop\IPI_images\5zf"
    save = lambda n : os.path.join(save_path, f"image{n}.png")

    save_run = False

    images_full = np.zeros((len(indices), 256, 256))
    images_masked = np.zeros((len(indices), 256, 256))
    images_restored = np.zeros((len(indices), 256, 256))
    metric = np.zeros((len(indices), 3))
    parameters = np.zeros(4)

    learning_rate = 1e-5
    max_iter = 5000
    # TV
    lam = 0.0025
    # L1
    alpha = 0

    parameters[0] = learning_rate
    parameters[1] = max_iter
    parameters[2] = lam
    parameters[3] = alpha

    for i in range(len(indices)):
        index = indices[i]
        full_image = full(index)
        masked_image = masked(index)

        full_image = np.load(full_image)
        masked_image = np.load(masked_image)

        save_image = save(i)

        img = full_image/np.max(np.abs(full_image))
        images_masked[i] = masked_image/np.max(np.abs(masked_image))
        images_full[i] = img

        # Acquired underampled k-space data
        y = undersample_fourier(img)

        # Standard inverse image
        # inverse_img_grad = images_masked[i]
        #--------------------------------------------------

        init = images_masked[i].copy()
        # inverse_img_grad = gradient_descent(y, lam, alpha, max_iter, learning_rate, init)
        inverse_img_grad = conjugate_gradient_descent(y, lam, max_iter, init)

        if save_run: plt.imsave(save_image, inverse_img_grad, cmap="gray")

        metric[i, 0] = metrics.mean_squared_error(img, inverse_img_grad)
        metric[i, 1] = metrics.peak_signal_noise_ratio(img, inverse_img_grad)
        metric[i, 2] = metrics.structural_similarity(img, inverse_img_grad)

        images_restored[i] = inverse_img_grad

    if save_run: 
        np.savetxt(os.path.join(save_path, "metrics.csv"), metric, delimiter=",")
        np.savetxt(os.path.join(save_path, "parameters.csv"), metric, delimiter=",")
    
    print(metric)
    print(np.mean(metric, axis=0))
    print(np.std(metric, axis=0))

    fig,ax = plt.subplots(3,3)

    for i in range(3):

        ax[i, 0].imshow(images_full[i], cmap = "gray")
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])

        ax[i, 1].imshow(images_masked[i], cmap = "gray")
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])

        ax[i, 2].imshow(images_restored[i], cmap = "gray")
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])



    plt.show()
