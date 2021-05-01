import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.util import img_as_ubyte

from cost import *

def gradient_descent(y, lam, max_iter = 50, learning_rate = 1e-4):

    method = "tik"

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
            lr = 1e-5

        if i == 35000:
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

    max_iter = 1000 
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
