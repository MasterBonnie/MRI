import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.util import random_noise
from skimage.transform import resize

# parameters
sigma = 0.1
alpha = 1
dt = 1e-6
niter = 2001
n = 256 

def forward_diff(u, dir):
    
    if dir == 2: 
        result = (u[2:,1:-1] - u[1:-1,1:-1])/(1/n)
    elif dir == 1:
        result = (u[1:-1,2:] - u[1:-1,1:-1])/(1/n)

    return result

def backward_diff(u, dir):

    if dir == 2: 
        result = (u[1:-1,1:-1] - u[:-2,1:-1])/(1/n)
    elif dir == 1:
        result = (u[1:-1,1:-1] - u[1:-1,:-2])/(1/n)

    return result

def minmod(a,b):
    return ((np.sign(a) + np.sign(b))/2) * np.minimum(np.abs(a), np.abs(b))


def filter(img):
    u = np.ones((n,n))

    for k in range(niter - 1):
        forward_diff_x = forward_diff(u, 1)
        forward_diff_y = forward_diff(u, 2)

        backward_diff_x = backward_diff(u, 1)
        backward_diff_y = backward_diff(u, 2)

        m_x = minmod(forward_diff_y, backward_diff_y)
        m_y = minmod(forward_diff_x, backward_diff_x)


        denom_x = np.sqrt(forward_diff_x**2 + m_x**2)
        denom_y = np.sqrt(forward_diff_y**2 + m_y**2)

        x_term = np.divide(forward_diff_x, denom_x, out=np.zeros_like(forward_diff_x), where= denom_x!=0)
        y_term = np.divide(forward_diff_y, denom_y, out=np.zeros_like(forward_diff_y), where= denom_y!=0)

        # NOTE: this np.pad here is probably not correct!
        x_term = backward_diff(np.pad(x_term,1), 1)
        y_term = backward_diff(np.pad(y_term,1), 2)

        u = u + dt*alpha*(np.pad(x_term, 1) + np.pad(y_term, 1)) + dt*(img - u)
        u = np.pad(u[1:-1,1:-1], 1, mode="edge")

    return u

# noisy image
f = resize(data.camera(),(n,n))
f_delta = random_noise(f,var=sigma**2)

# solve evolution equation
f_f = filter(f_delta)

# plot
fig,ax = plt.subplots(1,2)

ax[0].imshow(f_delta)
ax[0].set_title('Noisy image')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].imshow(f_f)
ax[1].set_title('Result')
ax[1].set_xticks([])
ax[1].set_yticks([])

plt.show()
