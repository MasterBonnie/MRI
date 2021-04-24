import numpy as np
import matplotlib.pyplot as plt

# parameters
sigma = 0.1
alpha = 1
dt = 1e-6
niter = 100
n = 256
coeff = lambda s : 1 + 0*s

# diffusion operator
def L(u,coeff = lambda s : 1 + 0*s):
    ue = np.pad(u,1,mode='edge') # padd edges to get array of size n+2 x n+2

    # diffusion coefficient (central differences)
    grad_norm = ((ue[2:,1:-1] - ue[:-2,1:-1])/(2/n))**2 + ((ue[1:-1,2:] - ue[1:-1,:-2])/(2/n))**2
    c = np.pad(coeff(grad_norm),1,mode='edge')

    # diffusion term (combination of forward and backward differences)
    uxx = ((c[1:-1,1:-1] + c[2:,1:-1])*(ue[2:,1:-1]-ue[1:-1,1:-1]) - (c[:-2,1:-1]+c[1:-1,1:-1])*(ue[1:-1,1:-1]-ue[:-2,1:-1]))/(2/n**2)
    uyy = ((c[1:-1,1:-1] + c[1:-1,2:])*(ue[1:-1,2:]-ue[1:-1,1:-1]) - (c[1:-1,:-2]+c[1:-1,1:-1])*(ue[1:-1,1:-1]-ue[1:-1,:-2,]))/(2/n**2)

    return uxx + uyy

def filter(img):
    # solve evolution equation
    u = np.zeros((n,n))

    for k in range(niter-1):
        u = u - dt*(u - alpha*L(u,coeff)) + dt*img

    return u
