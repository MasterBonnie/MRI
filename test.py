import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from skimage import data
from skimage.util import img_as_ubyte

import Perona_Malik_diffusion
import linear_diffusion

# # Some initial tests with images to see how the inversion works 

# Random test image
img = img_as_ubyte(data.brain())[0,:,:] 

# plt.imshow(img, "gray")
# plt.title("Original image")
# plt.show()

# Fourier transform
fourier_img = np.fft.fft2(img)
fourier_img = np.fft.fftshift(fourier_img)

#print(fourier_img)

def plot_spectrum(im_fft):
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm())
    plt.colorbar()

# plot_spectrum(fourier_img)
# plt.title('Fourier transform of the image')
# plt.show()

# Reconstruction
inverse_image = np.fft.ifft2(fourier_img)

#plt.imshow(np.abs(inverse_image), "gray") 
#plt.title("Inverse Image")
#plt.show()

# Create the mask for undersampling the data 
mask_indices = np.loadtxt("mask1.txt", dtype=np.int)

mask = np.zeros((256, 256))
mask[mask_indices[:,0], mask_indices[:,1]] = 1

# plt.imshow(mask)
# plt.show()

masked_fourier_img = mask * fourier_img


inverse_masked_image = np.abs(np.fft.ifft2(masked_fourier_img))

pm_filtered_inverse_masked_image = Perona_Malik_diffusion.filter(inverse_masked_image)
linear_filtered_inverse_masked_image = linear_diffusion.filter(inverse_masked_image)

fig,ax = plt.subplots(1,4)

ax[0].imshow(img)
ax[0].set_title('original image')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].imshow(inverse_masked_image)
ax[1].set_title('Masked inverted image')
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[2].imshow(pm_filtered_inverse_masked_image)
ax[2].set_title('PM Filtered Masked inverted image')
ax[2].set_xticks([])
ax[2].set_yticks([])

ax[3].imshow(linear_filtered_inverse_masked_image)
ax[3].set_title('Linear Filtered Masked inverted image')
ax[3].set_xticks([])
ax[3].set_yticks([])


plt.show()
