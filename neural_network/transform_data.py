import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import re

"""
    Transforms the data given in the folder disc1 to a more managable format
"""

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

def process_data(path_to_data, path_tfd, path_tmd, path_vfd, path_vmd, percentage_val_train_split=0.2):
    """
    Params:
        path_to_data:   path to where the data is located
        path_tfd:       path where the training full data is stored
        path_tmd:       path where the training masked data is stored
        path_vfd:       path where the validation full data is stored
        path_vmd:       path where the validation masked data is stored
    """
    # Whether to create and store subimages of the actual images or not.
    # see the function below, after the if statement
    sub_image = True

    save_full_string = lambda n :  os.path.join(path_tfd, f'mri{n}.npy')
    save_mask_string = lambda n :  os.path.join(path_tmd, f"mri{n}.npy")

    val_save_full_string = lambda n :  os.path.join(path_vfd, f'mri{n}.npy')
    val_save_mask_string = lambda n :  os.path.join(path_vmd, f"mri{n}.npy")

    print("Saving data to: ")
    print(save_full_string(0))
    print(save_mask_string(0))
    print(val_save_full_string(0))
    print(val_save_mask_string(0))
    print()

    dir = path_to_data
    dirs = []
    
    # Filter out all the desired images 
    r = re.compile("disc.*MPRAGE.*SUBJ_111.*OAS1.*\.img")

    # Find the desired paths to the images
    for path, _, files in os.walk(dir):
        for file in files:
            temp = os.path.join(path, file)
            if r.search(temp):
                dirs.append(os.path.join(path, file))   

    # Determine which ones to use for training/validation
    total_length = len(dirs)
    length_val = int(total_length*percentage_val_train_split)
    length_train = total_length - length_val

    if sub_image:

        # We use less images, otherwise too much data
        length_train = 3
        length_val = 2

        # Parameters specifying the subimages
        # See also the dataset.py file
        sub_image_size = 40
        stride = 24

        # Index of the image
        index = 1

        # Training data
        for i in range(length_train):
            images = io.imread(dirs[i]).astype(np.float64)

            for j in range(1, 160 + 1):
                print(f"{i}, {j}", end="\r")

                image = images[j-1]
                masked_image = undersample_fourier_adjoint(undersample_fourier(image))

                for k in range((image.shape[0] - sub_image_size)//stride):
                    for l in range((image.shape[1] - sub_image_size)//stride):
                        
                        lower_index_x = k*stride
                        upper_index_x = k*stride + sub_image_size
                        lower_index_y = l*stride
                        upper_index_y = l*stride + sub_image_size

                        sub_image = image[lower_index_x:upper_index_x, lower_index_y:upper_index_y]
                        sub_masked_image = masked_image[lower_index_x:upper_index_x, lower_index_y:upper_index_y]

                        np.save(save_full_string(index), sub_image)
                        np.save(save_mask_string(index), sub_masked_image)
                        
                        index += 1

        print(f"Total number of training images: {index - 1}")
        index = 1

        # Test data
        for i in range(length_train, length_train + length_val):
            images = io.imread(dirs[i]).astype(np.float64)

            for j in range(1, 160 + 1):
                print(f"{i}, {j}", end="\r")
                
                image = images[j-1]
                masked_image = undersample_fourier_adjoint(undersample_fourier(image))

                for k in range((image.shape[0] - sub_image_size)//stride):
                    for l in range((image.shape[1] - sub_image_size)//stride):
                        
                        lower_index_x = k*stride
                        upper_index_x = k*stride + sub_image_size
                        lower_index_y = l*stride
                        upper_index_y = l*stride + sub_image_size

                        sub_image = image[lower_index_x:upper_index_x, lower_index_y:upper_index_y]
                        sub_masked_image = masked_image[lower_index_x:upper_index_x, lower_index_y:upper_index_y]

                        np.save(val_save_full_string(index), sub_image)
                        np.save(val_save_mask_string(index), sub_masked_image)

                        index += 1

        print(f"Total number of images: {index - 1}")

    else:
        # Training data
        for i in range(length_train):
            images = io.imread(dirs[i]).astype(np.float64)

            for j in range(1, 160 + 1):
                print(f"{i}, {j}", end="\r")
                image = images[j-1]
                masked_image = undersample_fourier_adjoint(undersample_fourier(image))

                index = i*160 + j

                np.save(save_full_string(index), image)
                np.save(save_mask_string(index), masked_image)

        # Test data
        for i in range(length_train, length_train + length_val):
            images = io.imread(dirs[i]).astype(np.float64)

            for j in range(1, 160 + 1):
                print(f"{i}, {j}", end="\r")
                image = images[j-1]
                masked_image = undersample_fourier_adjoint(undersample_fourier(image))

                index = (i - length_train)*160 + j

                np.save(val_save_full_string(index), image)
                np.save(val_save_mask_string(index), masked_image)

def create_test_data(path_to_data, path_fd, path_md):
    """
        Saves some images in the full format, not the sub-imaged version as the one above
        Not in a train-validation split, because this will only be used for validation of an already trained 
        network

        Params:
            path_to_data:   Path to the MRI data
            path_fd:        Path where the full images are stored
            path_md:        Path where the masked images are stored
    """
    save_full_string = lambda n :  os.path.join(path_fd, f'mri{n}.npy')
    save_mask_string = lambda n :  os.path.join(path_md, f"mri{n}.npy")

    dir = path_to_data
    dirs = []
    
    # Filter out all the desired images 
    r = re.compile("disc.*MPRAGE.*SUBJ_111.*OAS1.*\.img")

    for path, _, files in os.walk(dir):
        for file in files:
            temp = os.path.join(path, file)
            if r.search(temp):
                dirs.append(os.path.join(path, file)) 

    length = 1

    for i in range(length):
        images = io.imread(dirs[i]).astype(np.float64)

        for j in range(1, 160 + 1):
            print(f"{i}, {j}", end="\r")
            image = images[j-1]
            masked_image = undersample_fourier_adjoint(undersample_fourier(image))

            index = i*160 + j

            np.save(save_full_string(index), image)
            np.save(save_mask_string(index), masked_image)
