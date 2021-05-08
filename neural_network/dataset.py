import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import transform_data

"""
    Class representing the dataset we use for training the neural network
"""

# Path to where the OASIS dataset is located
# If datageneration is true, this is also the location where the other dataset is created
# NOTE: In principle this is the only thing that has to be changed
path_to_data = "Y:\Datasets\OASIS I"

path_to_data_ = os.path.join(path_to_data, "transformed")

training = "training"
validation = "validation"

full = "full"
masked = "masked"

training_path = os.path.join(path_to_data_, training)
validation_path = os.path.join(path_to_data_, validation)

# Path to where the training portion of the data is stored
raw_path = os.path.join(training_path, full)
masked_path = os.path.join(training_path, masked)

# Path to where the validation portion of the data is stored
val_raw_path = os.path.join(validation_path, full)
val_masked_path = os.path.join(validation_path, masked)

def show_image(img):
    plt.imshow(img, cmap="gray")
    plt.axis("off")

class MRIDataset_2(Dataset):
    """ MRI images dataset """

    def __init__(self, raw_dir, masked_dir, length, transform=None):
        """
        Params:
            root_dir: Directory with the images
        """
        self.raw_dir = raw_dir
        self.masked_dir = masked_dir
        self.transform = transform
        self.length = length
        self.indices = pd.DataFrame([f"mri{n}.npy" for n in range(1,self.length+1)])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        raw_img_path = os.path.join(self.raw_dir, self.indices.iloc[idx, 0])
        masked_img_path = os.path.join(self.masked_dir, self.indices.iloc[idx, 0])

        raw_images = np.load(raw_img_path)
        masked_images = np.load(masked_img_path)

        if self.transform is not None:
            raw_images = self.transform(raw_images)
            masked_images = self.transform(masked_images)
        
        return masked_images, raw_images

def get_dataset(batch_size):
    # NOTE: You have to change this 32 and 7 here manually, dont really know for a good way to do this yet
    training_data = DataLoader(MRIDataset_2(raw_path, masked_path, 160*32, ToTensor()), batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(MRIDataset_2(val_raw_path, val_masked_path, 160*7, ToTensor()), batch_size=batch_size, shuffle=True)

    return training_data, validation_data

if __name__ == "__main__":

    # NOTE: datageneration is slow, make sure this is only run once
    # see the if name is main part below
    datageneration = True

    if datageneration:

        # Make the folders if they are not yet made
        try:
            os.mkdir(path_to_data_)
        except OSError as error:
            pass

        try:
            os.mkdir(training_path)
            os.mkdir(validation_path)
        except OSError as error:
            pass

        try:
            os.mkdir(raw_path)
            os.mkdir(masked_path)
            os.mkdir(val_raw_path)
            os.mkdir(val_masked_path)
        except OSError as error: 
            pass

        transform_data.process_data(path_to_data, raw_path, masked_path, val_raw_path, val_masked_path)
