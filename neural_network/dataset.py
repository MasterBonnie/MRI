from genericpath import exists
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
# Putting the data on your SSD is much faster, unsurprsingly
path_to_data_storage = r"C:\Users\daan\Desktop\datasets\MRI"

path_to_data_ = os.path.join(path_to_data_storage, "transformed")
path_to_data_k = os.path.join(path_to_data_storage, "kspace")

training = "training"
validation = "validation"
validation_full = "denoise_validation"

full = "full"
masked = "masked"

training_path = os.path.join(path_to_data_, training)
validation_path = os.path.join(path_to_data_, validation)
training_path_k = os.path.join(path_to_data_k, training)
validation_path_k = os.path.join(path_to_data_k, validation)
validation_full_path = os.path.join(path_to_data_, validation_full)

# Path to where the training portion of the data is stored
raw_path = os.path.join(training_path, full)
masked_path = os.path.join(training_path, masked)
raw_path_k = os.path.join(training_path_k, full)
masked_path_k = os.path.join(training_path_k, masked)

# Path to where the validation portion of the data is stored
val_raw_path = os.path.join(validation_path, full)
val_masked_path = os.path.join(validation_path, masked)
val_raw_path_k = os.path.join(validation_path_k, full)
val_masked_path_k = os.path.join(validation_path_k, masked)

# Path where the complete images used for validation are stored
val_full_raw_path = os.path.join(validation_full_path, full)
val_full_masked_path = os.path.join(validation_full_path, masked)

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
    # NOTE: You have to change the total datasize manually, dont know of a good way for this yet (103680, 38880)
    training_data = DataLoader(MRIDataset_2(raw_path, masked_path, 103680, ToTensor()), batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(MRIDataset_2(val_raw_path, val_masked_path, 38880, ToTensor()), batch_size=batch_size, shuffle=True)

    return training_data, validation_data

def get_dataset_full_image(batch_size):
    data = DataLoader(MRIDataset_2(val_full_raw_path, val_full_masked_path, 160, ToTensor()), batch_size=batch_size, shuffle=True)

    return data

def get_k_space_dataset(batch_size):
    # NOTE: You have to change the total datasize manually, dont know of a good way for this yet (103680, 38880)
    training_data = DataLoader(MRIDataset_2(raw_path_k, masked_path_k, 5120, ToTensor()), batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(MRIDataset_2(val_raw_path_k, val_masked_path_k, 1120, ToTensor()), batch_size=batch_size, shuffle=True)

    return training_data, validation_data

if __name__ == "__main__":

    # NOTE: datageneration is slow, make sure this is only run once
    # see the if name is main part below
    data_generation = False
    val_data_generation = False
    data_generation_k = True

    # Make the folders if they are not yet made
    try:
        os.makedirs(path_to_data_, exist_ok=True)
        os.makedirs(path_to_data_k, exist_ok=True)
    except OSError as error:
        pass

    try:
        os.makedirs(training_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)
        os.makedirs(training_path_k, exist_ok=True)
        os.makedirs(validation_path_k, exist_ok=True)
        os.makedirs(validation_full_path, exist_ok=True)
    except OSError as error:
        pass

    try:
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(masked_path, exist_ok=True)
        os.makedirs(val_raw_path, exist_ok=True)
        os.makedirs(val_masked_path, exist_ok=True)
        os.makedirs(raw_path_k, exist_ok=True)
        os.makedirs(masked_path_k, exist_ok=True)
        os.makedirs(val_raw_path_k, exist_ok=True)
        os.makedirs(val_masked_path_k, exist_ok=True)
        os.makedirs(val_full_raw_path, exist_ok=True)
        os.makedirs(val_full_masked_path, exist_ok=True)
    except OSError as error: 
        pass

    if data_generation:
        transform_data.process_data(path_to_data, raw_path, masked_path, val_raw_path, val_masked_path)

    if val_data_generation:
        transform_data.create_test_data(path_to_data, val_full_raw_path, val_full_masked_path)

    if data_generation_k:
        transform_data.create_k_space_data(path_to_data, raw_path_k, masked_path_k, val_raw_path_k, val_masked_path_k)