import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
"""
    Class representing the dataset we use for training the neural network
"""

filename_training = "Y:\Datasets\OASIS I\disc1\OAS1_0001_MR1\PROCESSED\MPRAGE\SUBJ_111\OAS1_0001_MR1_mpr_n4_anon_sbj_111.img"
filename_testing = "Y:\Datasets\OASIS I\disc1\OAS1_0002_MR1\PROCESSED\MPRAGE\SUBJ_111\OAS1_0002_MR1_mpr_n4_anon_sbj_111.img"

raw_path = "Y:\Datasets\OASIS I\\transformed\\full"
masked_path = "Y:\Datasets\OASIS I\\transformed\masked"

val_raw_path = "Y:\Datasets\OASIS I\\transformed\\validation\\full"
val_masked_path = "Y:\Datasets\OASIS I\\transformed\\validation\\masked"

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
    training_data = DataLoader(MRIDataset_2(raw_path, masked_path, 160*32, ToTensor()), batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(MRIDataset_2(val_raw_path, val_masked_path, 160*7, ToTensor()), batch_size=batch_size, shuffle=True)

    return training_data, validation_data


if __name__ == "__main__":

    # data = get_dataset_2(10)
    # raw_image, masked_images = next(iter(data))


    # train, test = get_dataset(4)
    # dataiter = iter(test)
    # images = dataiter.next()

    # sample = images[1,0]
    # show_image(sample)
    # plt.show()

    pass
