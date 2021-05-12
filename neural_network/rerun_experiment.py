import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import training
from dataset import get_dataset, get_dataset_full_image
from neural_network_MRI import MRINetwork, MRIConvolutionalNetwork

""" File for reusing previous models for testing """
device = "cuda"

model_path = "runs\MRI_test_15_08_47_12_May\model.pth"
path = os.path.join(os.getcwd(), model_path)

model = torch.load(path)
print("loaded model")

# Name of the experiment
name = "MRI_test_post_train"

# We add a timestamp to the name
date_now = datetime.now()
path = date_now.strftime("_%H_%M_%S_%d_%b")
path_1 = os.path.join(os.getcwd(), "runs")
path = os.path.join(path_1, name + path)

# And create a folder under the runs folder of that name
# for the tensorboard files
try:
    os.mkdir(path)
except OSError as error: 
    try:
        os.mkdir(path + "_1")
    except OSError as error:
        print("help")

writer = SummaryWriter(path)

# Check images after training, i.e. fully reconstruct them
# Grab random data
full_data = get_dataset_full_image(8)
masked, raw = next(iter(full_data))

print("starting reconstruction")
model.reconstruct_full_image_reg(raw, masked, writer, device)