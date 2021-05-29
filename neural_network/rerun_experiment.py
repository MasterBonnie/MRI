import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from skimage import io, data, metrics

# This is very ugly im sorry 
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, r"C:\Users\daan\Desktop\TUe\Master year 1\Master Math\Inverse Problems and Imaging\report\mri_regularization")

import update_method, cost

from torch.utils.tensorboard import SummaryWriter

import training
from dataset import get_dataset, get_dataset_full_image
from neural_network_MRI import MRIConvolutionalNetwork

""" File for reusing previous models for testing """
device = "cuda"

model_path = "runs\MRI_report_testing_5fold3\model.pth"
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

indices = [30, 40, 100, 110, 30+160, 65 + 160, 100 + 160, 115+160, 45 + 320, 80 + 320, 100 + 320, 120 + 320, 50 + 480, 60 + 480, 100 + 480, 120 + 480]
n = len(indices)

full_data = get_dataset_full_image(20)
# masked, raw = next(iter(full_data))
masked = torch.zeros(n,1,256,256)
raw = torch.zeros(n,1,256,256)


for i in range(n):
    masked_i, raw_i = full_data[indices[i]]
    masked[i,0] = masked_i
    raw[i,0] = raw_i

reconstruction = model.reconstruct_full_image(raw, masked, writer, device)

# --------------------------------------------------------------------------
# 1 == regularized
# 2 == initialization
save_path = r"C:\Users\daan\Desktop\IPI_images\5nn_grad2"
save = lambda n : os.path.join(save_path, f"image{n}.png")

images_full = np.zeros((n, 256, 256))
images_masked = np.zeros((n, 256, 256))
images_restored = np.zeros((n, 256, 256))
metric = np.zeros((n, 3))
parameters = np.zeros(4)

save_run = False

learning_rate = 0.5e-5
max_iter = 6000
# TV
lam = 0.004
# NN regularizer term
lam_2 = 8

parameters[0] = learning_rate
parameters[1] = max_iter
parameters[2] = lam
parameters[3] = lam_2

# Perform the reconstruction
for i in range(n):
    nn_reconstruction = reconstruction[i, 0].numpy()
    full_data = raw[i, 0].numpy()
    full_data = full_data/np.max(np.abs(full_data))
    images_full[i] = full_data
    noisy_data = masked[i,0].numpy()
    noisy_data = noisy_data/np.max(np.abs(noisy_data))
    images_masked[i] = noisy_data
    save_image = save(i)

    fourier_data = cost.undersample_fourier(full_data)    

    # inverted_image = update_method.gradient_descent_nn(fourier_data, nn_reconstruction, lam, 0, lam_2, max_iter, learning_rate, noisy_data)
    # inverted_image = update_method.gradient_descent(fourier_data, lam, 0, max_iter, learning_rate, nn_reconstruction)

    inverted_image = update_method.conjugate_gradient_descent_nn(fourier_data, nn_reconstruction, lam, lam_2, max_iter, noisy_data)
    # inverted_image = update_method.conjugate_gradient_descent(fourier_data, lam, max_iter, nn_reconstruction)

    if save_run: plt.imsave(save_image, inverted_image, cmap="gray")

    metric[i, 0] = metrics.mean_squared_error(full_data, inverted_image)
    metric[i, 1] = metrics.peak_signal_noise_ratio(full_data, inverted_image)
    metric[i, 2] = metrics.structural_similarity(full_data, inverted_image)

    images_restored[i] = inverted_image

if save_run: 
    np.savetxt(os.path.join(save_path, "metrics.csv"), metric, delimiter=",")
    np.savetxt(os.path.join(save_path, "parameters.csv"), parameters, delimiter=",")

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