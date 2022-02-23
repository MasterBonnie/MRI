# Data driven solution methods
This folder contains the solution methods that incorporate some data-driven element, in particular deep learning methods. The idea of the project was to learn a Convolutional Neural Network (CNN) to denoise a cheap recontruction of undersampled MRI images, and to then use this network in combination with classical methods to solve the inverse problem. Data used in the project came from [oasis](https://www.oasis-brains.org/).

`dataset.py` contains the definition of a `Dataset` class of PyTorch, which is used for training the CNN. It furthermore contains some helper code to move and process the raw OASIS dataset into a format that is easier to actually use.

`experiment.py` contains code that runs an experiment, i.e. load the data, train a CNN with the desired architecture and perform the reconstruction on some validation data.

`neural_network_MRI.py` contains the PyTorch neural network model, i.e. the architecture of the CNN. This class also contains several functions that implement actually performing the gradient descent with initialization and added CNN regularization term.

`rerun_experiment.py` contains code to load and reuse previously trained CNNs.

`training.py` contains the training loop used to train the CNN.

`transform_data.py` contains code that is used to process the raw OASIS dataset. In order to keep the amount of parameters in the CNN reasonable for personal computers, and to create more data to train on, we "cut up" the actually images into smaller ones, which is what the actual network is trained on. It also saves the images in a more explicit format, which makes it easier to load, but costs more storage space. In order to avoid having several path variables in different files, all this code is called in `dataset.py`.