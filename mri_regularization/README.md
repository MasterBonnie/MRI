# Classical solution methods
This folder contains the implementation of a classical solution method for solving inverse problems related to imaging, in this case Compressed Sensing MRI. This method is solving the total-variation regularized least squares problem. Two ways are implemented to solve this regularized least squares problem, gradient descent and the conjugate gradient method.

In `cost.py` several different cost functions and their gradients can be found. This includes some costs that are used in the data-driven part of this project, which have `_nn` in the name.

In `update_method.py` the implementation of gradient descent and conjugate gradient descent can be found. There are again several versions of the functions that are used in the data-driven part, labelled with `_nn`. Furtermore, at the bottom of the file some example code for running can be found.