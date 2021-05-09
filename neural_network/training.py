import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision


def training_loop(model, train_data, validation_data, device, optimizer, writer, epoch,
                    save_loss=True, generate_image=True, loss_freq=100):
    """
        Defines a generic training loop over the data.
    """

    # Training (for one epoch)
    #------------------------------------------------
    model.train()
    print("Beginning with training...")
    running_total_loss = 0.0
    
    size = len(train_data.dataset)
    nr_batches = len(train_data)
    for batch, (X, Y)  in enumerate(train_data):
        X, Y = X.to(device), Y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = model.loss_function(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_total_loss += loss.item()

        # Print a progress bar
        printProgressBar(batch, nr_batches)

        if batch % loss_freq == loss_freq-1:
            
            # We either save the loss to a tensorboard, or print it in the console
            if save_loss:
                writer.add_scalar('Total loss (training)', running_total_loss / loss_freq,
                                epoch * len(train_data) + batch)

            else: 
                loss, current = loss.item(), batch * len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            running_total_loss = 0
    
    
    # Validation
    #------------------------------------------------
    model.eval()
    print()
    print("Beginning with evaluation...")

    size = len(validation_data.dataset)
    nr_batches = len(validation_data)

    test_loss = 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(validation_data):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = model.loss_function(pred, Y)

            test_loss += loss.item()

            printProgressBar(batch, nr_batches)

            if batch == 0 and generate_image:
                n = min(X.size(0), 8)

                img_grid_1 = torchvision.utils.make_grid(X[:n].cpu(), normalize=True)
                img_grid_2 = torchvision.utils.make_grid(pred.view(X.size(0), 1, 40, 40)[:n].cpu(), normalize=True)
                img_grid_3 = torchvision.utils.make_grid(Y.view(X.size(0), 1, 40, 40)[:n].cpu(), normalize=True)

                writer.add_image(f'Ground truth images (epoch {epoch})', img_grid_3)
                writer.add_image(f'Noisy images (epoch {epoch})', img_grid_1)
                writer.add_image(f'Reconstructed image (epoch {epoch})', img_grid_2)

        test_loss /= size

        if save_loss:
            writer.add_scalar('Total loss (validation)', test_loss,
                    epoch)

        else:
            print(f"Avg loss: {test_loss:>8f} \n")

    print()


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
        https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()