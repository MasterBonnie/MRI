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

    running_total_loss = 0.0
    
    size = len(train_data.dataset)
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

        if batch % loss_freq == loss_freq-1:
            
            # We either save the loss to a tensorboard, or print it in the console
            if save_loss:
                writer.add_scalar('Total loss (training)', running_total_loss / loss_freq,
                                epoch * len(train_data) + batch)

            else: 
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            running_total_loss = 0
    
    
    # Validation
    #------------------------------------------------
    model.eval()

    size = len(validation_data.dataset)
    test_loss = 0

    with torch.no_grad():
        for i, (X, Y) in enumerate(validation_data):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = model.loss_function(pred, Y)

            test_loss += loss.item()

            if i == 0 and generate_image and epoch == 10:
                n = min(X.size(0), 8)

                img_grid_1 = torchvision.utils.make_grid(X[:n].cpu(), normalize=True)
                img_grid_2 = torchvision.utils.make_grid(pred.view(X.size(0), 1, 256, 256)[:n].cpu(), normalize=True)
                img_grid_3 = torchvision.utils.make_grid(Y.view(X.size(0), 1, 256, 256)[:n].cpu(), normalize=True)

                writer.add_image(f'Ground truth images (epoch {epoch})', img_grid_3)
                writer.add_image(f'Noisy images (epoch {epoch})', img_grid_1)
                writer.add_image(f'Reconstructed image (epoch {epoch})', img_grid_2)

        test_loss /= size

        if save_loss:
            writer.add_scalar('Total loss (validation)', test_loss,
                    epoch)

        else:
            print(f"Avg loss: {test_loss:>8f} \n")

