"""
Src : https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/oxford-iiit-pets-segmentation-using-pytorch-segnet-and-depth-wise-separable-convs.ipynb
Training utility code
"""
import os
import torch
import torch.nn as nn
import utils.core as core
import utils.metrics as metrics
import segnet
import utils.fetch_Oxford_IIIT_Pets as OxfordPets



def train_model(model, loader, optimizer):
    """
    This function trains the model for a single epoch
    :param model: torch.nn model
    :param loader: torch dataloader
    :param optimizer: optimizer instance
    :return:
    """
    core.to_device(model.train())
    cel = True
    if cel:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = metrics.IoULoss(softmax=True)
    # end if

    running_loss = 0.0
    running_samples = 0

    for batch_idx, (inputs, targets) in enumerate(loader, 0):
        optimizer.zero_grad()
        inputs = core.to_device(inputs)
        targets = core.to_device(targets)
        outputs = model(inputs)

        # The ground truth labels have a channel dimension (NCHW).
        # We need to remove it before passing it into
        # CrossEntropyLoss so that it has shape (NHW) and each element
        # is a value representing the class of the pixel.
        if cel:
            targets = targets.squeeze(dim=1)
        # end if
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_samples += targets.size(0)
        running_loss += loss.item()
    # end for

    print("Trained {} samples, Loss: {:.4f}".format(
        running_samples,
        running_loss / (batch_idx + 1),
    ))

# Define training loop. This will train the model for multiple epochs.
#
# epochs: A tuple containing the start epoch (inclusive) and end epoch (exclusive).
#         The model is trained for [epoch[0] .. epoch[1]) epochs.
#
def train_loop(model, loader, test_data, epochs, optimizer, scheduler, save_path):
    """
    This function manages training over multiple epochs.

    :param model:
    :param loader:
    :param test_data:
    :param epochs: A tuple containing the start epoch (inclusive) and end epoch (exclusive).
    The model is trained for [epoch[0] .. epoch[1]) epochs.
    :param optimizer:
    :param scheduler:
    :param save_path:
    :return:
    """
    test_inputs, test_targets = test_data
    epoch_i, epoch_j = epochs
    for i in range(epoch_i, epoch_j):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        train_model(model, loader, optimizer)
        with torch.inference_mode():
            # Display the plt in the final training epoch.
            pass
            #print_test_dataset_masks(model, test_inputs, test_targets, epoch=epoch, save_path=save_path, show_plot=(epoch == epoch_j-1))


        if scheduler is not None:
            scheduler.step()

        print("")


if __name__ == '__main__':

    model = segnet.ImageSegmentation(kernel_size=3)
    core.to_device(model)

    # Optimizer and Scheduler:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)

    # Create training set loader
    trainset, _ = OxfordPets.augmented()

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=4,
                                               shuffle=True)

    # Train our model for 20 epochs, and record the following:
    #
    # 1. Training Loss
    # 2. Test accuracy metrics for a single batch (21 images) of test images. The following
    #    metrics are computed:
    #   2.1. Pixel Accuracy
    #   2.2. IoU Accuracy (weighted)
    #   2.3. Custom IoU Accuracy
    #
    # We also plot the following for each of the 21 images in the validation batch:
    # 1. Input image
    # 2. Ground truth segmentation mask
    # 3. Predicted segmentation mask
    #
    # so that we can visually inspect the model's progres and determine how well the model
    # is doing qualitatively. Note that the validation metrics on the set of 21 images in
    # the validation set is displayed inline in the notebook only for the last training
    # epoch.
    #
    save_path = os.path.join("..\\data\\", "segnet_basic_training_progress_images")
    train_loop(model,train_loader, None, (1, 21), optimizer, scheduler, save_path)