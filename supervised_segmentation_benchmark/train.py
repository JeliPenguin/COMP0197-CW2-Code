"""
Src : https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/oxford-iiit-pets-segmentation-using-pytorch-segnet-and-depth-wise-separable-convs.ipynb
Training utility code
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import core
import metrics
import segnet
import fetch_Oxford_IIIT_Pets as OxfordPets

USE_TORCH_METRICS = True
if USE_TORCH_METRICS:
    import torchmetrics as TM


def train_model(epoch,model, loader, optimizer,USE_CROSS_ENTROPY_LOSS = True):
    """
    This function trains the model for a single epoch
    :param [int] epoch number (for output)
    :param model: torch.nn model
    :param loader: torch dataloader
    :param optimizer: optimizer instance
    :return:
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    batch_report_rate = 1 # Report every [batch_report_rate] batches

    #core.to_device(model.train())
    model.to(device)
    model.train()


    if USE_CROSS_ENTROPY_LOSS:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = metrics.IoULoss(softmax=True)
    # end if

    running_loss = 0.0
    running_samples = 0
    #(inputs,targets)
    for batch_idx, sample in enumerate(loader, 0):
        optimizer.zero_grad()
        #inputs = core.to_device(inputs)
        #targets = core.to_device(targets)
        inputs, targets = sample[0].to(device), sample[1].to(device)
        outputs = model(inputs)

        # The ground truth labels have a channel dimension (NCHW).
        # We need to remove it before passing it into
        # CrossEntropyLoss so that it has shape (NHW) and each element
        # is a value representing the class of the pixel.
        if USE_CROSS_ENTROPY_LOSS:
            targets = targets.squeeze(dim=1)
        # end if

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_samples += targets.size(0)
        running_loss += loss.item()
    # end for
        if batch_idx % batch_report_rate == 0:
            print("[Epoch: {}][Batch: {}][TimePerSample (s) = {:.3f}][Trained samples: {}][Loss: {:.4f}]".format(
       epoch,
            batch_idx,
            (time.time() - start_time)/running_samples,
            running_samples,
            running_loss / (batch_idx + 1),

    ))

# Define training loop. This will train the model for multiple epochs.
#
# epochs: A tuple containing the start epoch (inclusive) and end epoch (exclusive).
#         The model is trained for [epoch[0] .. epoch[1]) epochs.
#
def train_loop(model, loader, test_data, epochs, optimizer, scheduler,output_path):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if not os.path.exists(output_path):
        # Create the directory
        os.makedirs(output_path)

    test_inputs, test_targets = test_data
    epoch_i, epoch_j = epochs
    start_time = time.time()
    for epoch in range(epoch_i, epoch_j):
        T = time.time() - start_time
        print(f"[Elapsed time:{T:0.1f}][Epoch: {epoch:02d}][Learning Rate: {optimizer.param_groups[0]['lr']}]")
        train_model(epoch,model, loader, optimizer)
        core.save_model_checkpoint(model, output_path + "/segnet.pt")
        with torch.inference_mode():
            # Test set performance report #
            #core.to_device(model.eval())
            model.to(device)
            model.eval()
            model_predictions = model(core.to_device(test_inputs))
            labels = core.to_device(test_targets)
            # print("Predictions Shape: {}".format(predictions.shape))
            predictions = nn.Softmax(dim=1)(model_predictions)

            predicted_labels = predictions.argmax(dim=1)
            # Add a value 1 dimension at dim=1
            predicted_labels = predicted_labels.unsqueeze(1)
            # Create prediction for the mask:
            predicted_mask = predicted_labels.to(torch.float)

            # Calculate performance metrics
            custom_iou = metrics.IoUMetric(predicted_labels, labels)

            if USE_TORCH_METRICS:
                iou = core.to_device(TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=core.TrimapClasses.BACKGROUND))
                iou_accuracy = iou(predicted_mask,labels)

                pixel_metric = core.to_device(TM.classification.MulticlassAccuracy(3, average='micro'))
                pixel_accuracy = pixel_metric(predicted_labels,labels)

                report = f'[Epoch: {epoch:02d}] : Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}, Custom-IoU: {custom_iou:.4f}]'
                print(report)
            else:
                report = f'[Epoch: {epoch:02d}] : Custom-IoU: {custom_iou:.4f}]'
                print(report)



        if scheduler is not None:
            scheduler.step()

        print("")


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)



    # Create training set loader
    trainset,testset = OxfordPets.augmented()

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=16,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(testset,
                                               batch_size=16,
                                               shuffle=False)

    (test_inputs, test_targets) = next(iter(test_loader))

    # The code below trains both models consecutively. The following measures are displayed:
    #
    # 1. Training Loss
    # 2. Test accuracy metrics for a single batch (16 images) of test images. The following
    #    metrics are computed:
    #   2.1. Pixel Accuracy
    #   2.2. IoU Accuracy (weighted)
    #   2.3. Custom IoU Accuracy


    # Run standard segnet model:
    model = segnet.ImageSegmentation(kernel_size=3)
    # core.to_device(model)
    model.to(device)

    # Optimizer and Scheduler:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)

    save_path = os.path.join("output", "segnet_standard")
    #train_loop(model,train_loader, (test_inputs, test_targets), (1, 21), optimizer, scheduler, save_path)


    # Run segnet + DSC:
    model = segnet.ImageSegmentationDSC(kernel_size=3)
    # core.to_device(model)
    model.to(device)

    # Optimizer and Scheduler:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)

    save_path = os.path.join("output", "segnet_dsc")
    train_loop(model, train_loader, (test_inputs, test_targets), (1,5), optimizer, scheduler, save_path)