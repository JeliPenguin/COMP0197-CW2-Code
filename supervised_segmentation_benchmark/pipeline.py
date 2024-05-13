"""
This script executes supervised training of the SegNet model and then evaluates test set performance
"""
import os
import torch
import fetch_Oxford_IIIT_Pets as OxfordPets
import train
import test
import segnet
import core



# Use a GPU device where possible:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

_,testset = OxfordPets.augmented()

test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=16,
                                          shuffle=False)

# A single test set batch for validation - using training set data is too costly given the dataset is small
(test_inputs, test_targets) = next(iter(test_loader))

# The code below trains the model. The following measures are displayed:
#
#   1. Training set Loss (Cross-entropy)
#   2. Test accuracy metrics for a single batch (16 images) of test images. The following  metrics are computed:
#       2.1. Pixel Accuracy
#       2.2. IoU Accuracy (weighted)
#       2.3. Custom IoU Accuracy


# Run standard segnet model:
model = segnet.ImageSegmentation(kernel_size=3)

model.to(device)

# Optimizer and Scheduler:
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)

save_path = os.path.join("output") # destination for saving model checkpoints

# Train the models:
epoch_count = 50

train_set_size = [400]

for S in train_set_size:

    save_path = os.path.join("output","train_set_size_" + str(S))  # destination for saving model checkpoints

    # Create training set loader:
    trainset, _ = OxfordPets.augmented(training_set_size = S)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=16,
                                               shuffle=True)

    train.train_loop(model, train_loader, (test_inputs, test_targets), (0,epoch_count), optimizer, scheduler, save_path)


    model = {'name': "SegNet Standard Model " + "(trained with %d datpoints" % S,
             'model': segnet.ImageSegmentation(kernel_size=3),
             'checkpt': save_path + "/segnet.pt"}

    core.load_model_from_checkpoint(model['model'], model['checkpt'])

    with torch.inference_mode():
        # Accuracy of the model
        test.test_performance(model, test_loader)

        #test.show_examples(model, test_loader)


