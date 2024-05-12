import src.loaders.core as core
import src.segnet_bm.segnet as segnet
import src.segnet_bm.metrics as metrics
import src.loaders.fetch_Oxford_IIIT_Pets as OxfordPets
import torch
import torch.nn as nn
import torchvision

USE_TORCH_METRICS = True
if USE_TORCH_METRICS:
    import torchmetrics as TM


def test_performance(model_dict,test_loader):

    model = model_dict['model']

    print("\nAnalysisng test performance for :",model_dict['name'])
    core.print_model_parameters(model)

    core.to_device(model)
    model.eval()

    if USE_TORCH_METRICS:

        iou = core.to_device(TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=core.TrimapClasses.BACKGROUND))
        pixel_metric = core.to_device(TM.classification.MulticlassAccuracy(3, average='micro'))

        iou_accuracies = []
        pixel_accuracies = []
        custom_iou_accuracies = []

        with torch.inference_mode():

            for batch_idx, (inputs, targets) in enumerate(test_loader, 0):

                inputs  = core.to_device(inputs)
                targets = core.to_device(targets)
                predictions = model(inputs)

                pred_probabilities = nn.Softmax(dim=1)(predictions)
                pred_labels = predictions.argmax(dim=1)

                # Add a value 1 dimension at dim=1
                pred_labels = pred_labels.unsqueeze(1)
                # print("pred_labels.shape: {}".format(pred_labels.shape))
                pred_mask = pred_labels.to(torch.float)

                iou_accuracy = iou(pred_mask, targets)
                # pixel_accuracy = pixel_metric(pred_mask, targets)
                pixel_accuracy = pixel_metric(pred_labels, targets)
                custom_iou = metrics.IoUMetric(pred_probabilities, targets)
                iou_accuracies.append(iou_accuracy.item())
                pixel_accuracies.append(pixel_accuracy.item())
                custom_iou_accuracies.append(custom_iou.item())

                del inputs
                del targets
                del predictions
            # end for

            iou_tensor = torch.FloatTensor(iou_accuracies)
            pixel_tensor = torch.FloatTensor(pixel_accuracies)
            custom_iou_tensor = torch.FloatTensor(custom_iou_accuracies)

            print("Test Dataset Accuracy:")
            print(f"Pixel Accuracy: {pixel_tensor.mean():.4f}, IoU Accuracy: {iou_tensor.mean():.4f}, Custom IoU Accuracy: {custom_iou_tensor.mean():.4f}")
    else:

        custom_iou_accuracies = []

        with torch.inference_mode():

            for batch_idx, (inputs, targets) in enumerate(test_loader, 0):

                inputs = core.to_device(inputs)
                targets = core.to_device(targets)
                predictions = model(inputs)

                pred_probabilities = nn.Softmax(dim=1)(predictions)
                pred_labels = predictions.argmax(dim=1)

                # Add a value 1 dimension at dim=1
                pred_labels = pred_labels.unsqueeze(1)
                # print("pred_labels.shape: {}".format(pred_labels.shape))
                pred_mask = pred_labels.to(torch.float)

                custom_iou = metrics.IoUMetric(pred_probabilities, targets)
                custom_iou_accuracies.append(custom_iou.item())

                del inputs
                del targets
                del predictions

            custom_iou_tensor = torch.FloatTensor(custom_iou_accuracies)

            print("Test Dataset Accuracy")
            print(f"Custom IoU Accuracy: {custom_iou_tensor.mean():.4f}")

def show_examples(model_dict,test_loader):

    (test_inputs, test_targets) = next(iter(test_loader))

    # Inspecting input images
    input_grid = torchvision.utils.make_grid(test_inputs, nrow=8)
    core.t2img(input_grid).show()

    # Inspecting the segmentation masks corresponding to the input images
    #
    # When plotting the segmentation mask, we want to convert the tensor
    # into a float tensor with values in the range [0.0 to 1.0]. However, the
    # mask tensor has the values (0, 1, 2), so we divide by 2.0 to normalize.
    targets_grid = torchvision.utils.make_grid(test_targets / 2.0, nrow=8)
    core.t2img(targets_grid).show()


    # Get segmentation mask predicted by the model:
    model = model_dict['model']
    core.to_device(model)
    model.eval()
    predictions = model(core.to_device(test_inputs))
    # Apply softmax
    predictions = nn.Softmax(dim=1)(predictions)
    # Get label - max probability (ties get broken as first element with max value)
    predicted_labels = predictions.argmax(dim=1)
    # Add a value 1 dimension at dim=1
    predicted_labels = predicted_labels.unsqueeze(1)
    # print("pred_labels.shape: {}".format(pred_labels.shape))
    predicted_mask = predicted_labels.to(torch.float)

    predicted_mask_grid = torchvision.utils.make_grid(predicted_mask / 2.0, nrow=8)
    core.t2img(predicted_mask_grid ).show()


def test_all_segnets():
    # Create loader for test data:

   _, testset = OxfordPets.augmented()

   test_loader = torch.utils.data.DataLoader(testset,
                                             batch_size=16,
                                             shuffle=False)


   # Define the two models [this assumes you have saved models from training in the output dir]
   model_1 = {'name': "SegNet Standard Model",
            'model': segnet.ImageSegmentation(kernel_size=3),
            'checkpt': "./models/segnet/segnet_standard/segnet.pt"}

   model_2 = {'name': "SegNet depthwise separable convolution Model",
               'model': segnet.ImageSegmentationDSC(kernel_size=3),
               'checkpt': "./models/segnet/segnet_dsc/segnet.pt"}


    # Load the two models:
   core.load_model_from_checkpoint(model_1['model'],model_1['checkpt'])
   core.load_model_from_checkpoint(model_2['model'],model_2['checkpt'])

   with torch.inference_mode():

        # Accuracy of the model 1
        test_performance(model_1,test_loader)

        # Accuracy of the model 2
        test_performance(model_2,test_loader)

        # Dislplay some examples of predicitons + ground truth:
        show_examples(model_1,test_loader)

        show_examples(model_2,test_loader)


if __name__ == '__main__':
    test_all_segnets()
   