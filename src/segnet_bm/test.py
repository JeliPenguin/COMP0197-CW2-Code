import src.utils.core as core
import src.segnet_bm.segnet as segnet
from src.loaders.oxfordpets_loader import augmented
import torch
import torch.nn as nn
import torchvision
import torchmetrics as TM
import matplotlib.pyplot as plt

# Define the two models [this assumes you have saved models from training in the output dir]

dataset_proportions = [0.05,0.1,0.5,0.8,1]

def test_performance(model_dict,test_loader):

    model = model_dict['model']

    print("\nAnalysisng test performance for :",model_dict['name'])
    core.print_model_parameters(model)

    core.to_device(model)
    model.eval()

    iou = core.to_device(TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=core.TrimapClasses.BACKGROUND))
    pixel_metric = core.to_device(TM.classification.MulticlassAccuracy(3, average='micro'))

    iou_accuracies = []
    pixel_accuracies = []

    with torch.inference_mode():

        for batch_idx, (inputs, targets) in enumerate(test_loader, 0):

            inputs  = core.to_device(inputs)
            targets = core.to_device(targets)
            predictions = model(inputs)
            pred_labels = predictions.argmax(dim=1)

            # Add a value 1 dimension at dim=1
            pred_labels = pred_labels.unsqueeze(1)
            # print("pred_labels.shape: {}".format(pred_labels.shape))
            pred_mask = pred_labels.to(torch.float)

            iou_accuracy = iou(pred_mask, targets)
            # pixel_accuracy = pixel_metric(pred_mask, targets)
            pixel_accuracy = pixel_metric(pred_labels, targets)
            iou_accuracies.append(iou_accuracy.item())
            pixel_accuracies.append(pixel_accuracy.item())

            del inputs
            del targets
            del predictions
        # end for

        iou_tensor = torch.FloatTensor(iou_accuracies)
        pixel_tensor = torch.FloatTensor(pixel_accuracies)

        print("Test Dataset Accuracy:")
        print(f"Pixel Accuracy: {pixel_tensor.mean():.4f}, IoU Accuracy: {iou_tensor.mean():.4f}")

        return pixel_tensor.mean(),iou_tensor.mean()
    

def show_examples(model_dict,test_loader):

    (test_inputs, test_targets) = next(iter(test_loader))

    # Inspecting input images
    input_grid = torchvision.utils.make_grid(test_inputs, nrow=8)
    plt.imshow(core.t2img(input_grid))
    plt.show()

    # Inspecting the segmentation masks corresponding to the input images
    #
    # When plotting the segmentation mask, we want to convert the tensor
    # into a float tensor with values in the range [0.0 to 1.0]. However, the
    # mask tensor has the values (0, 1, 2), so we divide by 2.0 to normalize.
    targets_grid = torchvision.utils.make_grid(test_targets / 2.0, nrow=8)
    plt.imshow(core.t2img(targets_grid))

    plt.show()
    
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
    
    plt.imshow(core.t2img(predicted_mask_grid))

    plt.show()


def test_segnet(model):
    _, testset = augmented()
    
    test_loader = torch.utils.data.DataLoader(testset,
                                             batch_size=16,
                                             shuffle=False)

    # Load the two models:
    core.load_model_from_checkpoint(model['model'],model['checkpt'])
    
    with torch.inference_mode():
        # Dislplay some examples of predicitons + ground truth:
        show_examples(model,test_loader)

        # Accuracy of the model
        return test_performance(model,test_loader)


        


def test_all_segnets():
    # Create loader for test data:
    results = {}
    for p in dataset_proportions:
        print("Testing SegNets trained on proportion: ",p)
        segnet_model = {'name': f"SegNet Standard Model {p}",
        'model': segnet.ImageSegmentation(kernel_size=3),
        'checkpt': f"./models/segnet/segnet_standard_{p}/segnet.pt"}

        segnet_dsc_model = {'name': f"SegNet depthwise separable convolution Model {p}",
                'model': segnet.ImageSegmentationDSC(kernel_size=3),
                'checkpt': f"./models/segnet/segnet_dsc_{p}/segnet.pt"}
        
        results[p] = [test_segnet(segnet_model),test_segnet(segnet_dsc_model)]
    
    return results
   


if __name__ == '__main__':
    test_all_segnets()
   