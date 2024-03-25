# Supervised Segmentation Benchmark

The code in this folder is adpated from the ipython notebooks in this public repo:
https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/oxford-iiit-pets-segmentation-using-pytorch-segnet-and-depth-wise-separable-convs.ipynb
(Oxford IIIT Pets Segmentation using PyTorch)

It uses a so-called SegNet model (related to U-Nets - see below).

```
SegNet and U-Net are both popular architectures for semantic image segmentation, but they are not the same.

The main difference between them lies in how they handle the transfer of information from the encoding (downsampling) path to the decoding (upsampling) path:

A U-Net transfers the entire feature map from the encoding path to the decoding path. This approach uses more memory but can potentially capture more detailed information1.
SegNet only transfers the pooling indices (the locations of the maximum values) from the encoding path to the decoding path. This approach uses less memory1.
Another difference is in their depth. Seg-UNet uses five convolution blocks compared to U-SegNet, which has three convolution blocks2.
```
Note that here we use a model which was supposed to have 5 convolution blocks but since our input is 128x128
the fifth convolution would have a 4x4 output which would probably result in sever loss of information (see the 
author's comment on the network class ``ImageSegmentation`` in ``segnet.py``) - it is clear that this experiment
has already been conducted and you can see the fifth convolution is commented out.

The directory contains implementations for two related models :

1. A SegNet with ~15m params
2. A version using depthwise separable convolutions which reduces params to ~2m

I have done a short training run on these and find that I get sensible results with just 5 epochs (clearly we
should investigate training for longer and whether we can improve the training regimen though). Note that the code allows the
user to choose the loss function as either CrossEntropy loss or an IoU metric (see ``train.py``. I have only trained with CE loss so far. 
IoU is better suited to capturing what matters for segmentation tasks - see next section. 

## Metrics for segmentation tasks

Metrics like CrossEntropy and pixel accuracy are used in segmentation but the fundamental drawback that on any image
where the subject is 'small' you will get a high score for these by predicting that every pixel is background. There are 
other metrics that are more nuanced in this regard and so better suited for the purpose - one is the IoU metric (intersection
over union) and another is the related Dice score (see my literature review note for some commentary on this).

Here we are capable of reporting three metrics:
1. Pixel Accuracy
2. IoU (standard version)
3. A locally defined version of IoU 

*NOTE*: 1 and 2 are implemented by calling the relevant functions from ``torchmetrics``. **Torchmetrics is not included in the 
standard image for the project**. We could use up one of our three additional packages on this but I fear it's not worth it. Anyhow,
I've added flags for the torchmetrics import in the two files where it is used so we can switch it on and off. I'm sure we could
implement both those metrics ourselves if need be. 

My basic training run achieved pixel accuracy of 83% , IoU accuracy of 69%, and custom IoU accuracy of 55% on the test set. 
Clearly further training will improve these but not by much given the author's table in the repo. 

``metrics.py`` contains an implementation of metric number 3 above.

## Oxford Pets Data

The Pets data needs to be present in a folder called ``oxford-iit-pet``. The code that creates ``torchvision.datasets`` for training
and test data is found in ``fetch_Oxford_IIT_Pets.py``. There are two download functions in there that I wrote to get around
SSL errors that I encounter with the standard torchvision downloader - I believe this is because of the OLD versions of 
packages that we use in the standard ADL image ``cw1-pt``. Anyhow, the downloaders look for the files in the expected location
and pull them directly if they are missing. See the ``main`` section in ``fetch_Oxford_IIT_Pets.py`` for an example of code 
that creates data loaders for the training and test datasets (+ displays some images + segmentation masks).

*NOTE* : I implement two data utilities:
1. ``original()`` This function pulls in the data with NO transforms
2. ``augmented()`` This applies random horizontal image flips and color jitter as data augmentation for the training dataset 

## Model Definitions

These are all in ``segnet.py``.

## Training

Utility code for training is in ``train.py`` - if you run that module then main code will train both segnets for the
designated number of epochs in the ``epochs`` argument (a range actually) and it will save the models to an output folder
at the end of each epoch. 

## Testing

``test.py`` contains some test code with two functions:

1. ``test_performance`` : calculates all three metrics on the test set
2.  ``show_examples`` : plots the ground truth segmentation and true segmentation for a small number of examples from the test set

Again (provided you have trained models) you can run test.py to run these functions for both models.

## Transfer learning : Integration with a pre-trained model

Just a quick comment on this: There are a couple of ways we could integrate these models with a pre-trianed model to 
complete the project task. Both would involve using only the decoder component of the segnet as a segmentation head:
1. Match a pre-trained encoder with decoder from segnet model and fine-tune ALL params on Oxford Pets as we did here
2. Match a pre-trained encoder with decoder from segnet model and fine-tune ONLY segnet params on Oxford Pets as we did here

The pre-trained model could be the encoder part of a masked auto-encoder trained on ImageNet-1k for example.




