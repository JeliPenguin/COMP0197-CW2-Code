"""
Src:  https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/oxford-iiit-pets-segmentation-using-pytorch-segnet-and-depth-wise-separable-convs.ipynb
A set of utility functions for the project
"""

import torch
import os
import torchvision.transforms as T

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

def custom_threshold(mask):
    mask = torch.where(mask > 0.005, torch.tensor(0.0), torch.tensor(1.0))
    return mask

def args_to_dict(**kwargs):
    return kwargs

transform_dict = args_to_dict(
    pre_transform=T.ToTensor(),
    pre_target_transform=T.ToTensor(),
    common_transform=T.Compose([
        T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
    ]),
    post_target_transform=T.Compose([
        T.Lambda(custom_threshold),
    ]),
)

from enum import IntEnum
class TrimapClasses(IntEnum):
    PET = 1
    BACKGROUND = 0
