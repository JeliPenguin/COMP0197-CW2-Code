import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from fetch_dataset import download_annotations,download_images,OxfordIIITPetsAugmented
import core
import os

mean = torch.Tensor([0.4810, 0.4490, 0.3959]) 
std = torch.Tensor([0.2627, 0.2579, 0.2660])

def custom_augmented(img_size):
    download_images()
    download_annotations()

    # Fetch Oxford IIIT Pets Segmentation dataset using torchvision:
    root = os.getcwd()

    def_transform = core.transform_dict.copy()
    def_transform["post_transform"] = transforms.Compose([
        transforms.Normalize(mean=mean, std=std)
    ])
    
    def_transform["common_transform"] = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    trainset = OxfordIIITPetsAugmented(root = root,
                                       split="trainval",
                                       target_types="segmentation",
                                       download=True,
                                       **def_transform)

    testset = OxfordIIITPetsAugmented(root=root,
                                     split="test",
                                     target_types="segmentation",
                                     download=True,
                                     **def_transform)

    return trainset,testset
