import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from src.finetune.fetch_dataset import download_annotations,download_images,OxfordIIITPetsAugmented
import src.finetune.core as core
import os

mean = torch.Tensor([0.4810, 0.4490, 0.3959]) 
std = torch.Tensor([0.2627, 0.2579, 0.2660])

def custom_augmented_oxford_pets(img_size,training_set_proportion = None):
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
    
    if training_set_proportion is not None:
        if (training_set_proportion <= 1) and (training_set_proportion > 0):
            training_set_size = int(len(trainset) * training_set_proportion)
            trainset,_ = torch.utils.data.random_split(trainset, [training_set_size,len(trainset) - training_set_size])

    testset = OxfordIIITPetsAugmented(root=root,
                                     split="test",
                                     target_types="segmentation",
                                     download=True,
                                     **def_transform)

    return trainset,testset
