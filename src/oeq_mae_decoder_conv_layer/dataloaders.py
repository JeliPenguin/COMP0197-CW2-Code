from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from src.oeq_mae_decoder_conv_layer.fetch_dataset import download_annotations,download_images,OxfordIIITPetsAugmented
import src.oeq_mae_decoder_conv_layer.core as core
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



class EncodedDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.input_files = sorted(os.listdir(os.path.join(folder_path, "input")))
        self.target_files = sorted(os.listdir(os.path.join(folder_path, "target")))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_name = self.input_files[idx]
        target_file_name = self.target_files[idx]
        input_file_path = os.path.join(self.folder_path, "input", input_file_name)
        target_file_path = os.path.join(self.folder_path, "target", target_file_name)
        input_data = torch.load(input_file_path)
        target_data = torch.load(target_file_path)
        # Process the data if needed (e.g., apply transformations)
        return input_data, target_data

def gen_embed_dataloader():
    trainDataset = EncodedDataset("./mae_encoded_batches/train")
    trainDataloader = DataLoader(trainDataset, batch_size=1)
    testDataset = EncodedDataset("./mae_encoded_batches/train")
    testDataloader = DataLoader(testDataset, batch_size=1)
    return trainDataloader, testDataloader