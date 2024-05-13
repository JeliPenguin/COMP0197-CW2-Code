from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.loaders.fetch_Oxford_IIIT_Pets import download_annotations,download_images,OxfordIIITPetsAugmented
import src.utils.core as core
import os


mean = torch.Tensor([0.4810, 0.4490, 0.3959]) 
std = torch.Tensor([0.2627, 0.2579, 0.2660])

def custom_threshold(mask):
    mask = torch.where(mask > 0.005, torch.tensor(0.0), torch.tensor(1.0))
    return mask


def custom_augmented_oxford_pets(img_size):
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

    def_transform["post_target_transform"] = transforms.Compose([
        transforms.Lambda(custom_threshold),
    ]),

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



def get_hugging_face_loader_OxfordPets(img_size,batch_size):

        source = 'timm/oxford-iiit-pet'

        # Ensure the dataset is properly loaded with streaming set to True
        train_dataset = load_dataset(source, split="train", streaming=True,trust_remote_code=True)

        test_dataset = load_dataset(source, split="test", streaming=True,trust_remote_code=True)

        transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                    ])

        def collate_fn(batch):
            # Set up the transformation: convert all images to 3 channels, resize, and convert to tensor
            images, labels , image_ids = [], [], []
            for item in batch:
                image = transform(item['image'])
                label = torch.tensor(item['label'], dtype=torch.long)
                image_id = item['image_id']
                images.append(image)
                labels.append(label)
                image_ids.append(image_id)
            # return torch.stack(images), torch.stack(labels) , image_ids
            return torch.stack(images), torch.stack(labels)

        # Setup DataLoader with the custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch)
        )

        return train_loader, test_loader, mean, std