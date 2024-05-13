from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.loaders.fetch_Oxford_IIIT_Pets import download_annotations,download_images
import src.utils.core as core
import os
import torchvision

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


def original(training_set_proportion = None):
    """
    Fetch Oxford pets training and testing data original data (no transforms applied)
    :return: trainset,testset (torchvision datasets)
    """

    download_images()
    download_annotations()

    # Fetch Oxford IIIT Pets Segmentation dataset using torchvision:

    trainset = OxfordIIITPetsAugmented(root = os.getcwd(),
                                       split="trainval",
                                       target_types="segmentation",
                                       download=True,)

    if training_set_proportion is not None:
        if (training_set_proportion <= 1) and (training_set_proportion > 0):
            training_set_size = int(len(trainset) * training_set_proportion)
            trainset,_ = torch.utils.data.random_split(trainset, [training_set_size,len(trainset) - training_set_size])

    testset = torchvision.datasets.OxfordIIITPet(root=os.getcwd(),
                                                 split="test",
                                                 target_types="segmentation",
                                                 download=True)

    return trainset,testset


def augmented(training_set_proportion = None):
    """
    Fetch Oxford pets training and testing data and return augmented versions.
    :return: trainset,testset (torchvision datasets)
    """

    download_images()
    download_annotations()

    # Fetch Oxford IIIT Pets Segmentation dataset using torchvision:
    root = os.getcwd()

    trainset = OxfordIIITPetsAugmented(root = root,
                                       split="trainval",
                                       target_types="segmentation",
                                       download=True,
                                        **core.transform_dict)

    if training_set_proportion is not None:
        if (training_set_proportion <= 1) and (training_set_proportion > 0):
            training_set_size = int(len(trainset) * training_set_proportion)
            trainset,_ = torch.utils.data.random_split(trainset, [training_set_size,len(trainset) - training_set_size])


    testset = OxfordIIITPetsAugmented(root=root,
                                     split="test",
                                     target_types="segmentation",
                                     download=True,
                                     **core.transform_dict)

    return trainset,testset


class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    """
    Source : https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/oxford-iiit-pets-segmentation-using-pytorch-segnet-and-depth-wise-separable-convs.ipynb
    This class creates a dataset wrapper that allows for custom image augmentations on both the target and label
     (segmentation mask) images.
    These custom image augmentations are needed since we want to perform transforms such as:
     1. Random horizontal flip
     2. Image resize
    and these operations need to be applied consistently to both the input image and the segmentation mask.
    """
    def __init__(
            self,
            root: str,
            split: str,
            target_types="segmentation",
            download=False,
            pre_transform=None,
            post_transform=None,
            pre_target_transform=None,
            post_target_transform=None,
            common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)

        # Common transforms are performed on both the input and the labels
        # by creating a 4 channel image and running the transform on both.
        # Then the segmentation mask (4th channel) is separated out.
        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        # end if

        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)