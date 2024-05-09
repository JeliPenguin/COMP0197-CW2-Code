from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from fetch_Oxford_IIIT_Pets import download_annotations,download_images,OxfordIIITPetsAugmented
import core
import os


mean = torch.Tensor([0.485, 0.456, 0.406])
        
std = torch.Tensor([0.229, 0.224, 0.225])

def gen_transform(img_size):

    transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
    return transform


def custom_augmented(img_size):
    download_images()
    download_annotations()

    # Fetch Oxford IIIT Pets Segmentation dataset using torchvision:
    root = os.getcwd()

    def_transform = core.transform_dict.copy()
    def_transform["common_transform"] = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.RandomHorizontalFlip(p=0.5)
            # transforms.Normalize(mean=mean, std=std)
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



def get_hugging_face_loader_OxfordPets(img_size):

        source = 'timm/oxford-iiit-pet'

        # Ensure the dataset is properly loaded with streaming set to True
        # train_dataset = load_dataset("imagenet-1k", split="train", streaming=True,trust_remote_code=True)
        train_dataset = load_dataset(source, split="train", streaming=True,trust_remote_code=True)

        # test_dataset = load_dataset("imagenet-1k", split="test", streaming=True,trust_remote_code=True)
        test_dataset = load_dataset(source, split="test", streaming=True,trust_remote_code=True)

        transform = gen_transform(img_size)

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
            return torch.stack(images), torch.stack(labels) , image_ids

        # Setup DataLoader with the custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            collate_fn=lambda batch: collate_fn(batch)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=lambda batch: collate_fn(batch)
        )

        return train_loader, test_loader, mean, std