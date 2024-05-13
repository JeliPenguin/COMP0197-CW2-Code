
import torch
import torchvision
import requests
import os
import tarfile
import src.oeq_mae_decoder_conv_layer.core as core


def download_images():
    """
    This function downloads teh Oxford IIT Pets images dataset to  ../data if it does
    not find the tar.gz file there already. This circumvents SSL errors with
    torchvision downloader
    """

    # URL for the Images.tar.gz file
    url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

    # Local path where you want to store the dataset
    path = "./oxford-iiit-pet"

    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Full path to the downloaded file
    full_path = os.path.join(path, "images.tar.gz")

    # Proceed to fetch Images tar if it does not exist in ./data already:
    if not os.path.isfile(full_path):

        print("Images tar does not exist at expected location : [%s] => downloading" % full_path)

        try:
            # Download the file
            response = requests.get(url, stream=True, verify=True)
            with open(full_path, 'wb') as f:
                f.write(response.content)

            # Extract the downloaded file
            with tarfile.open(full_path, 'r:gz') as tar:
                tar.extractall(path=path)
        except RuntimeError as error:
            print("Failed to download Oxford pets data [images] from [%s]" % url)
            print(error)


def download_annotations():
    """
    This function downloads teh Oxford IIT Pets dataset annotations file to  ../data if it does
    not find the tar.gz file there already. Its purpose is to circumvent SSL error with torchvision loader
    """

    # URL for the Images.tar.gz file
    url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    # Local path where you want to store the dataset
    path = "./oxford-iiit-pet"

    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Full path to the downloaded file
    full_path = os.path.join(path, "annotations.tar.gz")

    # Proceed to fetch Images tar if it does not exist in ./data already:
    if not os.path.isfile(full_path):

        print("Annotations tar does not exist at expected location : [%s] => downloading" % full_path)

        try:
            # Download the file
            response = requests.get(url, stream=True, verify=True)
            with open(full_path, 'wb') as f:
                f.write(response.content)

            # Extract the downloaded file
            with tarfile.open(full_path, 'r:gz') as tar:
                tar.extractall(path=path)
        except RuntimeError as error:
            print("Failed to download Oxford pets data [annotations] from [%s]" % url)
            print(error)

def original():
    """
    Fetch Oxford pets training and testing data original data (no transforms applied)
    :return: trainset,testset (torchvision datasets)
    """

    download_images()
    download_annotations()

    # Fetch Oxford IIIT Pets Segmentation dataset using torchvision:

    trainset = torchvision.datasets.OxfordIIITPet(root=os.getcwd(),
                                                  split="trainval",
                                                  target_types="segmentation",
                                                  download=False)


    testset = torchvision.datasets.OxfordIIITPet(root=os.getcwd(),
                                                 split="test",
                                                 target_types="segmentation",
                                                 download=True)

    return trainset,testset


def augmented():
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

if __name__ == '__main__':

    trainset,testset = augmented()
    

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=64,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=21,
                                              shuffle=True)

    (train_pets_inputs, train_pets_targets) = next(iter(train_loader))
    train_pets_inputs.shape, train_pets_targets.shape

    # Inspecting input images
    pets_input_grid = torchvision.utils.make_grid(train_pets_inputs, nrow=8)
    core.t2img(pets_input_grid).show()

    # Inspecting the segmentation masks corresponding to the input images
    #
    # When plotting the segmentation mask, we want to convert the tensor
    # into a float tensor with values in the range [0.0 to 1.0]. However, the
    # mask tensor has the values (0, 1, 2), so we divide by 2.0 to normalize.
    pets_targets_grid = torchvision.utils.make_grid(train_pets_targets / 2.0, nrow=8)
    core.t2img(pets_targets_grid).show()