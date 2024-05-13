
import requests
import os
import tarfile


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