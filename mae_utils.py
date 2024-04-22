import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import json 
import os 
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

# todo - make this more efficient - want to calculatete the mean and std of pixel values automatically
# we do this but there must be a way to do this without creating so many loaders
def get_loaders(args):
    dataset_path = dataset_path = args.dataset
    
    # load data with basic transforms and compute mean and std of colour channels

    # image wil
    basic_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    
    dataset = datasets.ImageFolder(root=dataset_path, transform=basic_transforms)
   
    total_size = len(dataset)
    test_size = int(0.1 * total_size)  # 10% for testing
    train_size = total_size - test_size  # 90% for training
    
 

    mean, std = calculate_mean_std(dataset)


    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform= transform)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    #
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


    return train_loader, test_loader, mean, std



def collate_fn(batch,args,mean,std):
    # Set up the transformation: convert all images to 3 channels, resize, and convert to tensor

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Converts 1-channel grayscale to 3-channel grayscale
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    images, labels = [], []
    for item in batch:
        image = transform(item['image'])
        label = torch.tensor(item['label'], dtype=torch.long)
        images.append(image)
        labels.append(label)
    return torch.stack(images), torch.stack(labels)


def get_hugging_face_loaders(args):


    # Ensure the dataset is properly loaded with streaming set to True
    # train_dataset = load_dataset("imagenet-1k", split="train", streaming=True,trust_remote_code=True)
    train_dataset = load_dataset('Maysee/tiny-imagenet', split="train", streaming=True,trust_remote_code=True)

    # test_dataset = load_dataset("imagenet-1k", split="test", streaming=True,trust_remote_code=True)
    test_dataset = load_dataset("Maysee/tiny-imagenet", split="valid", streaming=True,trust_remote_code=True)


    mean = torch.Tensor([0.485, 0.456, 0.406])
    
    std = torch.Tensor([0.229, 0.224, 0.225])

    # Setup DataLoader with the custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, args,mean,std)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, args,mean,std)
    )


    return train_loader, test_loader, mean, std



def calculate_mean_std(dataset, first=3000 ):
    
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_pixels = 0

    # use only m examples to compute pixel means and stds
    m = first if len(dataset) > first else len(dataset)
    subset_indices = torch.randperm(len(dataset))[:m]  
    dataset_for_stats = Subset(dataset, subset_indices)
    # Manually iterate through the dataset
    for data, _ in dataset_for_stats:
        # Flatten the channel dimension
        data = data.view(3, -1)
        channel_sum += data.sum(dim=1)
        channel_squared_sum += (data ** 2).sum(dim=1)
        num_pixels += data.shape[1]

    mean = channel_sum / num_pixels
    std = (channel_squared_sum / num_pixels - mean ** 2) ** 0.5 # var = E[X**2] - (E[X])**2

    return mean, std




def save_config(args):
    """
    Saves the training configuration in a JSON file in a specified directory.
    """
    file_name = f'config_{args.run_id}.json'
    # Ensure the directory exists
    directory=  args.checkpoint_dir
    os.makedirs(directory, exist_ok=True)
    
    # Full path to the configuration file
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'w') as f:
        # Convert args to dictionary (if it's an argparse.Namespace)
        args_dict = vars(args)
        # we've saved  two tensors to args. serialize and unserialize in config to model
        args_dict['mean_pixels'] = args.mean_pixels.tolist()
        args_dict['std_pixels'] = args.std_pixels.tolist()

        # handle non serialiable objects specifically
        if 'device' in args_dict:
            args_dict['device'] = str(args.device)  # Convert device to string

        json.dump(args_dict, f, indent=4)
        
def load_config(filename='config.json'):
    """
    Loads the configuration from a JSON file.
    """
    with open(filename, 'r') as f:
        config = json.load(f)
        return config
    


def config_to_model(config):
    args = argparse.Namespace(**config)
    
    #unserialize
    args.mean_pixels = torch.tensor(config['mean_pixels'])
    args.std_pixels = torch.tensor(config['std_pixels'])

    model = MAE(args)  
    return model, args

def load_model(model_path, config):
    model, args = config_to_model(config)
    model.load_state_dict(torch.load(model_path))
    return model, args
