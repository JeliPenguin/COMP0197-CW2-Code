import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import json 
import os 
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import argparse
from mae_arch import MAE

mean = torch.Tensor([0.485, 0.456, 0.406])
    
std = torch.Tensor([0.229, 0.224, 0.225])

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


# import matplotlib.pyplot as plt
# def collate_fn(batch,args,mean,std):
#     # Set up the transformation: convert all images to 3 channels, resize, and convert to tensor

#     transform = transforms.Compose([
#         # transforms.Grayscale(num_output_channels=3),  # Converts 1-channel grayscale to 3-channel grayscale
#         transforms.Resize((args.img_size, args.img_size)),
#         transforms.Lambda(lambda x: x.convert("RGB")),  # Convert image to RGB
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)
#     ])
#     images, labels = [], []
#     for item in batch:
#         image = transform(item['image'])
#         label = torch.tensor(item['label'], dtype=torch.long)
#         images.append(image)
#         labels.append(label)
#     return torch.stack(images), torch.stack(labels)


def transform_image(image, args, mean, std):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Lambda(lambda x: x.convert("RGB")),  # Ensure image is RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(image)

def collate_fn(batch):
    # Preallocate lists for images and labels
    images = []
    labels = []

    # Loop through each item in the batch
    for item in batch:
        # Assuming 'item['image']' is already a tensor since preprocessing is done beforehand
        images.append(item['image'])

        # Assuming 'item['label']' might need to be converted to tensor
        # Check if it's already a tensor to avoid redundant operations
        if isinstance(item['label'], torch.Tensor):
            labels.append(item['label'])
        else:
            labels.append(torch.tensor(item['label'], dtype=torch.long))

    # Stack all images and labels into tensors
    images = torch.stack(images)
    labels = torch.stack(labels)

    return images, labels

# def get_hugging_face_loaders(args):


#     # Ensure the dataset is properly loaded with streaming set to True
#     # train_dataset = load_dataset("imagenet-1k", split="train", streaming=True,trust_remote_code=True)
#     train_dataset = load_dataset('Maysee/tiny-imagenet', split="train", streaming=True,trust_remote_code=True)

#     # test_dataset = load_dataset("imagenet-1k", split="test", streaming=True,trust_remote_code=True)
#     test_dataset = load_dataset("Maysee/tiny-imagenet", split="valid", streaming=True,trust_remote_code=True)

#     # Setup DataLoader with the custom collate function
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         collate_fn=lambda batch: collate_fn(batch, args,mean,std)
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         collate_fn=lambda batch: collate_fn(batch, args,mean,std)
#     )


#     return train_loader, test_loader, mean, std


def get_hugging_face_loaders(args):
    # Load the ImageNet-1k dataset in streaming mode
    print('Loading train dataset in streaming mode.')
    train_dataset = load_dataset("imagenet-1k", split="train", streaming=True, trust_remote_code=True)

    print('Loading test dataset in streaming mode.')
    test_dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)

    pet_classes = [
        1,      # Goldfish, Carassius auratus
        12,     # House finch, linnet, Carpodacus mexicanus
        87,     # African grey, African gray, Psittacus erithacus
        88,     # Macaw
        89,     # Sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita
        90,     # Lorikeet
        91,     # Coucal
        151,    # Chihuahua
        152,    # Japanese spaniel
        153,    # Maltese dog, Maltese terrier, Maltese
        154,    # Pekinese, Pekingese, Peke
        155,    # Shih-Tzu
        156,    # Blenheim spaniel
        157,    # Papillon
        158,    # Toy terrier
        159,    # Rhodesian ridgeback
        160,    # Afghan hound, Afghan
        161,    # Basset, basset hound
        162,    # Beagle
        163,    # Bloodhound, sleuthhound
        164,    # Bluetick
        165,    # Black-and-tan coonhound
        166,    # Walker hound, Walker foxhound
        167,    # English foxhound
        168,    # Redbone
        169,    # Borzoi, Russian wolfhound
        170,    # Irish wolfhound
        171,    # Italian greyhound
        172,    # Whippet
        173,    # Ibizan hound, Ibizan Podenco
        174,    # Norwegian elkhound, elkhound
        175,    # Otterhound, otter hound
        176,    # Saluki, gazelle hound
        177,    # Scottish deerhound, deerhound
        178,    # Weimaraner
        179,    # Staffordshire bull terrier, Staffordshire bullterrier
        180,    # American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier
        181,    # Bedlington terrier
        182,    # Border terrier
        183,    # Kerry blue terrier
        184,    # Irish terrier
        185,    # Norfolk terrier
        186,    # Norwich terrier
        187,    # Yorkshire terrier
        188,    # Wire-haired fox terrier
        189,    # Lakeland terrier
        190,    # Sealyham terrier, Sealyham
        191,    # Airedale, Airedale terrier
        192,    # Cairn, cairn terrier
        193,    # Australian terrier
        194,    # Dandie Dinmont, Dandie Dinmont terrier
        195,    # Boston bull, Boston terrier
        196,    # Miniature schnauzer
        197,    # Giant schnauzer
        198,    # Standard schnauzer
        199,    # Scotch terrier, Scottish terrier, Scottie
        200,    # Tibetan terrier, chrysanthemum dog
        201,    # Silky terrier, Sydney silky
        202,    # Soft-coated wheaten terrier
        203,    # West Highland white terrier
        204,    # Lhasa, Lhasa apso
        205,    # Flat-coated retriever
        206,    # Curly-coated retriever
        207,    # Golden retriever
        208,    # Labrador retriever
        209,    # Chesapeake Bay retriever
        245,    # French bulldog
        250,    # Siberian husky
        251,    # Dalmatian, coach dog, carriage dog
        259,    # Pomeranian
        260,    # Chow, chow chow
        261,    # Keeshond
        262,    # Brabancon griffon
        263,    # Pembroke, Pembroke Welsh corgi
        264,    # Cardigan, Cardigan Welsh corgi
        265,    # Toy poodle
        266,    # Miniature poodle
        267,    # Standard poodle
        268,    # Mexican hairless
        281,    # Tabby, tabby cat
        282,    # Tiger cat
        283,    # Persian cat
        284,    # Siamese cat, Siamese
        285,    # Egyptian cat
        330,    # Rabbit, wood rabbit, cottontail, cottontail rabbit
        333,    # Hamster
        338     # Guinea pig, Cavia cobaya
    ]

    # Setup DataLoader with the custom collate function
    print('Applying transformations and filtering datasets.')
    transform_lambda = lambda x: {'image': transform_image(x['image'], args, mean, std), 'label': x['label']}
    train_dataset = train_dataset.filter(lambda x: x['label'] in pet_classes).map(transform_lambda, batched=True, batch_size=1000)
    test_dataset = test_dataset.filter(lambda x: x['label'] in pet_classes).map(transform_lambda, batched=True, batch_size=1000)

    print('Saving training dataset to disk.')
    train_dataset.save_to_disk("/train_imagenet_animals")

    print('Saving test dataset to disk.')
    test_dataset.save_to_disk("/test_imagenet_animals")

    # Load training and test datasets from disk
    train_dataset = load_dataset("/train_imagenet_animals")
    test_dataset = load_dataset("/test_imagenet_animals")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=5)

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
    config = load_config(config)
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
