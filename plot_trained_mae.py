import torch
import torchvision
import torchvision.transforms as transforms


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler, SequentialSampler


from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader
import time
import datetime
from torch.optim.lr_scheduler import LambdaLR




from mae_arch import MAE
from mae_utils import get_loaders


import matplotlib.pyplot as plt
import numpy as np

from mae_utils import load_model

import os
import argparse
import pdb
import sys
import pickle
import logging
import random
import csv
import math
import json
import copy







parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to a model file, to be loaded into pytorch')


parser.add_argument('-d', '--dataset', type=str, help='path to the dataset of trials')
parser.add_argument('-c', '--config', type=str, help='path to config file') # must be a .json file

args = parser.parse_args()




def visualize_comparisons(loader, model):
    #compares masked original, image, autoencoder reconstruction
    # on a batch of images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model.args.device = device
    model.to(device)
    model.eval()
    with torch.no_grad():
          batch = next(iter(loader))
          images = batch[0].to(device)

          decoder_output, mask_idxs = model(images)
          reconstructions = model.reconstruct_image(decoder_output)
        
          masks = model.create_visual_mask(images, mask_idxs,model.args.patch_size)
          
          masked_images = images * masks

          #the number of examples to display (limited to a manageable number for visualization)
          batch_size = images.size(0)
          if batch_size > 4:
              print("Batch size is too large for effective visualization. Reducing to 4 for display.")
              batch_size = 4  # Adjust batch size here if needed

          fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))  # 3 columns for each type of image

          # no. of rows in plot is the no. of samples in batch
          for i in range(batch_size):
              imshow(masked_images[i], axes[i, 0],mean=model.args.mean_pixels,std=model.args.std_pixels)
              imshow(reconstructions[i], axes[i, 1],mean=model.args.mean_pixels,std=model.args.std_pixels)
              imshow(images[i], axes[i, 2],mean=model.args.mean_pixels,std=model.args.std_pixels)

          # Labeling columns
          columns = ['Masked Image', 'Reconstruction', 'Original Image']
          for ax, col in zip(axes[0], columns):
              ax.set_title(col)

          plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the spacing between images
          plt.show()
           # Show only one batch for demonstration

def imshow(img, ax, mean,std):
    # Helper function to unnormalize and show an image on a given Axes object.
    device = img.device
    mean = mean.view(3, 1, 1).to(device)
    std = std.view(3, 1, 1).to(device)
    img = img * std + mean  # Unnormalize and move to CPU
    npimg = img.cpu().numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert from Tensor image
    ax.axis('off')  # Hide axes ticks


def get_test_loader(dataset_path,args):

    # Define the transforms to resize the images and convert them to PyTorch tensors
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                std=[0.229, 0.224, 0.225])  
    ])


    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

   
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return test_loader


if __name__ == "__main__":
    model, old_args = load_model(args.model ,args.config) #old args are the args used to train model in path

    test_loader = get_test_loader(args.dataset,old_args)
    visualize_comparisons(test_loader, model)


















