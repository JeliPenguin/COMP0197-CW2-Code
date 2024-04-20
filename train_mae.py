# data
#import modules
import torch
import torchvision
import torchvision.transforms as transforms

import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler, SequentialSampler
import time
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import time

import os
import datetime
import torch

from mae_utils import get_hugging_face_loaders,get_loaders
from mae_arch import MAE

class Trainer:
    def __init__(self,args):
        self.args = args
        self.device = self.args.device 
        # self.train_loader, self.val_loader, self.mean_pixels, self.std_pixels  = get_loaders(self.args)
        self.train_loader, self.val_loader, self.mean_pixels, self.std_pixels  = get_hugging_face_loaders(self.args)
        # also gets means and stds for unnormalizing
        print(f'Created dataset loaders using dataset in {self.args.dataset} ')


        self.mae = MAE(self.args) 
        self.mae.to(self.device)


        self.base_lr = 1.5e-4
        self.lr = self.base_lr * (self.args.batch_size/256)
        self.optimizer =torch.optim.AdamW(self.mae.parameters(), lr=self.lr)
        
        self.checkpoint_dir = args.checkpoint_dir
        


    def train_one_epoch(self,model, dataloader, optimizer, device, print_freq):
        model.train()
        total_loss = 0
        for i, (images, _) in enumerate(dataloader):
            images = images.to(self.device)

            #forward pass through the MAE model
            reconstructed, mask_indices = model(images)

            # calculate loss (assuming your model's loss method is appropriately defined)
            loss = model.loss(images, reconstructed, mask_indices)
            total_loss += loss.item()

            # gradients and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Batch: ",i," Time: ",datetime.datetime.now(), " Loss: ",loss.item())


            if (i + 1) % print_freq == 0:
                
                val_loss = self.validate(model, self.val_loader, self.device)
                print(f' Iteration {i + 1}, Train Loss: {loss.item()} | Validation Loss:{val_loss}')

        return total_loss / (i+1)
    
    def validate(self, model, dataloader, device):
        model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                reconstructed, mask_indices = model(images)
                loss = model.loss(images, reconstructed, mask_indices)
                total_loss += loss.item()
                count += 1

        return total_loss / count
    



    def train_model(self, print_freq=100):
        model = self.mae
        num_epochs = self.args.n_epochs
        
   
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(model, self.train_loader, self.optimizer, self.device, print_freq)
            if epoch == 0:
                print(f'Training MAE: \n device{self.args.device} \n dataset {self.args.dataset}')

            val_loss = self.validate(model, self.val_loader, self.device)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

            #  model checkpoint every epoch 
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            # print(f"Checkpoint saved to {checkpoint_path}")








