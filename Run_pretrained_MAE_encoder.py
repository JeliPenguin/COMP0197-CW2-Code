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
import time
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
import numpy as np

from run_mae import args_parser 
from mae_arch import Embedder,TransformerEncoderBlock 
#from mae_utils import collate_fn

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader



class DefaultArgs:
    def __init__(self):
        self.img_size = 64  # adjusted for Tiny ImageNet
        self.patch_size = 16 # You can experiment with smaller sizes, like 8, if desired
        self.encoder_width = 512  # Adjusted for a smaller model
        self.n_heads = 8  # Fewer heads given the reduced complexity
        self.encoder_depth = 8  # Fewer layers
        self.decoder_width = 256  # Adjusted decoder width
        self.decoder_depth = 4  # Fewer layers in decoder
        self.mlp_ratio = 4.0
        self.dropout = 0.1
        self.mask_ratio = 0.75
        self.no_cls_token_encoder = False
        self.no_cls_token_decoder = False
        self.c = 3  # Number of color channels (RGB)



class MAE_encoder_eval(nn.Module):
    def __init__(self,args=DefaultArgs):
        super().__init__()
        self.args =args 
                 
                #  img_size = args.image_size, channels= 3, num_classes =10, embed_dim=1024, patch_size=self.patch_size, num_heads=10, encoder_depth=12, mlp_ratio=4, dropout=0.1):
        self.seq_len = (self.args.img_size**2)/(self.args.patch_size**2)
        self.seq_len = int(self.seq_len)
        #todos - these aren't really necessary
        # standard for ViTs

        self.flat_image_patch_size = self.args.patch_size**2 * self.args.c

        #cls token encoder 
        self.encoder_cls = not self.args.no_cls_token_encoder 
        self.decoder_cls = not self.args.no_cls_token_decoder
        
        # encoder 
       
        self.embed_patch_encoder= Embedder(self.args.patch_size, self.seq_len,self.flat_image_patch_size,self.args.encoder_width,cls=self.encoder_cls, pos_embed_sin_cos=True)
        
        self.encoder= nn.Sequential(*[TransformerEncoderBlock(embed_dim=self.args.encoder_width,num_heads=self.args.n_heads , mlp_ratio = 4, dropout=0.1)
                                     for l in range(self.args.encoder_depth)])
        self.encoder_norm = nn.LayerNorm(self.args.encoder_width)
        

        
        #MAE stuff:
        self.unshuffle_index = 0 # will be tensor of permutations for undoing the shuffle operation done when masking
        self.keep_len = int(self.seq_len *(1-self.args.mask_ratio)) # used to select the first (1-mask ratio) of the shuffled patches
        
        #decoder
        self.decoder_depth = self.args.decoder_depth #arguments
        self.decoder_width = self.args.decoder_width# arguments
        self.enc_to_dec= nn.Linear(self.args.encoder_width, self.args.decoder_width)

        self.mask_token= nn.Parameter(torch.zeros(1, 1, self.args.decoder_width))
        self.n_mask_tokens = int(self.seq_len * self.args.mask_ratio)

        self.embed_patch_decoder = Embedder(self.args.patch_size, self.seq_len,self.args.decoder_width,self.args.decoder_width,cls=self.decoder_cls,decoder=True,pos_embed_sin_cos=True) # don't 'linearly project' patches just add positional embedding (hence in_dims = out_dims = dec_width)
        self.decoder = nn.Sequential(*[TransformerEncoderBlock(embed_dim=self.args.decoder_width,num_heads=self.args.n_heads , mlp_ratio = self.args.mlp_ratio, dropout=self.args.dropout)
                                     for l in range(self.args.decoder_depth)])
        self.decoder_norm = nn.LayerNorm(self.args.decoder_width)
        
        #decoder to image reconstruction
        self.dec_to_image_patch = nn.Linear(self.args.decoder_width,self.flat_image_patch_size, bias=True) 

        self.apply(self._init_weights)
        self.initialize_params()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def initialize_params(self):
        torch.nn.init.normal_(self.mask_token)
        
    

    def forward(self,x):
        # input should be images of shape [batch_size, C, H,W ]
        
        x = self.embed_patch_encoder(x) # converts into patches, embeds patches, append cls token , then add positional embedding

        
        #x, unshuffle_indices, mask_idxs = self.rand_shuffle_and_sample(x)

        x= self.encoder(x)
        x = self.encoder_norm(x)
        #encoder and decoder have different depths - linear transformation to handle this as in MAE paper               
        
        
        return x #, mask_idxs.to(self.args.device) # x is the reconstructed image, masks_idxs are the indices of the patches in each image
    

    def loss(self, images ,x,mask_idxs):
     #'target' is image before flattening, need to convert into patches
    # loss on the MSE between reconstructed images and originals in pixel space but only on masked_patches
    # mean loss on removed patches

    
    # images (torch.Tensor): Input tensor of shape [batch_size, 3, H, W].
    # x are the pred should be of shape: [batch_size, sequence_length, P**2*C]
        targs = self.img_to_patch(images, self.args.patch_size, normalize=True)
        
        # now we'll use torch.gather to get the masked patches efficiently
        
        targs_masked_patches =  self.get_masked_patches(targs,mask_idxs)
        x_masked_patches = self.get_masked_patches(x,mask_idxs)
        # Debugging output
        
        return self.mse_per_patch(x_masked_patches,targs_masked_patches)


    def mse_per_patch(self,preds,targs):
        mse_per_patch = nn.MSELoss(reduction= 'mean')(preds,targs)
        
        return mse_per_patch
    


    # tidy only needs, mask_idxs as input, rest can be accessed from self.args
    def create_visual_mask(self, images,mask_idxs, patch_size):
        """
        converts masked patch idxs into masks for images.
        Used for visualization only. 
        
        Args:
            images (torch.Tensor): The original batch of images, shape [batch_size, 3, H, W].
            mask_idxs (torch.Tensor): Indices of the masked patches, shape [batch_size, num_masked_patches].
            patch_size (int): The size of each  patch.

        Returns:
            torch.Tensor: A tensor of the same shape as images with masked patched set to zero.
        """
        batch_size, _, H, W = images.shape
        num_patches_h = H // self.args.patch_size
        num_patches_w = W // self.args.patch_size
        num_patches = num_patches_h * num_patches_w

        # initialize the mask for all patches as ones (nothing is masked)
        patch_mask = torch.ones((batch_size, num_patches), dtype=torch.uint8, device=self.args.device)

        # set the masked patches to zero, .scatter is a covnvenient functiob
        patch_mask.scatter_(1, mask_idxs, 0)

        # reshape to match the image layout
        patch_mask = patch_mask.view(batch_size, num_patches_h, num_patches_w)

        # mask to the original image dimensions
        full_mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)

        # make mask  broadcastable over the color channels
        full_mask = full_mask.unsqueeze(1).repeat(1, 3, 1, 1)  # Shape: [batch_size, 3, H, W]

        return full_mask
    
def get_hugging_face_loader_OxfordPets(args):

        source = 'timm/oxford-iiit-pet'

        # Ensure the dataset is properly loaded with streaming set to True
        # train_dataset = load_dataset("imagenet-1k", split="train", streaming=True,trust_remote_code=True)
        train_dataset = load_dataset(source, split="train", streaming=True,trust_remote_code=True)

        # test_dataset = load_dataset("imagenet-1k", split="test", streaming=True,trust_remote_code=True)
        test_dataset = load_dataset(source, split="test", streaming=True,trust_remote_code=True)


        mean = torch.Tensor([0.485, 0.456, 0.406])
        
        std = torch.Tensor([0.229, 0.224, 0.225])

        # Setup DataLoader with the custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            collate_fn=lambda batch: collate_fn(batch, args,mean,std)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=lambda batch: collate_fn(batch, args,mean,std)
        )

        return train_loader, test_loader, mean, std
    
def collate_fn(batch,args,mean,std):
    # Set up the transformation: convert all images to 3 channels, resize, and convert to tensor

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Converts 1-channel grayscale to 3-channel grayscale
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    images, labels , image_ids = [], [], []
    for item in batch:
        image = transform(item['image'])
        label = torch.tensor(item['label'], dtype=torch.long)
        image_id = item['image_id']
        images.append(image)
        labels.append(label)
        image_ids.append(image_id)
    return torch.stack(images), torch.stack(labels) , image_ids

    
    
    
if __name__ == "__main__":
        
        args = args_parser()    
    
        args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        print(args.device)        
        
        model = MAE_encoder_eval(args)
        model.load_state_dict(torch.load("MAE/0232734/model.pth"))
        model.eval()
        
        # Create dataloader for HF pets dataset:
        trainloader,testloader,_,_  = get_hugging_face_loader_OxfordPets(args)
        
        for i, (images, labels,image_ids) in enumerate(trainloader):
            
            print(image_ids)
            
                       
            images = images.to(args.device)

            #forward pass through the MAE encoder only model
            encoded_features = model(images)
            encoded_features = encoded_features.squeeze()
            
            #print("Encoded features shape:", encoded_features.shape)
            
           
            
            filename = f"./mae_embeddings/train/{image_ids[0]}.pt"  # Saves as a .pt file
            torch.save(encoded_features,filename)
            
        for i, (images, labels,image_ids) in enumerate(testloader):
            
            print(image_ids)
            
                       
            images = images.to(args.device)

            #forward pass through the MAE encoder only model
            encoded_features = model(images)
            encoded_features = encoded_features.squeeze()
            
            #print("Encoded features shape:", encoded_features.shape)
            
           
            
            filename = f"./mae_embeddings/test/{image_ids[0]}.pt"  # Saves as a .pt file
            torch.save(encoded_features,filename)
            
