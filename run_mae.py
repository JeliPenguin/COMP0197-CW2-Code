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


# three modes:
    #pretrain just pretrains an encoder using masked auto encoder approach 
    # for image segmentation

    # finetune: specify an  --encoder path to add a decoder to an already finetuned encoder at encoder path; if none given, the model's will be pretrained with MAE for default n_pretrain epochs  and then finetuned. decoder_type - default: seg net - options [SETR, ViT]
    # supervised train a model from scratch for image segmentation


def args_parser(): 
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--mode', choices = ['pretrain','finetune','scratch'], default='pretrain')
    # masked autoencoder arguments
    ## encoder 
    # todo: get image size from image in loader
    parser.add_argument('--mask_ratio', type=float, default = 0.8,help='proportion of tokens masked')
    parser.add_argument('--img_size',type = int, default=224, help='img_size H=W resolution of images input to encoder')
    parser.add_argument('--c',type=int, default= 3, help='number of colour channels. default 3 for RGB color')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--encoder_width', type=int, default =1024,help='embedding dimension for encoder inputs')
    parser.add_argument('--encoder_depth',type=int, default= 24)

    
    parser.add_argument('--n_heads', type=int,default=16)
    parser.add_argument('--mlp_ratio', type=int,default=4)
    parser.add_argument('--dropout', type=float, default = 0.1,help='dropout for MLP blocks in transformer blocks')
    
    ## decoder arguments 
    parser.add_argument('--decoder_depth',type =int, default=8)
    parser.add_argument('--decoder_width', type =int, default= 512)
    #for now settings decoder transformer blocks are the same
    
    
    # class tokens for encoder and decoder - open question for whether this helps for when final task is semantic segmentation
    parser.add_argument('--no_cls_token_encoder',action='store_true', help= 'No cls token prepended to embedded token inputsfor the encoder.')
    parser.add_argument('--no_cls_token_decoder',action='store_true', help= 'No cls token prepended to embedded token inputs for the encoder.')

    






