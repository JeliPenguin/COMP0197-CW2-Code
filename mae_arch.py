
# data
#import modules
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


default_args = {
    'img_size': 224,
    'P': 16
    'C': 3
    'num_heads': 10
    ;

}



class MAE(nn.module):
    def __init__(self,args=default_args):
                 
                 
                 img_size = args.image_size, channels= 3, num_classes =10, embed_dim=1024, patch_size=self.patch_size, num_heads=10, encoder_depth=12, mlp_ratio=4, dropout=0.1):
       

        # standard for ViTs
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.embed_dim = embed_dim # i.e.  encoder width
        self.num_heads = num_heads
        self.mlp_ratio = 4
        self.dropout = 0.1
        self.flat_image_patch_size = self.patch_size**2 * channels
        
        # encoder 
        self.embed_patch_encoder= Embedder(img_size,patch_size,channels,embed_dim,mae=False,cls=True)
        self.encoder= nn.Sequential(*[TransformerEncoderBlock(embed_dim=self.embed_dim,num_heads=num_heads , mlp_ratio = 4.0, dropout=0.1)
                                     for l in range(self.encoder_depth)])
        self.encoder_norm = nn.LayerNorm(self.embed_dim)
        

        
        #MAE stuff:
        self.mask_ratio = 0.8
        self.unshuffle_index = 0 # will be tensor of permutations for undoing the shuffle operation done when masking
        self.keep_len = round(self.embed_patch.seq_len *(1-self.mask_ratio)) # used to select the first (1-mask ratio) of the shuffled patches
        


        
        
        #decoder
        self.decoder_depth = 8 #arguments
        self.decoder_width = 512 # arguments
        self.enc_to_dec_ = nn.Linear(self.embed_dim, self.decoder_width)
        self.mask_token= nn.Parameter(torch.zeros(1, 1, self.decoder_width))
        self.n_mask_tokens = round(self.embed_patch.seq_len * self.mask_ratio)

        self.decoder_embed_patch = Embedder(img_size,patch_size,channels,self.decoder_width,decoder=True) # don't 'linearly project' patches just add positional embedding
        self.decoder = nn.Sequential(*[TransformerEncoderBlock(embed_dim=self.decoder_width,num_heads=num_heads , mlp_ratio = 4.0, dropout=0.1)
                                     for l in range(self.decoder_depth)])
        self.decoder_norm = nn.LayerNorm(self.decoder_width)
        

        
        

        #decoder to image reconstruction
        self.dec_to_image_patch = nn.Linear(self.decoder_depth,self.flat_image_patch_size, bias=True) 

        self.apply(self._init_weights)
        self.initialize_params()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def initialize_params(self):
        torch.nn.init.normal(self.mask_tokens)
        
    

    def forward(self,x):
        # input should be images of shape [batch_size, C, H,W ]
        
        x = self.embed_patch_encoder(x,cls=True) # converts into patches, embeds patches, append cls token , then add positional embedding

        
        x, unshuffle_indices, mask_idxs = self.rand_shuffle_and_sample(x)

        x= self.encoder(x)
        x = self.encoder_norm(x)
        #encoder and decoder have different depths - linear transformation to handle this as in MAE paper 
        x= self.enc_to_dec(x) # this is our patch embedding for the decoder, no need to linearly project again. in self.embed_patch_decoder
        x = self.add_mask_tokens_and_unshuffle(x,unshuffle_indices)
        
        
        x = self.embed_patch_decoder(x,cls=True) # cls and add position embeddings
        x = self.decoder(x)
        x = self.decoder_norm(x)

        #upsample back to image. 
        x = self.dec_to_image_patch(x)
        
        # remove cls token: it's not a patch in the image but it's helped to aggregate the image
        x = x[:,1:, :]


        
        return x , mask_idxs # x is the reconstructed image, masks_idxs are the indices of the patches in each image
    

    def loss(self, images ,x,mask_idxs ):
     #'target' is image before flattening, need to convert into patches
    # loss on the MSE between reconstructed images and originals in pixel space but only on masked_patches
    # mean loss on removed patches

    
    # images (torch.Tensor): Input tensor of shape [batch_size, 3, H, W].
    # x are the pred should be of shape: [batch_size, sequence_length, P**2*C]
        targs = self.img_to_patch(images, self.patch_size)
        
        # now we'll use torch.gather to get the masked patches efficiently

        targs_masked_patches =  self.get_masked_patches(targs,mask_idxs)
        x_masked_patches = self.get_masked_patches(x,mask_idxs)

        return self.mse_per_patch(x_masked_patches,targs_masked_patches)


    def mse_per_patch(self,preds,targs):
        criterion = nn.MSELoss(reduction=None)
        mse_per_patch = criterion(preds,targs).mean(dim=(0,1))
        return mse_per_patch
        



    def get_masked_patches(self, x, mask_idxs):
        # use torch.gather to do this efficiently
        masked_patches = torch.gather(x,1, mask_idxs.unsqueeze(2).expand(-1,-1,x.shape[2]))
        return masked_patches


    


    # given a set of embedded patches 

    def rand_shuffle_and_sample(self,x):
        #returns subset (1-mask ratio) of randomly selected tokens to be passed to the encode
        # the indicess for unshuffling later
        # x should be of shape batch_size, seq_length, D]
        
        batch_size = x.shape[0]
        


        # model doesn't need to shuffled indices, only the inverse indices i.e. the indices to reverse the unshuffling. 


        # generate random permutations for each batch's token sequence
        shuffled_indices = torch.stack([torch.randperm(self.seq_len) for _ in range(batch_size)]) 

        

        # shuffle the sequences using the generated indices
        shuffled_tokens = torch.stack([x[i, idx] for i, idx in enumerate(shuffled_indices)]) 
        # 'enumerating' on a tensor creates an iteratable and the first element of our iteration variable i indexes the batch, the second indexes the sequences dimension. stack 

        # but with a permuation indexer. i.e. a list of all the indices for that dimension but in a different order. We do this for each batch. 
        # to unshuffle, use the inverse indices. This is the inverse of the permutation defined by shuffled indices
        
        inverse_indices = torch.argsort(shuffled_indices, dim=1) 

        #take the first (1-mask_ratio) shuffled tokens 
        unmasked_patches = shuffled_tokens[:, :self.keep_len, :]

       

        # to do: make this more efficient - this func. is called each train iteration which happens way more often than evaluation
    
        # image masks for comparing reconstruction
        from_idx = int(self.seq_len * (1-self.mask_ratio))  
        masks_idxs = shuffled_indices[:, from_idx:] # indices of patches in each batch of the original image that are masked.  
        # when evaluating you'lll have a sequence of the pure input image patches before encoder embeds them in
        # then take this sequence of pure image patches[mask_ids]= 0 then plot images # these will be patched out images

        return unmasked_patches, inverse_indices,  masks_idxs

    def add_mask_tokens_and_unshuffle(self,x, inverse_indices):

        # add mask tokens
        #we removed 
        #n_masktokens per batch 
        
        mask_tokens = self.mask_token.repeat(x.shape[0],self.n_mask_tokens,1)
        # add mask tokens to end 
        x_with_masks = torch.cat((x, mask_tokens), dim=1)
        #unshuffle tokens (check size )
        x_with_masks_unshuffled = torch.stack([x_with_masks[i, idx] for i, idx in enumerate    (inverse_indices)])
        return x_with_masks_unshuffled
    






    def img_to_patch(self,images):
        """
        Converts a batch of images into a batch of sequences of flattened patches.

        Args:
            images (torch.Tensor): Input tensor of shape [batch_size, 3, H, W].
            patch_size (int, optional): Size of each patch. Defaults to 4.


        Returns:
            torch.Tensor: Output tensor of shape [batch_size, HW / P^2, C * P^2].
        """

        batch_size, c , h, w= images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, -1, c * self.patch_size * self.patch_size)
        return patches
    


    # should take image size 
    def patch_to_img(self, patches):
        """
        Converts a batch of patch embeddings back into a batch of images.

        Args:
            patch_embeddings (torch.Tensor): Input tensor of shape [batch_size, HW / P^2, C * P^2].
            image_size (tuple, optional): Size of the original images (height, width). Defaults to (32, 32).
            patch_size (int, optional): Size of each patch. Defaults to 4.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 3, H, W].
        """

        batch_size, seq_len, channels = patches.shape 
        height, width = self.image_size
        channels_per_patch = channels // (self.patch_size * self.patch_size)

        patches = patches.view(batch_size, height // self.patch_size, width // self.patch_size, channels_per_patch, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        images = patches.view(batch_size, channels_per_patch, height, width)
        return images


# adds positional embedding after initial token embedding
class pos_embedder(nn.Module):
    def __init__(self,seq_len,embed_dim,sin_cos=False):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.sin_cos = sin_cos
        if sin_cos: # use sin-cos embedding from attention is all hou need
            self.pos_embed = nn.Parameter(self.sin_cos_pos_embed(), requires_grad=False)
        else: # elset trainable linear transformation
            self.pos_embed = nn.Parameter(torch.randn(seq_len, embed_dim))

    def forward(self,ins ):
        # ins is batch of patch(token) embeddings to which we add positional embeddings. 
        # ins should have shape [batch_size, seq_length, embed_dim]
        return ins + self.pos_embed
    
    def sin_cos_pos_embed(self):
        """ Generate sine-cosine positional encoding.

        Returns:
            pos_encoding: the positional encoding matrix of shape [seq_len, embed_dim].
        """
        position = torch.arange(self.seq_len, dtype=torch.float).reshape(-1, 1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embed_dim))

        pos_encoding = torch.zeros((self.seq_len,self.embed_dim ))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding


class Embedder( nn.Module):
    "token and position embedding for transformer blocks"

    # uses our images to patches function.
    # use torch.einsum to embed patches in batches.
    #img size is H=W
    #self.args.cls 
    def __init__(self,img_size,patch_size,channels,embed_dim=768,mae=False, cls=True,pos_embed_sin_cos=False, decoder=False):
        super().__init__()
        self.patch_size = patch_size
        self.in_dim = channels* patch_size**2 #flattened
        self.seq_len = int(img_size**2/patch_size**2) 
        self.embed_dim =embed_dim
        self.decoder = decoder
        
        
        if not self.decoder:
            self.patch_embed=  nn.Linear(self.in_dim, self.embed_dim)
        
        
        
        # introduce cls_token to be prepended to each sequence. Increments the effective sequence length and the cls token needs a positional embedding
        if cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.pos_embed  = pos_embedder(self.seq_len+1,self.embed_dim,sin_cos=True)
        
        # sequence 
        else:
            self.pos_embed  = pos_embedder(self.seq_len,self.embed_dim,sin_cos=True)
        


    def forward(self,tokens):
        # for encoder(decoder =False) inputs should be images of shape [batch_size, C,H,W]
        #  if embedding for the decoder inputs , then inputs are already of shape [batch_size, sequence_length]
        # if you're embedding tokens for the decoder the don't need to convert image into patches. else
        if not self.decoder: # i.e. if embedding for the encoder 
            patches =  self.img_to_patch(tokens,self.patch_size)
            patch_embeds = self.patch_embed(patches)
        
        # add clstokens and positional embedding
        if self.cls:
            cls_tokens = self.cls_token.expand(tokens.shape[0],1,self.embed_dim)

            z = torch.cat((cls_tokens, patch_embeds),dim=1)  # append cls tokens
        
        # always add positional embedding
        z = self.pos_embed(z) 




        #patch_embeds= torch.einsum('ik,blk -> bli',[self.patch_embed.weight, patches]) # project each patch in each sequence
         # NB: einsum in this way is cool but it's not necessary and nn.linear is much faster.
        #Key observation: nn.Linear can operate on tensors with more than two dimensions
        #i.e. tensor of shape [batch_size, *, input_features], nn.Linear applies the linear transformation to the last dimension (input_features) of the tensor,
        #treating all preceding dimensions as part of the batch. this is pretty cool

        

        return z




class TransformerEncoderBlock(nn.Module):
    def __init__(self,embed_dim=1024, num_heads=12, mlp_ratio = 4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp= MLP(embed_dim, mlp_ratio, dropout)



    def forward(self,x):

        res_x = x
        #need to transpose for multiheaded self atttention (see PyTorch documentation)
        x = self.ln1(x)
        x_transposed = x.transpose(0,1)
        msa_x, _ = self.msa(x_transposed,x_transposed,x_transposed)
        msa_x = msa_x.transpose(0,1)

        x = msa_x + res_x

        x_prime =  self.mlp(self.ln2(x)) + x

        return x_prime








class MLP(nn.Module):

    def __init__(self,input_dim, mlp_ratio=4.0, dropout_prop= 0.1):
        # mlp ratio is the how much larger the width of each hidden layer is relative to the input dim
        # dropout rate is the proportion of zeroed out layer activations
        super().__init__()
        hidden_width = int(input_dim* mlp_ratio)

        self.ff_1 = nn.Linear(input_dim , hidden_width)
        self.gelu = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_prop)
        self.ff_2=nn.Linear(hidden_width,input_dim)
        self.dropout_2 =nn.Dropout(dropout_prop)

    def forward(self, x):
        x= self.ff_1(x)
        x= self.gelu(x)
        x= self.dropout_1(x)
        x = self.ff_2(x)
        x= self.dropout_2(x)

        return x


class MLP_class_head(nn.Module):
    def __init__(self,input_dim,num_classes):
      super().__init__()
      self.ff = nn.Linear(input_dim , num_classes)
      self.tanh= nn.Tanh()

    def forward(self,x):
        x= self.ff(x)
        x= self.tanh(x)
        return x
    



