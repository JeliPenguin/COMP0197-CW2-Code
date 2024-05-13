import torch
import torch.nn as nn
from src.oeq_mae_decoder_conv_layer.mae_parts import Encoder, Embedder, TransformerEncoderBlock, SegmentHead
 
class MAESegmentor(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args 
                
        self.seq_len = (self.args.img_size**2)/(self.args.patch_size**2)
        self.seq_len = int(self.seq_len)
        self.flat_image_patch_size = self.args.patch_size**2 * self.args.c

        self.decoder_cls = not self.args.no_cls_token_decoder
        
        self.encoder_block = Encoder(args)       
        
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
        self.decoder_blocks = nn.Sequential(*[TransformerEncoderBlock(embed_dim=self.args.decoder_width,num_heads=self.args.n_heads , mlp_ratio = self.args.mlp_ratio, dropout=self.args.dropout)
                                     for l in range(self.args.decoder_depth)])
        self.decoder_norm = nn.LayerNorm(self.args.decoder_width)
        
        #decoder to image reconstruction
        self.dec_to_image_patch = nn.Linear(self.args.decoder_width,self.flat_image_patch_size, bias=True) 

        # self.detection_head = nn.Conv2d(in_channels=(3 * len(self.args.out_indices)), out_channels=1, kernel_size=1)
        self.detection_head = SegmentHead(in_ch=(3 * len(self.args.out_indices)))
        self.out_indices = self.args.out_indices
        self.apply(self._init_weights)
        self.initialize_params()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def initialize_params(self):
        torch.nn.init.normal_(self.mask_token)
    
    def loss(self, images ,x,mask_idxs):
        targs = self.img_to_patch(images, self.args.patch_size, normalize=True)
        
        # now we'll use torch.gather to get the masked patches efficiently
        
        targs_masked_patches =  self.get_masked_patches(targs,mask_idxs)
        x_masked_patches = self.get_masked_patches(x,mask_idxs)
        # Debugging output
        return self.mse_per_patch(x_masked_patches,targs_masked_patches)

    def mse_per_patch(self,preds,targs):
        mse_per_patch = nn.MSELoss(reduction= 'mean')(preds,targs)
        return mse_per_patch
    
    def forward(self,x):
        # input should be images of shape [batch_size, C, H,W ]
        x,unshuffle_indices,mask_idxs = self.encoder_block(x)
        
        #encoder and decoder have different depths - linear transformation to handle this as in MAE paper 
        x= self.enc_to_dec(x) # this is our patch embedding for the decoder, no need to linearly project again. in self.embed_patch_decoder

        # print(unshuffle_indices)
        x = self.add_mask_tokens_and_unshuffle(x,unshuffle_indices)
        
        # linear project just need pos
        x = self.embed_patch_decoder(x) # cls and add position embeddings
        features = []
        for i in range(self.decoder_depth):
            x = self.decoder_blocks[i](x)
            if i in self.out_indices:
                features.append(x)
            
        for i in range(len(features)):
            features[i] = self.decoder_norm(features[i])
            features[i] = self.dec_to_image_patch(features[i])          
            # if decoder has a cls remove cls token: it's not a patch in the image but it's helped to aggregate info in the image
            if self.decoder_cls:
                features[i] = features[i][:,1:, :]
            features[i] = self.reconstruct_image(features[i])
        x = torch.cat(features, axis=1)
        x = self.detection_head(x)
        # x = torch.sigmoid(x)
        return x , mask_idxs.to(self.args.device) # x is the reconstructed image, masks_idxs are the indices of the patches in each image

    # def reconstruct_decode(self, x):
    #     feature_size = ((self.args.img_size // self.args.patch_size) ** 2) 
    #     x = x.view(x.shape[0], feature_size, self.args.patch_size, self.args.patch_size * self.args.c)
    #     return x
    
    def reconstruct_image(self, x):
        """
        Reshape flattened patches to the image format.
        
        Args:
            patches (Tensor): mae's image reconstructions in patches
         shape [batch_size, num_patches, patch_height * patch_width * channels]
            batch_size (int): The batch size.
        
        Returns:
            Tensor: Reconstructed images of shape [batch_size, channels, height, width]
        """
        patch_dim = self.flat_image_patch_size
        num_patches_per_side = self.args.img_size // self.args.patch_size
        x = x.view(x.shape[0], num_patches_per_side, num_patches_per_side, self.args.c, self.args.patch_size, self.args.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        images = x.view(x.shape[0], self.args.c, self.args.img_size, self.args.img_size)
        return images
    
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

    def get_masked_patches(self, x, mask_idxs):
        # use torch.gather to do this efficiently
        # NB: this doesn't mask things
        # if the input x is a patchified image, this retreives the patches that were later masked.
        # if the input x is a prediction, it retreives the predictions for the masked patches

        masked_patches = torch.gather(x,1, mask_idxs.unsqueeze(2).expand(-1,-1,x.shape[2]))
        return masked_patches

    def add_mask_tokens_and_unshuffle(self,x, inverse_indices):

        # add mask tokens
        #we removed 
        #n_masktokens per batch 

        
        
        mask_tokens = self.mask_token.repeat(x.shape[0],self.n_mask_tokens+1,1)
        # add mask tokens to end 
        x_with_masks = torch.cat((x, mask_tokens), dim=1)

        # print(x_with_masks.shape)
        #unshuffle tokens (check size )
        x_with_masks_unshuffled = torch.stack([x_with_masks[i, idx] for i, idx in enumerate(inverse_indices)])
        return x_with_masks_unshuffled
    

    def img_to_patch(self,images,patch_size, normalize=False):
        """
        Converts a batch of images into a batch of sequences of flattened patches.

        Args:
            images (torch.Tensor): Input tensor of shape [batch_size, 3, H, W].
            patch_size (int, optional): Size of each patch. Defaults to 4.


        Returns:
            torch.Tensor: Output tensor of shape [batch_size, HW / P^2, C * P^2].
        """

        batch_size, c , h, w= images.shape
        patches = images.unfold(2, patch_size,patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, -1, c * patch_size * patch_size)
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

