import torch
from src.mae_code.mae_parts import Encoder
import torch.nn as nn

class DefaultArgs:
    def __init__(self):
        self.img_size = 128  # adjusted for Tiny ImageNet
        self.patch_size = 4 # You can experiment with smaller sizes, like 8, if desired
        self.encoder_width = 512  # Adjusted for a smaller model
        self.n_heads = 8  # Fewer heads given the reduced complexity
        self.encoder_depth = 5  # Fewer layers [5]
        self.decoder_width = 256  # Adjusted decoder width
        self.decoder_depth = 4  # Fewer layers in decoder
        self.mlp_ratio = 4.0
        self.dropout = 0.1
        self.mask_ratio = 0.75
        self.no_cls_token_encoder = False
        self.no_cls_token_decoder = False
        self.c = 3  # Number of colorchannels  (RGB)


class MAE(nn.Module):
    def __init__(self,args=DefaultArgs()):
        super().__init__()
        self.args = args 
        
        self.encoder_block = Encoder(args) 
                
        self.seq_len = (self.args.img_size**2)/(self.args.patch_size**2)
        self.seq_len = int(self.seq_len)
        self.flat_image_patch_size = self.args.patch_size**2 * self.args.c

        self.decoder_cls = not self.args.no_cls_token_decoder
                
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
        x, unshuffle_indices,mask_idxs = self.encoder_block(x)
        
        #encoder and decoder have different depths - linear transformation to handle this as in MAE paper 
        x= self.enc_to_dec(x) # this is our patch embedding for the decoder, no need to linearly project again. in self.embed_patch_decoder

        # print(unshuffle_indices)
        x = self.add_mask_tokens_and_unshuffle(x,unshuffle_indices)
        
        # linear project just need pos
        x = self.embed_patch_decoder(x) # cls and add position embeddings
        x = self.decoder(x)
        x = self.decoder_norm(x)

        #upsample back to image. 
        x = self.dec_to_image_patch(x)
        
        # if decoder has a cls remove cls token: it's not a patch in the image but it's helped to aggregate info in the image
        if self.decoder_cls:
            x = x[:,1:, :]
        
        
        return x , mask_idxs.to(self.args.device) # x is the reconstructed image, masks_idxs are the indices of the patches in each image
    

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

        
        position = torch.arange(self.seq_len, dtype=torch.float).reshape(-1, 1) # we're
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embed_dim))

        pos_encoding = torch.zeros((self.seq_len,self.embed_dim ))
        pos_encoding[:, 0::2] = torch.sin(position * div_term) # sine in even indices of embed dim 
        pos_encoding[:, 1::2] = torch.cos(position * div_term) #cos in odd indices of embed im
        
        return pos_encoding


class Embedder( nn.Module):
    "token and position embedding for transformer blocks"

    # uses our images to patches function.
    # use torch.einsum to embed patches in batches.
    #img size is H=W
    #self.args.cls 
    def __init__(self,patch_size,seq_len,in_dim, embed_dim,cls=True,pos_embed_sin_cos=False, decoder=False):
        super().__init__()

        self.decoder = decoder # whether we're doing this for the encoder or decoder
        self.seq_len = seq_len # sequence length before cls token prepended
    
        self.patch_size = patch_size
        self.in_dim = in_dim 

        self.cls = cls
        # embed encoder tokens with lin proj.
        if not self.decoder:
            self.patch_embed=  nn.Linear(in_dim, embed_dim)
        
        # introduce cls_token to be prepended to each sequence. Increments the effective sequence length and the cls token needs a positional embedding

        if cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1,embed_dim))

        self.pos_embed  = pos_embedder(self.seq_len + (1 if cls else 0 ),embed_dim,sin_cos=True) # 

    def forward(self,tokens):
        # for encoder(decoder =False) inputs should be images of shape [batch_size, C,H,W]
        #  if embedding for the decoder inputs , then inputs are already of shape [batch_size, sequence_length]
        # if you're embedding tokens for the decoder the don't need to convert image into patches. else and linear projection was handled outside of this just 
        if not self.decoder: # i.e. if embedding for the encoder 
            patches =  self.img_to_patch(tokens,self.patch_size)
            patch_embeds = self.patch_embed(patches)
        else: 
            patch_embeds = tokens
        
        # add clstokens and positional embedding
        if self.cls:
            cls_tokens = self.cls_token.expand(tokens.shape[0],-1,-1)

            z = torch.cat((cls_tokens, patch_embeds),dim=1)  # append cls tokens
        else:
            z = patch_embeds
        
        # always add positional embedding
        z = self.pos_embed(z) 

        #patch_embeds= torch.einsum('ik,blk -> bli',[self.patch_embed.weight, patches]) # project each patch in each sequence
         # NB: einsum in this way is cool but it's not necessary and nn.linear is much faster.
        #Key observation: nn.Linear can operate on tensors with more than two dimensions
        #i.e. tensor of shape [batch_size, *, input_features], nn.Linear applies the linear transformation to the last dimension (input_features) of the tensor,
        #treating all preceding dimensions as part of the batch. this is pretty cool
        return z
    

    def img_to_patch(self,images, normalize=False):
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
        self.ff_2= nn.Linear(hidden_width,input_dim)
        self.dropout_2 = nn.Dropout(dropout_prop)

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
    