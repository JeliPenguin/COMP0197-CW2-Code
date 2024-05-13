import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
class Encoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.encoder_cls = not self.args.no_cls_token_encoder 
        
        self.seq_len = (self.args.img_size**2)/(self.args.patch_size**2)
        self.seq_len = int(self.seq_len)
        self.flat_image_patch_size = self.args.patch_size**2 * self.args.c

        self.embed_patch_encoder= Embedder(self.args.patch_size, self.seq_len,self.flat_image_patch_size,self.args.encoder_width,cls=self.encoder_cls, pos_embed_sin_cos=True)
        self.encoder= nn.Sequential(*[TransformerEncoderBlock(embed_dim=self.args.encoder_width,num_heads=self.args.n_heads , mlp_ratio = 4, dropout=0.1)
                                     for l in range(self.args.encoder_depth)])
        self.encoder_norm = nn.LayerNorm(self.args.encoder_width)
        self.keep_len = int(self.seq_len *(1-self.args.mask_ratio)) # used to select the first (1-mask ratio) of the shuffled patches

    def forward(self, x):
        x = self.embed_patch_encoder(x) # converts into patches, embeds patches, append cls token , then add positional embedding
        x, unshuffle_indices, mask_idxs = self.rand_shuffle_and_sample(x)
        x= self.encoder(x)

        x = self.encoder_norm(x)
        return x, unshuffle_indices, mask_idxs


    def rand_shuffle_and_sample(self,x):
        #returns subset (1-mask ratio) of randomly selected tokens to be passed to the encode
        # the indicess for unshuffling later
        # x should be of shape batch_size, seq_length, D]
        batch_size = x.shape[0]
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
        masks_idxs = shuffled_indices[:, self.keep_len:] # indices of patches in each batch of the original image that are masked.  
        # when evaluating you'lll have a sequence of the pure input image patches before encoder embeds them in
        # then take this sequence of pure image patches[mask_ids]= 0 then plot images # these will be patched out images
        return unmasked_patches, inverse_indices,  masks_idxs

class SegmentHead(nn.Module):
    def __init__(self, in_ch=1024, output_shape=(128, 128), bilinear=True):
        super(SegmentHead, self).__init__()
        self.output_shape = output_shape
        self.up6 = (Up(15, 8, bilinear=bilinear, up_scale=1))
        self.outc = (OutConv(8, 1))

    def forward(self, x):
        x = self.up6(x)
        x = self.outc(x)
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        x = torch.sigmoid(x)
        return x
    
    def process_res(self, res):
        b, n, c = res.shape
        res = res.permute(0, 2, 1).view(b, c, 17, 12)
        return res
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, up_scale=2):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=True)  
            self.conv = DoubleConv(in_channels, out_channels)
        else: 
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
