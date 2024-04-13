import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class OverlappingPatchEmbed(nn.Module):
    """
    A module that creates overlapping patches from an input tensor and then projects them into an embedding space.
    
    Args:
        patch_size (int): The size of each patch. Default is 16.
        stride (int): The stride of the convolution. Default is 14.
        padding (int): The padding of the convolution. Default is 2.
        embed_dim (int): The number of output channels of the convolution, i.e., the dimension of the embedding space. Default is 768.
        channels (int): The number of input channels of the convolution. Default is 3.
    """
    def __init__(self, patch_size=16, stride=14, padding=2, embed_dim=768, channels=3):
        super().__init__()
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Forward pass of the OverlappingPatchEmbed module.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor with patches embedded.
        """
        x = self.proj(x)  # [B, C, H, W]
        x = x.flatten(2)  # [B, C, HW]
        x = x.transpose(1, 2)  # [B, HW, C]
        return x

class TransformerBlock(nn.Module):
    """
    A transformer block with LayerNorm and residual connections.
    
    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        feedforward_dim (int): The dimension of the feedforward network.
        dropout (float): The dropout rate. Default is 0.1.
    """
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of the TransformerBlock module.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after the transformer block.
        """
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    """
    A stack of Transformer blocks forming the encoder.
    
    Args:
        depth (int): The number of Transformer blocks in the stack.
        **kwargs: Additional keyword arguments for the TransformerBlock.
    """
    def __init__(self, depth, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(**kwargs) for _ in range(depth)])

    def forward(self, x):
        """
        Forward pass of the TransformerEncoder module.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after the encoder.
        """
        for layer in self.layers:
            x = layer(x)
        return x

class HybridViT(nn.Module):
    """
    A hybrid Vision Transformer model.
    
    Args:
        image_size (int): The size of the input images.
        patch_size (int): The size of each patch.
        stride (int): The stride of the patch embedding convolution.
        num_classes (int): The number of output classes.
        embed_dim (int): The dimension of the embedding space.
        depth (int): The number of Transformer blocks in the encoder.
        num_heads (int): The number of attention heads.
        feedforward_dim (int): The dimension of the feedforward network.
        channels (int): The number of input channels of the patch embedding convolution. Default is 3.
    """
    def __init__(self, image_size, patch_size, stride, num_classes, embed_dim, depth, num_heads, feedforward_dim, channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim

        # Calculate grid_size considering any padding (assuming no padding here)
        grid_size = ((image_size - patch_size) // stride) + 1
        num_patches = grid_size * grid_size

        self.patch_embed = OverlappingPatchEmbed(
            patch_size=patch_size, stride=stride, padding=0, embed_dim=embed_dim, channels=channels
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.encoder = TransformerEncoder(depth=depth, embed_dim=embed_dim, num_heads=num_heads, feedforward_dim=feedforward_dim)

        # Decoder setup
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        """
        Forward pass of the HybridViT module.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after the model.
        """
        x = self.patch_embed(x)
        b, n, _ = x.shape  # batch size, number of patches, embedding dimension

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.encoder(x)
        x = x[:, 1:].transpose(1, 2).view(b, self.embed_dim, int(np.sqrt(n)), int(np.sqrt(n)))
        x = self.decoder(x)
        return x
