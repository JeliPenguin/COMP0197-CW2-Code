import torch
import torch.nn as nn

class PatchCreator(nn.Module):
    """
    Creates flattened, embedded patches from input images using a convolutional layer.

    This module converts images into a sequence of flattened patches by applying a single convolutional layer,
    where each patch is then embedded into a higher-dimensional space.
    """

    def __init__(
        self,
        size_patch=16,
        channels_input=3,
        embed_dimension=768
    ):
        """
        Initializes the PatchCreator module.

        Arguments:
            size_patch (int): Height and width of each square patch
            channels_input (int): Number of channels in the input images
            embed_dimension (int): Dimensionality of the patch embeddings
        """
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels=channels_input,  # Number of input channels
            out_channels=embed_dimension,  # Dimension of the embedding space
            kernel_size=size_patch,  # Size of the patch
            stride=size_patch  # Stride equal to the patch size
        )

    def forward(self, input_x):
        """
        Forward pass of the PatchCreator.

        Arguments:
            input_x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Embedded patches
        """
        patches = self.patch(input_x).flatten(2).transpose(1, 2)
        return patches

class TransformerAttentionBlock(nn.Module):
    """
    Implements a single Transformer attention block with a multi-head self-attention mechanism and a feed-forward network.

    The block processes input features through a layer normalization, followed by a multi-head self-attention mechanism and
    a simple feed-forward network, with residual connections around each.
    """

    def __init__(self, dimension_embed, rate_dropout=0.0, heads_num=1, dimension_hidden=512):
        """
        Initializes the TransformerAttentionBlock module.

        Arguments:
            dimension_embed (int): Dimensionality of the input and output feature space
            rate_dropout (float): Dropout rate applied in the attention mechanism and the feed-forward network
            heads_num (int): Number of heads in the multi-head attention mechanism
            dimension_hidden (int): Dimensionality of the hidden layer in the feed-forward network
        """
        super().__init__()
        self.norm_pre = nn.LayerNorm(dimension_embed, eps=1e-06)

        # Multi-head self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=dimension_embed,
            num_heads=heads_num,  # Number of attention heads
            dropout=rate_dropout,  # Dropout rate
            batch_first=True  # Batch dimension is the first dimension
        )

        # Normalization
        self.norm = nn.LayerNorm(dimension_embed, eps=1e-06)

        # Feed-forward network applied after the attention mechanism
        self.MLP = nn.Sequential(
            nn.Linear(dimension_embed, dimension_hidden),  # Increase dimension to hidden_dim
            nn.GELU(),  # GELU activation function for non-linearity
            nn.Dropout(rate_dropout),  # Dropout for regularization
            nn.Linear(dimension_hidden, dimension_embed),  # Project back to the embedding dimension
            nn.Dropout(rate_dropout)  # Dropout for regularization
        )

    def forward(self, input_x):
        """
        Forward pass of the TransformerAttentionBlock.

        Arguments:
            input_x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        norm_x = self.norm_pre(input_x)

        # Apply multi-head attention; ignore attention weights returned
        input_x = input_x + self.attention(norm_x, norm_x, norm_x)[0]

        # Apply the MLP to the output of the attention block
        input_x = input_x + self.MLP(self.norm(input_x))

        return input_x

class VisionTransformer(nn.Module):
    """
    Implements a Vision Transformer (ViT) model for image classification.

    The model applies a convolutional layer to create flattened, embedded patches from input images,
    adds a class token and positional embeddings, and processes these through a sequence of Transformer
    attention blocks. The output of the class token is then used for classification.
    """

    def __init__(
        self,
        size_image=32,
        size_patch=4,
        channels_in=3,
        dimension_embed=384,
        layers_number=2,
        heads_number=12,
        dimension_hidden=1536,
        dropout_rate=0.0,
        classes_num=10
    ):
        """
        Initializes the VisionTransformer module.

        Arguments:
            size_image (int): Height and width of the input images
            size_patch (int): Height and width of each square patch
            channels_in (int): Number of channels in the input images
            dimension_embed (int): Dimensionality of the patch embeddings and the Transformer encoder
            layers_number (int): Number of Transformer blocks/layers
            heads_number (int): Number of heads in the Transformer's multi-head attention mechanism
            dimension_hidden (int): Dimensionality of the hidden layer in the Transformer's feed-forward network
            dropout_rate (float): Dropout rate applied in the Transformer
            classes_num (int): Number of classes for the classification task
        """
        super().__init__()

        # Compute the number of patches
        num_patches = (size_image // size_patch) ** 2

        self.patches = PatchCreator(
            size_patch=size_patch,
            channels_input=channels_in,
            embed_dimension=dimension_embed
        )

        # Positional embeddings
        self.embedding_pos = nn.Parameter(torch.randn(1, num_patches + 1, dimension_embed))

        # For Classification
        self.token_cls = nn.Parameter(torch.randn(1, 1, dimension_embed))

        # Stack of attention blocks
        self.layers_attn = nn.ModuleList([
            TransformerAttentionBlock(dimension_embed, dropout_rate, heads_number, dimension_hidden) for _ in range(layers_number)
        ])

        # Dropout layer for the output of the positional embeddings
        self.dropout = nn.Dropout(dropout_rate)

        # Final layer normalization
        self.norm_final = nn.LayerNorm(dimension_embed, eps=1e-06)

        # Linear layer for classification
        self.head_linear = nn.Linear(dimension_embed, classes_num)

        # Initialize weights
        self.apply(self.init_weights_custom)

    def init_weights_custom(self, m):
        """
        Initializes weights of the VisionTransformer module.

        Arguments:
            m (nn.Module): A module within the VisionTransformer
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_x):
        """
        Forward pass of the VisionTransformer.

        Arguments:
            input_x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: The logits tensor
        """
        # Convert images into patch embeddings
        x = self.patches(input_x)
        b, n, _ = x.shape  # batch size, number of patches, embedding dimension

        # Expand class token for the batch and concatenate with patch embeddings
        cls_tokens = self.token_cls.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x += self.embedding_pos

        # Apply dropout to the embeddings
        x = self.dropout(x)

        # Pass through the stack of attention layers
        for layer in self.layers_attn:
            x = layer(x)

        # Apply the final layer normalization
        x = self.norm_final(x)

        # Take the output of the class token for classification
        x = x[:, 0]

        return self.head_linear(x)
