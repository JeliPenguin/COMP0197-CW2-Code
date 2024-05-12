import torch
from src.finetune.mae_parts import Encoder
import torch.nn as nn
import torch.nn.functional as F

class MAE(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args 
        
        self.encoder = Encoder(args)      
        self.decoder = Decoder()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.args.encoder_width))
        self.apply(self._init_weights)
        self.initialize_params()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def initialize_params(self):
        torch.nn.init.normal_(self.mask_token)
    
    def forward(self,x):
        # input should be images of shape [batch_size, C, H,W ]
        x, unshuffle_indices, mask_idxs = self.encoder(x)
        b, n, c = x.shape
        h_w = int(n**0.5)
        x = x.permute(0, 2, 1).view(b, c, 17, 12)       
        x = self.decoder(x) 
        return x
    
class Decoder(nn.Module):
    def __init__(self, output_shape=(128, 128), bilinear=True):
        super(Decoder, self).__init__()
        self.output_shape = output_shape

        self.up1 = (Up(1024, 512, bilinear=bilinear))
        self.up2 = (Up(512, 256, bilinear=bilinear))
        self.up3 = (Up(256, 128, bilinear=bilinear))
        self.up4 = (Up(128, 64, bilinear=bilinear, up_scale=1))
        self.outc = (OutConv(64, 1))

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
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
