import torch
import torch.nn as nn
from .attention import AttentionGate

"""
U-Net implementation with Attention Gates to improve crack segmentation.

This model extends the standard U-Net by adding attention gate modules in the decoder
so the network can focus on crack-related features and suppress background noise.
"""

class DoubleConv(nn.Module):
    """
    Double convolution block (U-Net building block).

    Two consecutive 3x3 convolutions, each followed by BatchNorm and ReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Improved U-Net for crack segmentation.

    Main changes compared to a vanilla U-Net:
    1) AttentionGate on each decoder level.
    2) BatchNorm for training stability.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        out_channels: Number of output channels (default: 1 for binary mask logits).
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder path: downsampling with increasing channels.
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck with the largest number of channels.
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder path: upsampling + attention + concatenation + convolution.
        
        # Decoder level 4 (deepest)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec4 = DoubleConv(1024, 512)
        
        # Decoder level 3
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = DoubleConv(512, 256)
        
        # Decoder level 2
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = DoubleConv(256, 128)
        
        # Decoder level 1 (shallowest)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv(128, 64)
        
        # Final output: 1x1 conv to logits.
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Downsampling with max pooling.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass:
        1) Encoder extracts multi-scale features.
        2) Bottleneck captures global context.
        3) Decoder restores spatial detail with attention-gated skips.
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder (deepest level)
        dec4 = self.up4(bottleneck)
        enc4_att = self.att4(dec4, enc4)
        dec4 = torch.cat((dec4, enc4_att), dim=1)
        dec4 = self.dec4(dec4)
        
        # Decoder level 3
        dec3 = self.up3(dec4)
        enc3_att = self.att3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3_att), dim=1)
        dec3 = self.dec3(dec3)
        
        # Decoder level 2
        dec2 = self.up2(dec3)
        enc2_att = self.att2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2_att), dim=1)
        dec2 = self.dec2(dec2)
        
        # Decoder level 1
        dec1 = self.up1(dec2)
        enc1_att = self.att1(dec1, enc1)
        dec1 = torch.cat((dec1, enc1_att), dim=1)
        dec1 = self.dec1(dec1)
        
        # Output logits (apply sigmoid externally to get probabilities)
        return self.final_conv(dec1) 
