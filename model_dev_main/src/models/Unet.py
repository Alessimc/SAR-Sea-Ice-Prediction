import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
""" 
Implementation of the U-Net architecture with some modifications.
Original paper: http://arxiv.org/abs/1505.04597

UP and DOWN blocks can be configured to use either standard DoubleConv or
a residual version DoubleConvRes.
"""

# def timer(func):
#     def wrapper(*args, **kwargs):
#         t1 = time()
#         result = func(*args, **kwargs)
#         t2 = time()
#         print(f"Function {func.__name__!r} executed in {(t2 - t1):.4f}s")
#         return result
#     return wrapper

# Double convolutions used in contracting and expansive path
class DoubleConv(nn.Module):
    """
    (Conv -> ReLU -> Conv -> ReLU)
    Original U-Net paper uses no padding. 
    Keeping same spatial dimensions with padding=1 for 3x3 convolutions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # TODO: possibly add batchnorm here (wasn't typical at the time of U-Net paper)
            nn.GroupNorm(1, out_channels),  # instance normalization (group norm with 1 group) to handle small batch sizes
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # TODO: possibly add batchnorm here
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
# Alteration of the above with residual connection. Making it a residual U-Net.
class DoubleConvRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(1, out_channels)
        
        # if input and output channels differ, add a 1x1 conv to match dimensions
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        x += residual
        x = self.relu(x)
        return x



# Downsampling block with avgpool then double conv (was maxpool in original U-Net)
class Down(nn.Module):
    """
    NOTE: Kvanum et al. (2024) 10.5194/egusphere-2023-3107 differences.
    - Downsample by a factor of 4 and use average pooling instead of max pooling.
    - Mention batch normalization, but replace with group normalization due to small batch sizes during training on A100 GPU.
    """
    def __init__(self, in_channels, out_channels, block=DoubleConv):
        super().__init__()
        self.conv = block(in_channels, out_channels)
        self.pool = nn.AvgPool2d(2) # halving spatial dimensions.
        

    def forward(self, x):
        x_conv = self.conv(x)
        x_pooled = self.pool(x_conv)
        return x_conv, x_pooled  # return both conv output (for skip connection) and pooled output (for next layer input).
    

# Upsampling block with transposed conv, concatenation, then double conv
class Up(nn.Module):
        """
        in_ch: number of channels coming into the up block (usually from bottleneck or previous up)
        out_ch: desired output channels after the DoubleConvRes or DoubleConv
        """
        def __init__(self, in_channels, out_channels, block=DoubleConv):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) # learned upsampling
            self.conv = block(in_channels // 2 + in_channels // 2, out_channels) # concatenation of skip connection and upsampled input gets mapped to out_channels

        def forward(self, x, x_skip_connection):
            x = self.up(x)
            x = torch.cat([x, x_skip_connection], dim=1) # concatenate decoder input and skip connection.
            return self.conv(x)
        

# Assembly of the full U-Net
class UNet_4layers(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32, block=DoubleConv):
        super().__init__()
        # Encoder
        self.down1 = Down(in_channels, base_channels, block=block)
        self.down2 = Down(base_channels, base_channels * 2, block=block)
        self.down3 = Down(base_channels * 2, base_channels * 4, block=block)
        self.down4 = Down(base_channels * 4, base_channels * 8, block=block)

        # Bottleneck
        self.bottleneck = block(base_channels * 8, base_channels * 16)

        # Decoder
        self.up4 = Up(base_channels * 16, base_channels * 8, block=block)
        self.up3 = Up(base_channels * 8, base_channels * 4, block=block)
        self.up2 = Up(base_channels * 4, base_channels * 2, block=block)
        self.up1 = Up(base_channels * 2, base_channels, block=block)

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1, p1 = self.down1(x) # e(ncode)1 for skip connection, p(ooled)1 for next layer input.
        e2, p2 = self.down2(p1)
        e3, p3 = self.down3(p2)
        e4, p4 = self.down4(p3)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.up4(b, e4) # d(ecode)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        output = self.final_conv(d1)
        return output


class ResUNet_4layers(UNet_4layers):
    def __init__(self, in_channels, out_channels, base_channels=32):
        super().__init__(in_channels, out_channels, base_channels=base_channels, block=DoubleConvRes)


class UNet_3layers(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32, block=DoubleConv):
        super().__init__()
        # Encoder
        self.down1 = Down(in_channels, base_channels, block=block)
        self.down2 = Down(base_channels, base_channels * 2, block=block)
        self.down3 = Down(base_channels * 2, base_channels * 4, block=block)

        # Bottleneck
        self.bottleneck = block(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = Up(base_channels * 8, base_channels * 4, block=block)
        self.up2 = Up(base_channels * 4, base_channels * 2, block=block)
        self.up1 = Up(base_channels * 2, base_channels, block=block)

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1, p1 = self.down1(x) # e(ncode)1 for skip connection, p(ooled)1 for next layer input.
        e2, p2 = self.down2(p1)
        e3, p3 = self.down3(p2)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        output = self.final_conv(d1)
        return output


class UNet_5layers(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32, block=DoubleConv):
        super().__init__()
        # Encoder
        self.down1 = Down(in_channels, base_channels, block=block)
        self.down2 = Down(base_channels, base_channels * 2, block=block)
        self.down3 = Down(base_channels * 2, base_channels * 4, block=block)
        self.down4 = Down(base_channels * 4, base_channels * 8, block=block)
        self.down5 = Down(base_channels * 8, base_channels * 16, block=block)

        # Bottleneck
        self.bottleneck = block(base_channels * 16, base_channels * 32)

        # Decoder
        self.up5 = Up(base_channels * 32, base_channels * 16, block=block)
        self.up4 = Up(base_channels * 16, base_channels * 8, block=block)
        self.up3 = Up(base_channels * 8, base_channels * 4, block=block)
        self.up2 = Up(base_channels * 4, base_channels * 2, block=block)
        self.up1 = Up(base_channels * 2, base_channels, block=block)

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1, p1 = self.down1(x) # e(ncode)1 for skip connection, p(ooled)1 for next layer input.
        e2, p2 = self.down2(p1)
        e3, p3 = self.down3(p2)
        e4, p4 = self.down4(p3)
        e5, p5 = self.down5(p4)

        # Bottleneck
        b = self.bottleneck(p5)

        # Decoder
        d5 = self.up5(b, e5)
        d4 = self.up4(d5, e4) # d(ecode)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        output = self.final_conv(d1)
        return output