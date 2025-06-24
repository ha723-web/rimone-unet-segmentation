# model.py

""" 
This file defines what the U-Net model looks like, its architecture, 
layers, and how it processes images to generate predictions.
example - Like designing the blueprint of a car (engine, wheels, structure).
"""

#Importing libraries 

import torch  # Main PyTorch library
import torch.nn as nn # Tools for building neural network layers
import torch.nn.functional as F #extra functions like activation or loss functions


# Creating a model called UNet, which inherits from PyTorch’s nn.Module.

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # zooming out to get a general understanding of the image.
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)
        
        # To extract the most compressed and powerful representation of the image.
        self.bottleneck = conv_block(512, 1024)
        
        # zooming back in to recover the original image shape.
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        # The image is first downsampled by the encoder to extract deep features
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # passes through the bottleneck for compressed understanding
        b = self.bottleneck(self.pool(e4))
        
        # upsampled by the decoder while merging skip connections to restore spatial detail.
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # final output is a 2-channel mask predicting the optic disc and cup regions.
        return self.final(d1)
