from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two convolution blocks, the basic building block of U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    """Small educational U-Net for semantic segmentation."""

    def __init__(self, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.enc1 = DoubleConv(3, channels[0])
        self.enc2 = DoubleConv(channels[0], channels[1])
        self.enc3 = DoubleConv(channels[1], channels[2])
        self.bottleneck = DoubleConv(channels[2], channels[3])

        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(channels[3], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(channels[2], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(channels[1], channels[0])
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(F.max_pool2d(skip1, kernel_size=2))
        skip3 = self.enc3(F.max_pool2d(skip2, kernel_size=2))
        x = self.bottleneck(F.max_pool2d(skip3, kernel_size=2))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, skip3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, skip2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, skip1], dim=1))
        return self.classifier(x)
