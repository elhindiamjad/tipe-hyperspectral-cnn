"""2D CNN for patch-based hyperspectral classification."""
import torch
import torch.nn as nn


class HSI2DCNN(nn.Module):
    """Three conv blocks (spectral bands = input channels) followed by an MLP head.

    Architecture (patch 11x11, 200 bands):
        conv 3x3 200->32  + BN + ReLU                 -> 32 x 11 x 11
        conv 3x3 32->64   + BN + ReLU + maxpool 2x2   -> 64 x 5 x 5
        conv 3x3 64->128  + BN + ReLU + maxpool 2x2   -> 128 x 2 x 2
        flatten -> linear 512->256 + ReLU + dropout -> linear 256->16
    """

    def __init__(self, in_channels=200, num_classes=16, patch_size=11, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Infer the flattened size with a dummy input instead of computing it by hand,
        # so the model works for any patch size.
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, in_channels, patch_size, patch_size)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
