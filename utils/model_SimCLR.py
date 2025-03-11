import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class SimCLR_encoder(nn.Module):
    """
    SimCLR encoder class with a projection head for feature extraction and dimensionality reduction.

    Attributes:
    - f: Sequential layers of the modified ResNet model.
    - g: Projection head with linear layers, batch normalization, and ReLU activation.
    - feature_dim: Dimension of the output features.
    """
    def __init__(self, feature_dim=128):
        """
        Initialize the SimCLR_encoder with the specified feature dimension.

        Parameters:
        - feature_dim: Dimension of the output features (default is 128).
        """
        super(SimCLR_encoder, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # Encoder
        self.f = nn.Sequential(*self.f)
        # Projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Forward pass through the SimCLR encoder and projection head.

        Parameters:
        - x: Input tensor with shape [batch_size, channels, height, width].

        Returns:
        - Normalized feature output with shape [batch_size, feature_dim].
        """
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1)
