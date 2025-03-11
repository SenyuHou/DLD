import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet_encoder(nn.Module):
    """
    ResNet encoder class with a projection head for feature extraction and dimensionality reduction.

    Attributes:
    - f: Sequential layers of the base ResNet model without the fully connected layer.
    - g: Projection head with linear layers, batch normalization, and ReLU activation.
    - feature_dim: Dimension of the output features.
    """
    def __init__(self, feature_dim=128, base_model='resnet50'):
        """
        Initialize the ResNet_encoder with the specified feature dimension and base model.

        Parameters:
        - feature_dim: Dimension of the output features (default is 128).
        - base_model: The base ResNet model to use ('resnet34' or 'resnet50').
        """
        super(ResNet_encoder, self).__init__()

        # Load the appropriate pre-trained ResNet model
        if base_model == 'resnet34':
            base = models.resnet34(pretrained=True)
            out_features = base.fc.in_features
        elif base_model == 'resnet50':
            base = models.resnet50(pretrained=True)
            out_features = base.fc.in_features
        else:
            raise ValueError("Invalid base model. Choose 'resnet34' or 'resnet50'.")

        # Remove the fully connected layer of the ResNet model
        self.f = nn.Sequential(*(list(base.children())[:-1]))

        # Add a projection head
        self.g = nn.Sequential(
            nn.Linear(out_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Forward pass through the ResNet encoder and projection head.

        Parameters:
        - x: Input tensor with shape [batch_size, channels, height, width].

        Returns:
        - Normalized feature output with shape [batch_size, feature_dim].
        """
        # Forward pass through the base model
        x = self.f(x)  # x shape will be [batch_size, out_features, 1, 1]
        x = torch.flatten(x, 1)  # Flatten to [batch_size, out_features]
        feature = x
        # Forward pass through the projection head
        out = self.g(x)
        # Normalize the feature output before returning
        return F.normalize(feature, dim=-1)

