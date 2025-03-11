import os
import clip
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class vit_img_wrap(nn.Module):
    def __init__(self, clip_model='ViT-L/14', device='cpu', center=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
        super().__init__()

        self.model, self.preprocess = clip.load(clip_model, device)
        self.name = '-'.join(clip_model.split('/'))
        self.device = device
        self.dim = self.model.text_projection.shape[1]
        self.inv_normalize = _transform(self.model.visual.input_resolution, center, std)

    def forward(self, image):

        image = self.inv_normalize(image)
        with torch.no_grad():
            image_features = self.model.encode_image(image)

        return image_features.float()

class Adapter(nn.Module):
    """
    An adapter module for transforming feature dimensions.

    Attributes:
    - fc: A sequential neural network for feature transformation.
    """
    def __init__(self, dim):
        """
        Initialize the adapter with the given feature dimension.

        Parameters:
        - dim: The dimension of the input and output features.
        """
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Softplus(),
            nn.Linear(dim, dim),
            nn.Softplus(),
        )

    def forward(self, x):
        """
        Forward pass through the adapter.

        Parameters:
        - x: The input features.

        Returns:
        - x: The transformed features.
        """
        x = self.fc(x)
        return x


class clip_img_adapter(nn.Module):
    """
    A class combining the CLIP image encoder and the adapter.

    Attributes:
    - clip_encoder: The CLIP image encoder.
    - adapter: The adapter module for feature transformation.
    - device: The device to run the model on.
    """
    def __init__(self, device='cuda'):
        """
        Initialize the CLIP image adapter with the given device.

        Parameters:
        - device: The device to run the model on.
        """
        super().__init__()

        self.clip_encoder = vit_img_wrap(clip_model='ViT-L/14', device=device)
        self.adapter = Adapter(dim=768)
        self.device = device
        self.clip_encoder.to(device)
        self.clip_encoder.eval()
        self.adapter.to(device)
        self.adapter.eval()

    def forward(self, image):
        """
        Forward pass through the CLIP image encoder and the adapter.

        Parameters:
        - image: The input image.

        Returns:
        - feature: The transformed image features.
        """
        with torch.no_grad():
            feature = self.clip_encoder(image)
            feature = self.adapter(feature)

        return feature

def _transform(n_px, center=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """
    Create a composition of transformations for preprocessing images.

    Parameters:
    - n_px: The size to resize the images to.
    - center: The mean values for normalization.
    - std: The standard deviation values for normalization.

    Returns:
    - A Compose object containing the transformations.
    """
    return Compose([
        Normalize(mean=[-center[0] / std[0], -center[1] / std[1], -center[2] / std[2]], std=[1 / std[0], 1 / std[1], 1 / std[2]]),
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


