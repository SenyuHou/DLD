import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from torchvision import transforms

# Maximum parameter value for transformations
PARAMETER_MAX = 10

# Function to apply auto contrast to an image
def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)

# Function to adjust brightness of an image
def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)

# Function to adjust color balance of an image
def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

# Function to adjust contrast of an image
def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)

# Function to apply cutout augmentation to an image
def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)

# Function to apply cutout with an absolute size
def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    color = (127, 127, 127)  # gray color
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

# Function to equalize the histogram of an image
def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)

# Function that returns the image without any changes
def Identity(img, **kwarg):
    return img

# Function to invert the colors of an image
def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)

# Function to posterize an image by reducing the number of bits for each color channel
def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)

# Function to rotate an image
def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)

# Function to adjust sharpness of an image
def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

# Function to shear an image along the X axis
def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

# Function to shear an image along the Y axis
def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

# Function to solarize an image
def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)

# Function to solarize an image with an additional value added to each pixel
def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

# Function to translate an image along the X axis
def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

# Function to translate an image along the Y axis
def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

# Helper function to convert a parameter to a float
def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

# Helper function to convert a parameter to an int
def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

# Function to return a pool of augmentation operations for FixMatch
def fixmatch_augment_pool():
    augs = [
        (AutoContrast, None, None),
        (Brightness, 0.9, 0.05),
        (Color, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Equalize, None, None),
        (Identity, None, None),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 0.9, 0.05),
        (ShearX, 0.3, 0),
        (ShearY, 0.3, 0),
        (Solarize, 256, 0),
        (TranslateX, 0.3, 0),
        (TranslateY, 0.3, 0)
    ]
    return augs

# Function to return a pool of weak augmentation operations
def weak_augment_pool():
    augs = [
        (Contrast, 1.8, 0.1),
        (Cutout, 0.1, 0),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 1.8, 0.1),
    ]
    return augs

# Function to return a pool of new augmentation operations
def new_augment_pool():
    augs = [
        (AutoContrast, None, None),
        (Brightness, 1.8, 0.1),
        (Color, 1.8, 0.1),
        (Contrast, 1.8, 0.1),
        (Cutout, 0.1, 0),
        (Equalize, None, None),
        (Invert, None, None),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 1.8, 0.1),
        (ShearX, 0.1, 0),
        (ShearY, 0.1, 0),
        (Solarize, 256, 0),
        (SolarizeAdd, 110, 0),
        (TranslateX, 0.05, 0),
        (TranslateY, 0.05, 0)
    ]
    return augs

# Class for random augmentation without geometric transformations
class RandAugmentwogeo(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = new_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        size = img.size[0]
        img = CutoutAbs(img, int(size * 0.15))
        return img

# Class for weak and strong augmentation for CIFAR-10 dataset
class TransformFixMatch_CIFAR10(object):
    def __init__(self, mean, std, n=2, m=5):
        self.n = n
        self.m = m
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
            RandAugmentwogeo(n=self.n, m=self.m),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

# Class for weak and strong augmentation for Clothing dataset
class TransformFixMatchCloth(object):
    def __init__(self, mean, std, n=2, m=10):
        self.n = n
        self.m = m
        self.weak = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.strong = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugmentwogeo(n=self.n, m=self.m),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

# Class for weak and strong augmentation for Medium dataset
class TransformFixMatchMedium(object):
    def __init__(self, mean, std, n=2, m=10):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64),
            RandAugmentwogeo(n=n, m=m),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

# Class for weak and strong augmentation for Web dataset
class TransformFixMatchWeb(object):
    def __init__(self, mean, std, n=2, m=10):
        self.n = n
        self.m = m
        self.weak = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.strong = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugmentwogeo(n=self.n, m=self.m),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
