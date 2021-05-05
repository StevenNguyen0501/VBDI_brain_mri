from albumentations.augmentations.transforms import Cutout
from torchvision import transforms
import torch
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import random
import numpy as np
import torchvision.transforms.functional as TF
import random
from albumentations import Compose, Normalize, ShiftScaleRotate
from albumentations.pytorch import ToTensor

class AlbuAugment:
    def __init__(self):
        transformation = []
        transformation += [ShiftScaleRotate(),
                        ]
        self.transform = Compose(transformation)

    def __call__(self, x):
        return self.transform(image=x)['image']

class to_tensor_albu:
    def __init__(self):
        transformation = []
        transformation += [Normalize(),
                           ToTensor()]
        self.transform = Compose(transformation)

    def __call__(self, x):
        return self.transform(image=x)['image']