from collections import OrderedDict
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import gc
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import pytorch
from .randaug import RandAugment, to_tensor_randaug
from .albu import AlbuAugment, to_tensor_albu
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
import torchvision
from torch.utils.data import DataLoader, Subset

class ImageLabelDataset(Dataset):
    def __init__(self, images, label, mode='train', cfg=None):
        self.cfg = cfg
        self.images = images
        self.mode = mode
        assert self.cfg.DATA.TYPE in ("multiclass", "multilabel")
        assert self.mode in ("train", "valid", "test")
        if mode == "train":
            self.dir = self.cfg.DIRS.TRAIN_IMAGES
        elif mode == "valid":
            self.dir = self.cfg.DIRS.VALIDATION_IMAGES
        else:
            self.dir = self.cfg.DIRS.TEST_IMAGES

        self.numpies = True if cfg.DATA.NUMPIES else False

        if self.mode !="test":
            self.label = label
        
        if not self.numpies:
            assert self.cfg.DATA.AUGMENT in ("randaug", "albumentations")
            if self.cfg.DATA.AUGMENT == "randaug":
                self.transform = RandAugment(n=self.cfg.DATA.RANDAUG.N,
                    m=self.cfg.DATA.RANDAUG.M, random_magnitude=self.cfg.DATA.RANDAUG.RANDOM_MAGNITUDE)
                self.to_tensor = to_tensor_randaug()
            elif self.cfg.DATA.AUGMENT == "albumentations":
                self.transform = AlbuAugment()
                self.to_tensor = to_tensor_albu()

            if self.cfg.DATA.CROP.ENABLED:
                self.resize_crop = torchvision.transforms.Compose(
                    [torchvision.transforms.RandomResizedCrop(
                        self.cfg.DATA.IMG_SIZE,scale=self.cfg.DATA.CROP.SCALE,
                        ratio=self.cfg.DATA.CROP.RATIO, interpolation=self.cfg.DATA.CROP.INTERPOLATION)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.mode != "test":
            lb = self.label[idx]
            if self.cfg.DATA.TYPE == "multilabel":
                lb = lb.astype(np.float32)
                if not isinstance(lb, list):
                    lb = [lb]
                lb = torch.Tensor(lb).squeeze(0)
        if self.numpies:
            image = np.load(os.path.join(self.dir, self.images[idx] + ".npy"))
            image = torch.Tensor(image).unsqueeze(0)
            if self.mode == "train" or self.mode == "valid":
                return image, lb
            elif self.mode == "test":
                return image, self.image[idx]
        else:
            image = Image.open(os.path.join(self.dir, self.images[idx] + ".jpg"))
            if self.cfg.DATA.INP_CHANNEL == 3:
                image = image.convert("RGB")
            elif self.cfg.DATA.INP_CHANNEL == 1:
                image = image.convert("L")
            if self.mode == "train" and self.cfg.DATA.CROP.ENABLED:
                image = self.resize_crop(image)
            else:
                image = image.resize(self.cfg.DATA.IMG_SIZE)
            if self.mode == "train":
                if isinstance(self.transform, AlbuAugment):
                    image = np.asarray(image)
                image = self.transform(image)
                image = self.to_tensor(image)
                return image, lb
            elif self.mode == "valid":
                if isinstance(self.transform, AlbuAugment):
                    image = np.asarray(image)
                image = self.to_tensor(image)
                return image, lb
            else:
                if isinstance(self.transform, AlbuAugment):
                    image = np.asarray(image)
                image = self.to_tensor(image)
                return image, self.images[idx]


def make_image_label_dataloader(cfg, mode, images, labels):
    dataset = ImageLabelDataset(images, labels, mode=mode, cfg=cfg)
    if cfg.DATA.DEBUG:
        dataset = Subset(dataset,
                         np.random.choice(np.arange(len(dataset)), 50))
    shuffle = True if mode == "train" else False
    dataloader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE,
                            pin_memory=False, shuffle=shuffle,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader