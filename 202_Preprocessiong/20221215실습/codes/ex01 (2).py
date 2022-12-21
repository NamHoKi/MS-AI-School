from PIL import Image
from matplotlib import pyplot as plt
import cv2

import numpy as np
import time

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations
from albumentations.pytorch import ToTensorV2
# pip install -U albumentations
# albumentations Data pipeline


class alb_cat_dataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]

        # read an image with opencv
        image = cv2.imread(file_path)

        # BRG -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.st_time = time.time()
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            total_time = (time.time() - self.st_time)

        return image, total_time

    def __len__(self):
        return len(self.file_paths)

        # 기존 torchvision Data pipeline
        # 1. dataset class -> image loader -> transform


class CatDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]

        # 원래 라면 image label
        # Read an image with PIL
        image = Image.open(file_path).convert("RGB")

        # transform time check
        start_tiem = time.time()
        if self.transform:
            image = self.transform(image)
        end_tiem = (time.time() - start_tiem)

        return image, end_tiem

    def __len__(self):
        return len(self.file_paths)


# data aug transforms
# # train
# train_transform = transforms.Compose([
#     transforms.Pad(padding=10),
#     transforms.Resize((256, 256)),
#     transforms.CenterCrop(size=(30)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ColorJitter(brightness=0.2, contrast=0.3),
#     transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5)),
#     transforms.RandomPerspective(distortion_scale=0.7, p=0.5),
#     transforms.ToTensor()
# ])

# # val
# val_transform = transforms.Compose([
#     transforms.Pad(padding=10),
#     transforms.Resize((256, 256)),
#     transforms.CenterCrop(size=(30)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ColorJitter(brightness=0.2, contrast=0.3),
#     transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5)),
#     transforms.ToTensor()

# ])


torchvision_transform = transforms.Compose([
    # transforms.Pad(padding=10),
    # transforms.Resize((256, 256)),
    # transforms.CenterCrop(size=(30)),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.ColorJitter(brightness=0.2, contrast=0.3),
    # transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5)),
    # transforms.RandomPerspective(distortion_scale=0.7, p=0.5),
    # transforms.RandomRotation(degrees=(0, 100)),
    # transforms.RandomAffine(
    #     degrees=(30, 60), translate=(0.1, 0.3), scale=(0.5, 0.7)),
    # transforms.ElasticTransform(alpha=255.0),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.AutoAugment(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

albumentations_transform = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.RandomCrop(224, 224),
    albumentations.HorizontalFlip(),
    albumentations.VerticalFlip(),
    # albumentations.pytorch.transforms.ToTensor(),
    ToTensorV2()
])

albumentations_transform_oneof = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.RandomCrop(224, 224),
    albumentations.RandomRotate90(p=1),
    albumentations.OneOf([
        albumentations.HorizontalFlip(p=.5),
        albumentations.MotionBlur(p=1),
        albumentations.OpticalDistortion(p=1),
        albumentations.GaussNoise(p=1)
    ], p=1),
    ToTensorV2()
])

alb_dataset = alb_cat_dataset(
    file_paths=['cat01.jpeg'], transform=albumentations_transform)

cat_dataset = CatDataset(
    file_paths=["cat01.jpeg"], transform=torchvision_transform)

abl_oneof = alb_cat_dataset(
    file_paths=['cat01.jpeg'], transform=albumentations_transform_oneof
)

# from matplotlib import pyplot as plt
total_time = 0
for i in range(100):
    image, end_ime = cat_dataset[0]
    total_time += end_ime

print("torchvision tiem/image >> ", total_time*10)

alb_total_tiem = 0
for i in range(100):
    alb_image, alb_time = abl_oneof[0]
    alb_total_tiem += alb_time

print("alb time >> ", alb_total_tiem*10)
plt.figure(figsize=(10, 10))
plt.imshow(transforms.ToPILImage()(image).convert("RGB"))
plt.show()

"""
torchvision tiem/image >>  0.8959126472473145
alb time >>  0.14220476150512695
"""
