from customdataset import customdataset

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import rexnetv1
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import  torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from utils import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = A.Compose([
    A.SmallestMaxSize(max_size = 256),
    A.Resize(height=224, width=224),
    A.RandomShadow(p=0.5),
    A.RandomFog(p=0.4),
    A.RandomSnow(p=0.4),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                       rotate_limit=15, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std =(0.229,0.224,0.225)),
    ToTensorV2()
])
val_transform = A.Compose([
    A.SmallestMaxSize(max_size = 256),
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()

])

# data set
train_dataset = customdataset("./data/train/", transform=train_transform)
val_dataset = customdataset("./data/val/", transform=val_transform)
# data loader
train_loader = DataLoader(train_dataset,batch_size=126, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=126, shuffle=False)

# Pretrained start
# model = rexnetv1.ReXNetV1()
# model.load_state_dict(torch.load("./rexnetv1_1.0.pth"))
# model.output[1] = nn.Conv2d(1280, 50, kernel_size=1, stride=1)
# model.to(device)

# Pretrained no start
model = rexnetv1.ReXNetV1(classes=50)
model.to(device)

criterion = LabelSmoothingCrossEntropy()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
save_dir = "./"

num_epoch = 100

# train(num_epoch, model, train_loader, val_loader, criterion, optimizer,
#           save_dir, device):
train(num_epoch, model, train_loader, val_loader,criterion,optimizer,
      save_dir, device)
