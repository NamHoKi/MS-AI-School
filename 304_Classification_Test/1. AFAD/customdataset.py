import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
# from utils import *

class my_dataset(Dataset) :
    def __init__(self, path, transform=None):
        self.all_data_path = glob.glob(os.path.join(path, '*', '*.png'))
        self.transform = transform

        self.label_dict = {}

        for i, (label) in enumerate(sorted(os.listdir('./dataset/train/'))):
            self.label_dict[label] = i


    def __getitem__(self, item):
        image = self.all_data_path[item]
        label = self.label_dict[image.split('\\')[1]]
        image = cv2.imread(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None :
            image = self.transform(image=image)['image']

        return image, label


    def __len__(self):
        return len(self.all_data_path)
