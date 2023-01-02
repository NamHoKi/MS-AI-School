import os
import glob
from PIL import Image

from torch.utils.data import Dataset

class customDataset(Dataset) :
    def __init__(self, path , transform=None):
        # path -> dataset/train/
        self.all_image_path = glob.glob(os.path.join(path,"*","*.png"))
        self.transform = transform
        self.label_dic = {"cloudy": 0, "desert": 1, "green_area":2, "water":3}
        # self.img_list =[]
        # for img_path in self.all_image_path :
        #     self.img_list.append(Image.open(img_path))
        # 라벨 리스트 저장 !!

    def __getitem__(self, item):
        # img = self.img_list[item]
        # print(img)
        img_path = self.all_image_path[item]
        label_temp = img_path.split("\\")
        label = self.label_dic[label_temp[1]]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.all_image_path)

# test = customDataset("./dataset/train", transform=None)
# for i in test :
#     print(i)