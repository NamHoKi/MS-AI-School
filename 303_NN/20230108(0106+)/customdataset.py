# pip install petastorm
import os
import glob
import cv2
from torch.utils.data import Dataset

class customDataset(Dataset):
    def __init__(self, path, transform=None):
        self.image_path = glob.glob(os.path.join(path, "*", "*.png"))
        self.transform = transform
        self.label_dict = {}

        for i, (category) in enumerate(os.listdir(".\\dataset\\train\\")):
            self.label_dict[category] = int(i)

    def __getitem__(self, item):
        # image = self.image_list[item]
        # label = self.label_list[item]
        image_path = self.image_path[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_path.split("\\")[-2]
        label = self.label_dict[label]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

    def __len__(self):
        return len(self.image_path)

# if __name__ == "__main__":
#     # customDataset(".\\data\\train\\")
#     test = customDataset(".\\dataset\\train\\", transform=None)
#     for i in test:
#         pass