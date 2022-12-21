from torch.utils.data.dataset import Dataset
from torchvision import transforms
label_dic = {"cat": 0, "dog": 1}


class MyCustomDataset(Dataset):
    def __init__(self, path, transforms=None):
        # data path
        self.all_data_path = "./image/*.jpg"
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.all_data_path[index]
        # "image01.png , image02.png , image03.png ......"
        label_temp = image_path.split("/")
        label_temp = label_temp[2]
        label_temp = label_temp.replace(".jpg", "")
        label = label_dic[label_temp]
        image = cv2.imread(image_path)

        if self.transforms is not None:
            image = self.transforms(image)

        return filename, bbox
        return image, label

    def __len__(self):
        return len(self.all_data_path)


temp = MyCustomDataset("./dataset")

for i in temp:
    print(i)
    # image01 xywh
