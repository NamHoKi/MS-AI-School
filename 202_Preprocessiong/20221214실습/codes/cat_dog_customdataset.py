"""
dataset 
    - train
        - cat 
            - cat1.jpg....
        - dog
    - val  
        - cat 
        - dog
    - test
        - cat 
        - dog
"""
from torch.utils.data.dataset import Dataset
import os
import glob


class cat_dog_mycustomdataset(Dataset):
    def __init__(self, data_path):
        # data_path -> ./dataset/train/
        # train -> ./dataset/train/
        # val -> ./dataset/val/
        # test -> ./dataset/test/
        # csv folder 읽기, 변환 할당, 데이터 필터링 등 과 같은 초기 논리가 발생
        self.all_data_path = glob.glob(os.path.join(data_path, '*', '*.jpg'))
        print(self.all_data_path)
        pass

    def __getitem__(self, index):
        # 데이터 레이블 반환 image, label
        pass

    def __len__(self):
        # 전체 데이터 길이 반환
        pass


test = cat_dog_mycustomdataset("./dataset/train/")

for i in test:
    pass
