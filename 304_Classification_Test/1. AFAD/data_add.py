import glob
import os
from sklearn.model_selection import train_test_split
import cv2
import shutil

def data_move(path) :
    all_image_path = glob.glob((os.path.join(path, '*.jpg')))

    for image_path in all_image_path :
        temp = image_path.split('_')
        age, sex, data = temp[0].split('\\')[1], temp[1], temp[-1].split('.')[0]
        # print(age, sex, data)
        if int(age) >= 40 :
            if sex == '0' :
                sex = '111'
            else:
                sex = '112'
            data_save(image_path, data, age, sex)


def data_save(path, data, age, sex) :
    # 1. 폴더 구성
    folder_path = f".\\AFAD-Full\\{age}\\{sex}\\"
    os.makedirs(folder_path, exist_ok=True)

    # 2. 복사 후 옮기기
    dpath = os.path.join(folder_path, data + '.png')
    shutil.copy(path, dpath)

data = data_move('./part2/')
