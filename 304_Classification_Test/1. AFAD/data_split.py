import glob
import os
from sklearn.model_selection import train_test_split
import cv2
import shutil


def data_split(age, sex):
    age = age + '*'
    image_path = f"./AFAD_remove/{age}/{sex}/"
    image = glob.glob(os.path.join(image_path, "*.png"))

    train_list, val_list = train_test_split(image, test_size=0.15, train_size=0.85, random_state=777)
    # val_data, test_data = train_test_split(val_list, test_size=0.5, random_state=777)
    # print(len(acacia_train_list), len(acacia_val_data), len(acacia_test_data))

    return train_list, val_list

def data_save(data, mode, folder_name):
    for path in data:
        # # 0. 폴더명 구하기
        # folder_name = path.split("\\")[2]

        # 1. 폴더 구성
        folder_path = f".\\dataset\\{mode}\\{folder_name}\\"
        os.makedirs(folder_path, exist_ok=True)

        # 2. 이미지 이름 구하기
        image_name = path.split('\\')[3].split('.')[0]


        # 3. 복사 후 옮기기
        dpath = os.path.join(folder_path, image_name + '.png')
        shutil.copy(path, dpath)

    print(mode, folder_name, "OK")

# data = glob.glob(os.path.join('./AFAD-Full','2*','*','*.jpg'))
# print(data[1].split('\\')[3].split('.')[0])

# age_list = ['2030','4050']
age_list = ['6090']
for age in age_list :
    # detail_age = [age[0], age[2]]
    detail_age = ['6', '7', '8', '9']
    for a in detail_age :
        # 남자
        train_data, val_data = data_split(a, '111')
        data_save(train_data, 'train', 'm' + age)
        data_save(val_data, 'val', 'm'+age)

        # 여자
        train_data, val_data = data_split(a, '112')
        data_save(train_data, 'train', 'w' + age)
        data_save(val_data, 'val', 'w' + age)

        print(a+'0 - 완료')
