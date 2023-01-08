import glob
import os
from sklearn.model_selection import train_test_split
import cv2

def data_split(name):
    image_path = f".\\data\\{name}"
    image = glob.glob(os.path.join(image_path, "*.png"))

    train_list, val_list = train_test_split(image, test_size=0.2, random_state=777)
    val_data, test_data = train_test_split(val_list, test_size=0.5, random_state=777)
    # print(len(acacia_train_list), len(acacia_val_data), len(acacia_test_data))

    return train_list, val_data, test_data

def data_save(data, mode):
    for path in data:
        # 0. 폴더명 구하기
        folder_name = path.split("\\")[2]

        # 1. 폴더 구성
        folder_path = f".\\dataset\\{mode}\\{folder_name}"
        os.makedirs(folder_path, exist_ok=True)

        # 2. 이미지 이름 구하기
        image_name = path.split("\\")[-1]
        image_name = image_name.split(".")[0]

        # 4. 이미지 읽기
        img = cv2.imread(path)

        # 5. 이미지 저장
        # print(os.path.join(folder_path, image_name + ".jpg"))
        cv2.imwrite(os.path.join(folder_path, image_name + ".png"), img)
        # print(folder_name)
    print("작성 완료")

list = []
data = glob.glob(os.path.join(".\\data\\", "*"))
for path in data:
    list.append(os.path.basename(path))
print(list)

for i in list:
    train_list, val_data, test_data = data_split(name=i)

    print(i,"의 개수", "train 개수=", len(train_list), "val 개수=", len(val_data), "test 개수=", len(test_data))
    print()
    print(f"{i} 첫번째 실행")
    data_save(train_list, mode = "train")
    print()
    print(f"{i} 두번째 실행")
    data_save(val_data, mode="val")
    print()
    data_save(val_data, mode = "val")
    print()
    print(f"{i }세번째 실행")
    data_save(test_data, mode = "test")
    print()