import os
import glob
import cv2
import random
import shutil

def create_train_val_split_folder(path):
    all_categories = os.listdir(path)
    os.makedirs("./dataset/train/", exist_ok=True)
    os.makedirs("./dataset/val/", exist_ok=True)
    os.makedirs("./dataset/test/", exist_ok=True)

    for category in sorted(all_categories):
        os.makedirs(f"./dataset/train/{category}", exist_ok=True)
        all_image = os.listdir(f"./data/{category}/")
        for image in random.sample(all_image, int(0.8 * len(all_image))):
            # Origin dataset, new dataset
            shutil.move(f"./data/{category}/{image}", f"./dataset/train/{category}/")

    for category in sorted(all_categories):
        os.makedirs(f"./dataset/val/{category}", exist_ok=True)
        all_image = os.listdir(f"./data/{category}/")
        for image in random.sample(all_image, int(0.5 * len(all_image))):
            # Origin dataset, new dataset
            shutil.move(f"./data/{category}/{image}", f"./dataset/val/{category}/")

    for category in sorted(all_categories):
        os.makedirs(f"./dataset/test/{category}", exist_ok=True)
        all_image = os.listdir(f"./data/{category}/")
        for image in all_image:
            # Origin dataset, new dataset
            shutil.move(f"./data/{category}/{image}", f"./dataset/test/{category}/")

if __name__ == "__main__":
    path = "./data"
    # image_size(path)
    create_train_val_split_folder(path)
