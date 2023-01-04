import random
import shutil

import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
import os

def data_size_show(data_dir) :
    x_plt = []
    y_plt = []
    for directory in os.listdir(data_dir) :
        x_plt.append(directory)
        y_plt.append(len(os.listdir(os.path.join(data_dir, directory))))

    # creating the bar plot
    flg, ax = plt.subplots(figsize=(16,16))
    plt.barh(x_plt, y_plt, color="maroon")

    # remove x, y Ticks
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)

    # show top values
    ax.invert_yaxis()

    plt.ylabel('Bark Type')
    plt.xlabel('No. of images')
    plt.title("Bark Texture Dataset")
    plt.show()

# data_dir = "./dataset"
# data_size_show(data_dir)

"""
8 : 1 : 1
data
  train 
    - label folder 
        - image.png
  val 
    - label folder 
         - image.png
  test
    - label folder 
          - image.png
"""

# 1. Renaming the files
for category in os.listdir('./dataset/') :
    i = 1
    for image in os.listdir('./dataset/' + category + "/") :
        os.rename("./dataset/" + category + "/" + image,
                  './dataset/' + category + "/" + category + str(i)
                  + ".png")
        i+=1

# Create Train Test Folder
def create_train_val_test_folder() :
    all_categories = os.listdir("./dataset")

    os.makedirs("./data/train/" , exist_ok=True)
    for category in sorted(all_categories) :
        os.makedirs("./data/train/" + category)
        all_images = os.listdir("./dataset/" + category + "/")
        for image in random.sample(all_images, int(0.8 * len(all_images))) :
            shutil.move("./dataset/" + category + "/" + image,
                        "./data/train/" + category + "/")

    os.makedirs("./data/val/", exist_ok=True)
    for category in sorted(all_categories) :
        os.makedirs("./data/val/" + category)
        all_images = os.listdir("./dataset/" + category + "/")
        for image in random.sample(all_images, int(0.5 * len(all_images))) :
            shutil.move("./dataset/" + category + "/" + image,
                        "./data/val/" + category + "/")

    os.makedirs("./data/test/", exist_ok=True)
    for category in sorted(all_categories) :
        os.makedirs("./data/test/" + category)
        all_images = os.listdir("./dataset/" + category + "/")
        for image in all_images:
            shutil.move("./dataset/" + category + "/" + image,
                        "./data/test/" + category + "/")

